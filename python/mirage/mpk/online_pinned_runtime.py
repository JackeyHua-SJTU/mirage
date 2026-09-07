"""
OnlinePinnedRuntime — CPU-side driver for MODE_ONLINE_PINNED persistent kernels.

The kernel uses two lock-free power-of-2 ring buffers backed by pinned
(page-locked) memory so both CPU and GPU can access them without DMA copies.

  Request ring  (CPU→GPU): CPU writes {rid, prompt_len, initial_step} then
                            sets ready=1; GPU drains directly into running batch.
  Completion ring (GPU→CPU): GPU writes {rid, buffer_row, final_step} then
                              sets ready=1; CPU polls, collects, clears to 0.

Both rings carry the OIPL page-lifecycle payload: a completion also reports the
request's final KV page list, and a request also carries the prefix pages the
CPU wants imported.  KV page ownership is inverted — the GPU allocates pages
but never releases them, so every exported page is returned to the GPU by the
CPU, through :attr:`_page_return_ring`.  With the prefix cache enabled the
return is no longer a pass-through: a completed request's frozen prompt pages
are indexed instead of returned, and a later request that shares that prompt
prefix imports them and skips the prefill.

Each ring slot has its own independent pinned inbox buffer, so concurrent
submits can write prompt tokens without overwriting each other.  The GPU
copies inbox tokens to the assigned buffer row when admitting a request.

A single *owner thread* drives both rings.  :meth:`submit` only appends the
request to a CPU-side admission deque; every owner
tick then reaps completions (page accounting, cache insert, page return),
trims the cache, and publishes as many queued requests as the ring has free
slots.  Request threads never touch the request ring, so the ring publish and
the match/refcount/prefix-array write that must be atomic with it happen on one
thread with no lock hierarchy to get wrong.  A queued request holds nothing --
the match happens at the publish moment, not at enqueue -- so it can never pin
pages the ring head needs.  The price is up to one tick (0.2 ms) of admission
latency, which is noise next to a kernel iteration.

Usage::

    runtime = OnlinePinnedRuntime(mpk)
    runtime.submit(rid=0, token_ids=[...])
    runtime.submit(rid=1, token_ids=[...])
    buffer_row, final_step = runtime.wait_for_request(rid=0, timeout=60)
    tokens = runtime.read_tokens_at_row(buffer_row, final_step)

Without :meth:`start` there is no owner thread, and :meth:`wait_for_request`
drives the ticks itself, so the snippet above works single-threaded too.
"""

import collections
import logging
import threading
import time
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import torch

from .persistent_kernel import max_pages_per_request
from .prefix_cache import EMPTY_MATCH, Block, PrefixCache, PrefixMatch

logger = logging.getLogger(__name__)

# Owner ticks (≈0.2 ms each) the oldest published request may stay unadmitted
# before the cache starts handing the page pool back one block at a time --
# and then only if the pool is what it is waiting for (§6.4 v2.1-c′).  50 ticks
# ≈ 10 ms is an order of magnitude above one kernel iteration, so a merely busy
# GPU never trips it.
PREFIX_CACHE_STALL_TICKS = 50


@dataclass
class _Published:
    """A request sitting on the request ring: what it pinned, what it will cache."""

    rid: int
    slot: int
    npp: int
    blocks: Tuple[Block, ...]
    prompt_ids: Optional[List[int]]


class OnlinePinnedRuntime:
    """CPU-side helper for the online_pinned persistent kernel mode."""

    def __init__(
        self,
        mpk,
        enable_prefix_cache: bool = False,
        prefix_cache_pages: int = 0,
        gen_tail_len: int = 0,
    ):
        assert mpk.metadata.mode == "online_pinned", (
            f"OnlinePinnedRuntime requires mode='online_pinned', got {mpk.metadata.mode}"
        )
        self._mpk = mpk
        self._cap = mpk.pinned_ring_capacity
        self._mask = self._cap - 1
        self._total_inflight = mpk.total_num_requests

        # Pinned CPU↔GPU ring arrays (allocated in mpk.py, shape=[cap]).
        #
        # A buffer the owner tick *writes* is held as a numpy view instead of
        # a tensor.  The view aliases the same page-locked memory --
        # the GPU cannot tell the difference -- but `tensor[i] = v` costs a full
        # ATen dispatch (~4 us measured), and the owner tick writes a prefix
        # array per publish plus up to MAX_EVICTIONS_PER_TICK page ids per
        # return, on the thread that also drives admission.  The read side
        # already batches through a tensor slice + tolist(); this gives the
        # write side the same treatment.  Only the view is kept, so there is no
        # second, slower way to write the same buffer -- and the view owns the
        # storage, so the tensor needs no separate reference.
        self._req_ready         = mpk.pinned_req_ready        # int32, pinned
        self._req_request_id    = mpk.pinned_req_request_id.numpy()
        self._req_prompt_len    = mpk.pinned_req_prompt_len.numpy()
        self._req_initial_step  = mpk.pinned_req_initial_step.numpy()
        self._comp_ready        = mpk.pinned_comp_ready       # int32, pinned
        self._comp_request_id   = mpk.pinned_comp_request_id  # int32, pinned
        self._comp_buffer_row   = mpk.pinned_comp_buffer_row  # int32, pinned
        self._comp_final_step   = mpk.pinned_comp_final_step  # int32, pinned
        self._shutdown          = mpk.pinned_shutdown         # int32[1], pinned
        self._pinned_step       = mpk.pinned_step             # int32[max_batched], pinned
        self._inbox_tokens      = mpk.pinned_inbox_tokens     # int64[cap, max_seq_len], pinned
        self._pinned_rid_at_row = mpk.pinned_rid_at_row       # int32[max_batched], pinned

        # OIPL page-lifecycle channels.  The CPU is the only page releaser: it
        # takes each completed request's exported page list and writes it into
        # the return ring, where the GPU's Step 0 drains it back into the free
        # queue.  With no prefix cache the return is a pass-through, and the
        # import channel stays inert -- num_prefix_pages is always 0.
        self._comp_num_pages       = mpk.pinned_comp_num_pages
        self._comp_pages           = mpk.pinned_comp_pages
        self._req_num_prefix_pages = mpk.pinned_req_num_prefix_pages.numpy()
        self._req_prefix_pages     = mpk.pinned_req_prefix_pages.numpy()
        self._page_return_ring     = mpk.pinned_page_return_ring.numpy()
        self._page_return_tail     = mpk.pinned_page_return_tail
        self._page_return_head_mirror = mpk.pinned_page_return_head_mirror
        self._page_free_count_mirror  = mpk.pinned_page_free_count_mirror

        self._page_size = mpk.metadata.page_size
        self._max_num_pages = mpk.max_num_pages
        self._pages_per_req = max_pages_per_request(
            mpk.max_seq_length, self._page_size)
        self._return_mask = self._page_return_ring.shape[0] - 1
        # CPU-private tail of the page-return ring; the pinned copy is what the
        # GPU reads.
        self._cpu_return_tail = 0
        # Page-lifecycle counters, read by tests through `page_stats`.
        self._page_stats: Dict[str, int] = collections.Counter()

        # CPU-private ring cursors.
        self._cpu_req_tail  = 0  # next ring slot to write
        self._cpu_req_ack   = 0  # last slot known to be consumed by GPU
        self._cpu_comp_head = 0

        # CPU-side admission queue: every submitted request lands here first
        # and is published by the owner thread.  Holding a request here costs
        # nothing -- no ring slot, no page pin -- so a queued request can never
        # wedge admission.
        self._admission: Deque[
            Tuple[int, torch.Tensor, int, Optional[List[int]]]
        ] = collections.deque()
        self._admission_lock = threading.Lock()

        # Prefix cache and its owner-thread-private bookkeeping.  ``_published``
        # holds what each published request pinned until its completion is
        # reaped; ``_unadmitted`` is the oldest-first view of it the pin budget
        # (v2.1-a) and the stall detector (v2.1-c) need.
        self._cache: Optional[PrefixCache] = None
        if enable_prefix_cache:
            self._cache = PrefixCache(
                page_size=self._page_size,
                total_pages=self._max_num_pages,
                max_pages_per_req=self._pages_per_req,
                capacity_pages=prefix_cache_pages or None,
                stall_ticks=PREFIX_CACHE_STALL_TICKS,
            )
        self._gen_tail_len = gen_tail_len
        self._published: Dict[int, _Published] = {}
        self._unadmitted: Deque[_Published] = collections.deque()

        # Dedicated stream for HtoD / DtoH copies.
        self._write_stream = torch.cuda.Stream(device=mpk.tokens.device)

        # Serialises one owner tick against another.  The owner thread is
        # normally the only ticker, but a driver running without one polls
        # through drain_completions(); this lock keeps the two strictly
        # exclusive so the request ring and the page-return ring keep their
        # single producer either way.
        self._owner_lock = threading.Lock()

        # Completion tracking: rid → (buffer_row, final_step)
        self._completions: Dict[int, Tuple[int, int]] = {}
        self._abandoned: set[int] = set()
        # rid → the exception that stopped it from ever being published.
        self._failed: Dict[int, Exception] = {}
        self._lock = threading.RLock()

        # The owner thread starts only after reset() has initialized all shared
        # state. Starting it here races reset() and can corrupt ring cursors.
        self._owner_stop = threading.Event()
        self._owner_thread: threading.Thread | None = None
        # Fail-closed latch. Named for the drain half of the owner tick it has
        # always guarded: an owner that dies still makes every later caller --
        # submit included -- raise at once instead of queueing into a void.
        self._drain_error: Exception | None = None

    # ── Public API ────────────────────────────────────────────────────────

    def submit(self, rid: int, token_ids: torch.Tensor, initial_step: int = 0) -> None:
        """Queue a request for the owner thread to publish.

        The submitting thread does not touch the request ring: it clones the
        prompt tokens (the caller may reuse its tensor) and appends them to the
        admission deque.  The next owner tick copies them into the slot's inbox
        and publishes the ring entry, so admission is delayed by at most one
        tick.

        Raises if the owner thread has died, so a dead owner fails submissions
        loudly rather than accepting requests nothing will ever publish.

        Parameters
        ----------
        rid          : unique request identifier (never repeats)
        token_ids    : 1-D int64 tensor of token IDs (CPU or CUDA)
        initial_step : starting decode step; leave at 0 to let the prefix cache
                       pick one.  A caller-chosen start disables matching for
                       that request, since the two cannot both set it.
        """
        self._raise_drain_error()
        # The prefix cache keys on plain ints, and this is the last thread that
        # can pay for the conversion without stealing owner-tick time.
        prompt_ids = token_ids.tolist() if self._cache is not None else None
        with self._admission_lock:
            self._admission.append(
                (rid, token_ids.clone(), initial_step, prompt_ids))

    def _publish_admissions(self) -> int:
        """Match, pin and publish queued requests while the ring has free slots.

        Owner-serial: the caller holds ``_owner_lock``, which makes this the
        request ring's single producer, so a slot can never be claimed twice
        and the ring can never grow a hole.  (The GPU stops draining at the
        first ``ready == 0`` slot, so a hole would block everything behind it
        forever.)  Returns the number of requests published.
        """
        published = 0
        while True:
            slot = self._cpu_req_tail & self._mask
            if self._load_i32_acquire(self._req_ready, slot) != 0:
                break  # ring full: the GPU has not drained this slot yet
            with self._admission_lock:
                if not self._admission:
                    break
                entry = self._admission.popleft()
            rid, token_ids, initial_step, prompt_ids = entry
            match = self._match_at_publish(prompt_ids, initial_step)
            if not self._staging_room(match.num_prefix_pages):
                # The GPU would refuse this request at its admission gate and
                # the slot would sit ready until it stopped refusing.  Put the
                # request back in the queue, where it pins nothing, and let the
                # cache tick hand the pool back instead.
                self._release_match(match)
                with self._admission_lock:
                    self._admission.appendleft(entry)
                self._page_stats["publish_holds"] += 1
                break
            if match.num_prefix_pages:
                # v2.1-(0): the kernel reads the prefill start from its own
                # channel, so the two are derived together, here, or Step 4
                # skips a prefill it has no imported pages for.
                initial_step = match.num_prefix_pages * self._page_size
            try:
                self._publish_request_locked(
                    slot, rid, token_ids, initial_step, match.page_ids)
            except Exception as exc:
                # A request the CPU cannot stage (an over-long prompt, say)
                # must not take the owner thread down with it: record the
                # failure against its rid so its waiter raises, drop the pins it
                # will now never export, and carry on with the rest of the
                # queue.  The slot is untouched.
                self._release_match(match)
                self._fail_request(rid, exc)
                continue
            # Per-publish hit rate.  The cache's own hits/misses count match
            # attempts, and a request the watermark holds is matched again on
            # the next tick, so only these two are one-per-request.
            if match.num_prefix_pages:
                self._page_stats["publish_hits"] += 1
                self._page_stats["imported_pages"] += match.num_prefix_pages
            else:
                self._page_stats["publish_misses"] += 1
            info = _Published(rid, slot, match.num_prefix_pages,
                              match.blocks, prompt_ids)
            self._published[rid] = info
            self._unadmitted.append(info)
            self._cpu_req_tail += 1
            published += 1
        return published

    def _match_at_publish(
        self, prompt_ids: Optional[List[int]], initial_step: int
    ) -> PrefixMatch:
        """Longest cached prefix of this request, pinned, clamped to the budget.

        Match and pin happen inside the publish loop rather than at submit
        time: a queued request that already held refcounts could pin the pages
        the ring head needs to be admitted and wedge the pool (§10-3).
        """
        if self._cache is None or prompt_ids is None or initial_step != 0:
            return EMPTY_MATCH
        budget = self._cache.publish_pin_budget(
            [info.npp for info in self._unadmitted])
        return self._cache.match(prompt_ids, budget)

    def _release_match(self, match: PrefixMatch) -> None:
        """Give back pins taken for a publish that did not happen."""
        if self._cache is not None and match.blocks:
            self._cache.release(match.blocks)

    def _staging_room(self, num_prefix_pages: int) -> bool:
        """Advisory watermark: can the GPU still admit one more publish?

        The estimate is the free-page mirror plus what is still in flight in
        the return ring.  Both mirrors are published together at the end of a
        GPU iteration, so the in-flight pages are not double counted, but the
        estimate ignores the pages the ledger has reserved for already admitted
        requests -- it is an upper bound, and the GPU's admission gate (§6.3a)
        is the safety net, not this (§10-1).  With nothing published there is
        nothing to wait for and no stall for the cache tick to react to, so the
        head request always goes out.
        """
        if self._cache is None or not self._unadmitted:
            return True
        estimate = self.gpu_free_page_count + self.pending_page_returns
        return estimate >= self._pages_per_req - num_prefix_pages

    def _fail_request(self, rid: int, exc: Exception) -> None:
        """Record that *rid* could never be published; its waiter will raise."""
        logger.exception("failed to publish rid=%s", rid)
        with self._lock:
            self._failed[rid] = exc

    def _publish_request_locked(
        self,
        slot: int,
        rid: int,
        token_ids: torch.Tensor,
        initial_step: int,
        prefix_pages: Sequence[int] = (),
    ) -> None:
        """Copy and publish one request from the owner tick.

        *prefix_pages* are the cached pages the GPU should import in place of
        allocating fresh ones; *initial_step* must be ``len(prefix_pages) *
        page_size`` (v2.1-(0)).  Everything is written before the single
        release-store of ``ready``, which is what makes the whole payload --
        prefix array included -- visible to Step 4 as one unit.
        """
        prompt_len = token_ids.shape[0]
        with torch.cuda.stream(self._write_stream):
            self._inbox_tokens[slot, :prompt_len].copy_(
                token_ids, non_blocking=True)
        self._write_stream.synchronize()
        self._req_request_id[slot] = rid
        self._req_prompt_len[slot] = prompt_len
        self._req_initial_step[slot] = initial_step
        npp = len(prefix_pages)
        base = slot * self._pages_per_req
        self._req_prefix_pages[base:base + npp] = prefix_pages
        self._req_num_prefix_pages[slot] = npp
        self._store_i32_release(self._req_ready, slot, 1)

    def drain_completions(self) -> List[Tuple[int, int, int]]:
        """Run one owner tick out of band and report the newly completed requests.

        The owner thread ticks on its own; this is the entry point for a driver
        polling the runtime without one (see the module docstring).  It takes
        ``_owner_lock``, so such a caller is strictly serialised with the owner
        thread rather than racing it for ring slots.

        Returns a list of ``(rid, buffer_row, final_step)`` tuples for
        requests that have finished since the last tick.
        """
        self._raise_drain_error()
        with self._owner_lock:
            return self._owner_tick()

    def _owner_tick(self) -> List[Tuple[int, int, int]]:
        """One pass of the owner thread: observe, reap, trim, publish.

        The caller holds ``_owner_lock``.  Observing admissions first keeps
        ``_unadmitted`` a subset of ``_published``, which the reap then pops
        from; completions come before publishes so the pages and ring slots
        they free are available to the requests published in the same tick.
        """
        self._observe_admissions()
        finished = self._reap_completions()
        self._cache_tick()
        self._publish_admissions()
        return finished

    def _observe_admissions(self) -> None:
        """Retire published requests the GPU has taken off the ring.

        The GPU drains the request ring in FIFO order and clears ``ready`` at
        the end of the admit body, so ``ready == 0`` on the oldest published
        slot means exactly "that request was admitted".  No extra channel is
        needed for the CPU to see admission (§6.4 v2.1-a).
        """
        while self._unadmitted:
            if self._load_i32_acquire(
                    self._req_ready, self._unadmitted[0].slot) != 0:
                break
            self._unadmitted.popleft()

    def _cache_tick(self) -> None:
        """Trim the cache to budget and react to a *page-starved* ring head.

        Resident pages hold the GPU's ``avail_uncommitted`` down 1:1 and the
        GPU has no "I am starving" channel, so a cache that never reacts to an
        unadmitted ring head can park admission indefinitely without breaking
        any invariant (§6.4 v2.1-c).  But a head can also sit unadmitted because
        every row is busy, and evicting cannot buy a row: P2 measured 1061
        backpressure ticks in 0.8 s with 47-56 of 64 pages free, which cost 163
        blocks and bought nothing.  So the stall is paired with an attribution
        the owner can compute from the mirrors it already reads (§6.4 v2.1-c′).

        ``head_worst_need`` is the whole ``max_pages_per_req``, not
        ``max_pages_per_req - npp_head``, even though the head's npp is known
        here: ``free_estimate`` is an upper bound on ``avail_uncommitted`` (it
        cannot see the reservations held by admitted requests), so it is paired
        with an upper bound on the need.  An attribution that is too tight would
        suppress a genuinely page-starved stall and wedge admission; one that is
        too loose only lets a redundant eviction through.
        """
        if self._cache is None:
            return
        oldest = None
        free_estimate = None
        if self._unadmitted:
            oldest = self._unadmitted[0].rid
            free_estimate = self.gpu_free_page_count + self.pending_page_returns
        self._return_pages_locked(self._cache.tick(
            oldest,
            free_estimate=free_estimate,
            head_worst_need=self._pages_per_req,
        ))

    def _reap_completions(self) -> List[Tuple[int, int, int]]:
        """Collect every completion the GPU has published since the last tick."""
        finished = []
        while True:
            with self._lock:
                slot = self._cpu_comp_head & self._mask
                if self._load_i32_acquire(self._comp_ready, slot) == 0:
                    break
                rid         = int(self._comp_request_id[slot].item())
                buffer_row  = int(self._comp_buffer_row[slot].item())
                final_step  = int(self._comp_final_step[slot].item())
                # Page accounting is unconditional: the pages of an abandoned
                # request must be cached or come back just like any other --
                # only session delivery is conditional.  It runs before the
                # ready clear, while the exported ids are still the GPU's
                # published payload for this slot.
                self._account_pages_locked(
                    rid, final_step,
                    self._check_exported_pages(slot, rid, final_step))
                if rid in self._abandoned:
                    self._release_row_locked(rid, buffer_row)
                    self._abandoned.remove(rid)
                else:
                    self._completions[rid] = (buffer_row, final_step)
                # Do not make the slot reusable until completion bookkeeping
                # and any abandoned-row release both succeed.
                self._store_i32_release(self._comp_ready, slot, 0)
                self._cpu_comp_head += 1
            finished.append((rid, buffer_row, final_step))
        return finished

    def _account_pages_locked(
        self, rid: int, final_step: int, pages: List[int]
    ) -> None:
        """Cache what this completion froze; return everything else.

        The exported pages are the request's whole page table, so page ``j``
        holds prompt tokens ``[j*PS, (j+1)*PS)``.  Below ``npp`` they are the
        pages the request imported -- their pin drops here; above it they are
        fresh, and the ones entirely inside the cacheable prompt range extend
        the chain.  The rest go to the return ring, as they always did.
        """
        info = self._published.pop(rid, None)
        if self._cache is None or info is None or info.prompt_ids is None:
            self._return_pages_locked(pages)
            return
        # An export the CPU could not read (a page-count anomaly) leaves this
        # request's pins with no page to drop them against; drop them here or
        # the blocks stay pinned for the life of the cache.
        exported = set(pages)
        stranded = [b for b in info.blocks if b.page_id not in exported]
        self._return_pages_locked(self._cache.insert(
            info.prompt_ids, pages, final_step, self._gen_tail_len,
            num_prefix_pages=info.npp))
        self._cache.release(stranded)

    def _check_exported_pages(self, slot: int, rid: int, final_step: int) -> List[int]:
        """Reconcile one completion's exported KV page list.

        The GPU exports the page table it built for the request's last
        iteration, so its length is fixed by the same arithmetic Step 1 used to
        report ``final_step``: the batch consumed positions up to ``final_step``
        exclusive of the freshly generated token, giving a KV length of
        ``final_step`` and ``ceil(final_step / page_size)`` pages.  Anything
        else means the export and the page table have drifted apart, which is
        exactly the failure the CPU could not see before this channel existed.

        Returns the exported page ids -- the caller returns exactly these to
        the GPU -- and records any disagreement in :attr:`page_stats`; it never
        raises, so a mismatch cannot wedge the drain thread.
        """
        num_pages = int(self._comp_num_pages[slot].item())
        expected = (final_step + self._page_size - 1) // self._page_size
        stats = self._page_stats
        stats["completions"] += 1

        # The bound is the per-slot stride of the export array, so a count
        # outside it has no readable page list at all.
        if not 0 <= num_pages <= self._pages_per_req:
            stats["bad_page_count"] += 1
            self._log_page_anomaly(
                f"rid={rid} exported num_pages={num_pages} outside "
                f"[0, {self._pages_per_req}]")
            return []

        base = slot * self._pages_per_req
        pages = self._comp_pages[base:base + num_pages].tolist()
        stats["exported_pages"] += num_pages

        if num_pages != expected:
            stats["count_mismatch"] += 1
            self._log_page_anomaly(
                f"rid={rid} exported {num_pages} pages, final_step="
                f"{final_step} implies {expected}")
        if any(not 0 <= page < self._max_num_pages for page in pages):
            stats["page_id_out_of_range"] += 1
            self._log_page_anomaly(f"rid={rid} exported page ids {pages}")
        if len(set(pages)) != num_pages:
            stats["duplicate_page"] += 1
            self._log_page_anomaly(f"rid={rid} exported repeated pages {pages}")
        return pages

    def _return_pages_locked(self, pages: List[int]) -> None:
        """Hand *pages* back to the GPU through the page-return ring.

        The owner tick is the ring's single producer -- this is the only
        writer, and ``_owner_lock`` keeps an out-of-band
        :meth:`drain_completions` caller exclusive with the owner thread -- so
        the tail is advanced by one writer at a time.  The ids are written
        first and the tail is release-stored last, pairing with the GPU's
        acquire load in Step 0.

        No occupancy check is needed: a page is in the ring only between its
        export and the GPU's drain, so the ring holds at most ``max_num_pages``
        entries and its capacity is a power of two at least that large.
        """
        if not pages:
            return
        n = len(pages)
        ring = self._page_return_ring
        tail = self._cpu_return_tail
        start = tail & self._return_mask
        # The occupancy argument above bounds one write by the ring capacity,
        # so it wraps at most once and two slices always suffice.  A batch that
        # broke that bound raises a broadcast error here rather than silently
        # overwriting entries the GPU has not drained yet.
        head_len = min(n, ring.shape[0] - start)
        ring[start:start + head_len] = pages[:head_len]
        if head_len < n:
            ring[:n - head_len] = pages[head_len:]
        tail += n
        self._cpu_return_tail = tail
        self._store_i32_release(self._page_return_tail, 0, tail)
        self._page_stats["returned_pages"] += len(pages)

    def _log_page_anomaly(self, message: str) -> None:
        """Log the first page-export anomaly only; the rest are counted."""
        if self._page_stats["anomalies"] == 0:
            logger.warning("page export mismatch: %s", message)
        self._page_stats["anomalies"] += 1

    @property
    def page_stats(self) -> Dict[str, int]:
        """Page-export reconciliation counters, merged with the cache's.

        Cache counters are prefixed ``cache_``.  They live on the owner
        thread's private structures, so this takes ``_owner_lock`` -- after
        releasing ``_lock``, never nested inside it -- and may block a caller
        for up to one tick.
        """
        with self._lock:
            stats = dict(self._page_stats)
        with self._owner_lock:
            stats["published_unadmitted"] = len(self._unadmitted)
            cache_stats = None if self._cache is None else self._cache.stats()
        stats["cache_enabled"] = int(cache_stats is not None)
        if cache_stats is not None:
            stats["cache_capacity_pages"] = self._cache.capacity_pages
            stats.update(("cache_" + key, value)
                         for key, value in cache_stats.items())
        return stats

    @property
    def gpu_free_page_count(self) -> int:
        """Free pages the GPU published at the end of its last iteration."""
        return self._load_i32_acquire(self._page_free_count_mirror, 0)

    @property
    def pending_page_returns(self) -> int:
        """Returned pages the GPU has not drained yet.

        The head mirror lags the GPU's private head by up to one iteration, so
        this is an upper bound.  Together with :attr:`gpu_free_page_count` it
        accounts for every page not currently held by a live request.
        """
        head = self._load_i32_acquire(self._page_return_head_mirror, 0)
        return self._cpu_return_tail - head

    def _owner_loop(self) -> None:
        """Own both rings: reap completions and publish admissions, forever."""
        while not self._owner_stop.is_set():
            try:
                with self._owner_lock:
                    self._owner_tick()
            except Exception as exc:
                with self._lock:
                    self._drain_error = exc
                self._owner_stop.set()
                break
            self._owner_stop.wait(0.0002)

    def _owner_running(self) -> bool:
        """True while the background owner thread is ticking."""
        thread = self._owner_thread
        return thread is not None and thread.is_alive()

    def wait_for_request(
        self,
        rid: int,
        timeout: float = 60.0,
        poll_interval: float = 1e-4,
    ) -> Tuple[int, int]:
        """Block until a specific *rid* completes.

        Returns ``(buffer_row, final_step)``.
        Raises ``TimeoutError`` if the request does not complete in time.
        """
        deadline = time.monotonic() + timeout
        while True:
            if not self._owner_running():
                # No owner thread: this caller drives the ticks itself.
                self.drain_completions()
            with self._lock:
                self._raise_drain_error_locked()
                self._raise_request_error_locked(rid)
                if rid in self._completions:
                    return self._completions[rid]
            if time.monotonic() > deadline:
                self.abandon_request(rid)
                raise TimeoutError(
                    f"wait_for_request timed out for rid={rid}"
                )
            time.sleep(poll_interval)

    def read_tokens_at_row(self, buffer_row: int, final_step: int) -> torch.Tensor:
        """Read tokens [0..final_step] from *buffer_row*.

        Parameters
        ----------
        buffer_row : row index into the token buffer
        final_step : number of tokens available (prompt + generated) - 1
        """
        with torch.cuda.stream(self._write_stream):
            result = self._mpk.tokens[buffer_row, : final_step + 1].clone()
        self._write_stream.synchronize()
        return result

    def read_tokens_range(self, buffer_row: int, start: int, end: int) -> torch.Tensor:
        """Read tokens [start..end] (inclusive) from *buffer_row*."""
        with torch.cuda.stream(self._write_stream):
            result = self._mpk.tokens[buffer_row, start : end + 1].clone()
        self._write_stream.synchronize()
        return result

    def get_completion(self, rid: int) -> Tuple[int, int] | None:
        """Return the completion for *rid*, if the GPU has published it."""
        with self._lock:
            self._raise_drain_error_locked()
            self._raise_request_error_locked(rid)
            return self._completions.get(rid)

    def release_request(self, rid: int) -> bool:
        """Acknowledge that the CPU finished reading a completed row."""
        with self._lock:
            completion = self._completions.get(rid)
            if completion is None:
                return False
            buffer_row, _ = completion
            self._release_row_locked(rid, buffer_row)
            del self._completions[rid]
            return True

    def abandon_request(self, rid: int) -> None:
        """Release *rid* when it completes without retaining its output."""
        with self._lock:
            completion = self._completions.get(rid)
            if completion is None:
                self._abandoned.add(rid)
                return
            buffer_row, _ = completion
            self._release_row_locked(rid, buffer_row)
            del self._completions[rid]

    def _release_row_locked(self, rid: int, buffer_row: int) -> None:
        owner = self._load_i32_acquire(self._pinned_rid_at_row, buffer_row)
        if owner != rid:
            raise RuntimeError(
                f"row {buffer_row} belongs to rid={owner}, expected rid={rid}"
            )
        # -1 is a phase-disjoint CPU release acknowledgement. The GPU keeps
        # the owner mapping and row stable until it observes this value.
        self._store_i32_release(self._pinned_step, buffer_row, -1)

    def get_current_step_at_row(self, buffer_row: int) -> int:
        """Return the latest decode step written by the GPU for *buffer_row*."""
        return self._load_i32_acquire(self._pinned_step, buffer_row)

    def find_row_for_rid(self, rid: int) -> int:
        """Scan ``pinned_rid_at_row`` to find which buffer row holds *rid*.

        Returns the row index, or -1 if the GPU hasn't assigned a row yet.
        """
        for r in range(self._total_inflight):
            if self._load_i32_acquire(self._pinned_rid_at_row, r) == rid:
                return r
        return -1

    @property
    def waiting_count(self) -> int:
        """Number of submitted requests the owner has not published yet."""
        with self._admission_lock:
            return len(self._admission)

    def request_shutdown(self) -> None:
        """Signal the GPU persistent kernel to terminate when it is idle."""
        self._store_i32_release(self._shutdown, 0, 1)

    def stop(self) -> None:
        """Stop the owner thread after the GPU kernel has exited."""
        self._owner_stop.set()
        if self._owner_thread is not None:
            self._owner_thread.join()

    def shutdown(self) -> None:
        """Signal the GPU kernel and stop the owner thread."""
        self.request_shutdown()
        self.stop()

    def start(self) -> None:
        """Start the background owner thread after shared state is reset."""
        if self._owner_running():
            return
        self._raise_drain_error()
        self._owner_stop.clear()
        self._owner_thread = threading.Thread(
            target=self._owner_loop, daemon=True)
        self._owner_thread.start()

    def reset(self) -> None:
        """Clear completion bookkeeping and ring state for a new session."""
        if self._owner_running():
            raise RuntimeError("cannot reset a running online runtime")

        with self._owner_lock:
            self._cpu_req_tail = 0
            self._cpu_req_ack = 0
            with self._admission_lock:
                self._admission.clear()
            self._published.clear()
            self._unadmitted.clear()
            if self._cache is not None:
                # The relaunched kernel refills page_queue with every page, so
                # the resident pages this hands back are already the GPU's; the
                # index simply stops claiming them.
                self._cache.reset()
            self._req_ready.zero_()
            self._req_request_id[:] = 0
            self._req_num_prefix_pages[:] = 0
            self._req_prefix_pages[:] = 0

        with self._lock:
            self._cpu_comp_head = 0
            self._completions.clear()
            self._abandoned.clear()
            self._failed.clear()
            self._drain_error = None
            self._comp_ready.zero_()
            self._comp_num_pages.zero_()
            self._comp_pages.zero_()
            # The relaunched kernel re-initializes its private return-ring head
            # to 0 in init_kernel and refills page_queue with every page, so the
            # producer cursor, the ring and both mirrors restart at 0 too --
            # anything left in the ring from the previous session is a page the
            # new page_queue already owns.
            self._cpu_return_tail = 0
            self._page_return_ring[:] = 0
            self._page_return_tail.zero_()
            self._page_return_head_mirror.zero_()
            self._page_free_count_mirror.zero_()
            self._page_stats.clear()

        self._shutdown.zero_()
        self._pinned_step.zero_()
        self._pinned_rid_at_row.fill_(-1)

    def _load_i32_acquire(self, tensor: torch.Tensor, index: int) -> int:
        return int(self._mpk.persistent_kernel.load_i32_acquire(
            tensor.data_ptr(), index))

    def _store_i32_release(
        self, tensor: torch.Tensor, index: int, value: int,
    ) -> None:
        self._mpk.persistent_kernel.store_i32_release(
            tensor.data_ptr(), index, value)

    def _raise_drain_error(self) -> None:
        with self._lock:
            self._raise_drain_error_locked()

    def _raise_drain_error_locked(self) -> None:
        if self._drain_error is not None:
            raise RuntimeError("completion drainer failed") from self._drain_error

    def _raise_request_error_locked(self, rid: int) -> None:
        """Surface a publish failure to the thread waiting on that request."""
        error = self._failed.get(rid)
        if error is not None:
            raise RuntimeError(f"rid={rid} was never published") from error
