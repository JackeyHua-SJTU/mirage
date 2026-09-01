"""Prompt-prefix KV page cache for MODE_ONLINE_PINNED (OIPL §6.4).

Under OIPL the GPU allocates KV pages and the CPU releases them, so a
completed request's pages are frozen: the GPU never writes them again and
cannot reclaim them until the CPU hands them back through the return ring.
That makes them safe to index and re-import.  This module is that index —
pure Python, no torch, no GPU state.  It is driven exclusively by the runtime's
single owner (drain) thread and holds no lock of its own.

The unit is one physical page = ``page_size`` prompt tokens of full-layer KV.
Blocks are keyed by a chained hash ``h_j = H(h_{j-1}, ids_j)`` and a hit is
confirmed by comparing every matched block's stored token ids *and* walking its
parent link (§6.4 v2.1-b): a bucketed chained hash can hold a block with
identical ids under a different preceding context, and importing it would be a
silently wrong hit rather than a miss.

Only prompt tokens are cached (v1 scope), and only pages entirely inside
``[0, min(prompt_len, final_kv_len) - gen_tail_len)`` (v2.1-d plus the §11-1
generation-prompt tail exclusion): a page that is partly generated text, partly
unwritten, or inside the chat template's per-turn tail is not a stable prefix of
any later turn.

Usage from the owner thread::

    match = cache.match(prompt_ids, cache.publish_pin_budget(unadmitted_npps))
    # publish num_prefix_pages=match.num_prefix_pages, initial_step=npp*page_size
    ...
    free = cache.insert(prompt_ids, exported_pages, final_step, gen_tail_len,
                        num_prefix_pages=match.num_prefix_pages)
    free += cache.tick(oldest_unadmitted_rid, free_estimate=free_pages_seen,
                       head_worst_need=max_pages_per_req)
    runtime.return_pages(free)
"""

from __future__ import annotations

import collections
import hashlib
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Set, Tuple

# Trimming pushes every evicted page into the return ring, which the GPU drains
# at a fixed budget per iteration (§6.3 Step 0), so evicting more than that in
# one owner tick only lengthens the queue.
MAX_EVICTIONS_PER_TICK = 32

# Owner ticks the oldest published request may stay unadmitted before the cache
# starts giving the pool back one block at a time -- and only if the stall is
# attributable to pages (§6.4 v2.1-c′).
DEFAULT_STALL_TICKS = 16


def chain_hash(parent_hash: int, token_ids: Sequence[int]) -> int:
    """64-bit chained block hash ``H(h_{j-1}, ids)``."""
    digest = hashlib.blake2b(digest_size=8)
    digest.update(parent_hash.to_bytes(8, "little"))
    for token in token_ids:
        digest.update((token & 0xFFFFFFFFFFFFFFFF).to_bytes(8, "little"))
    return int.from_bytes(digest.digest(), "little")


@dataclass(eq=False)
class Block:
    """One cached page: ``page_size`` prompt tokens of frozen KV."""

    page_id: int
    token_ids: Tuple[int, ...]
    parent: Optional["Block"] = field(default=None, repr=False)
    chain_hash: int = 0
    refcount: int = 0
    lru_tick: int = 0
    children: Set["Block"] = field(default_factory=set, repr=False)


class PrefixMatch(NamedTuple):
    """Result of :meth:`PrefixCache.match` — pinned until the request completes."""

    num_prefix_pages: int
    page_ids: Tuple[int, ...]
    blocks: Tuple[Block, ...]


EMPTY_MATCH = PrefixMatch(0, (), ())


class PrefixCache:
    """Chained-hash index over frozen KV pages, owned by the drain thread.

    Parameters
    ----------
    page_size, total_pages, max_pages_per_req:
        The kernel's page geometry; ``max_pages_per_req`` is
        ``ceil(max_seq_len / page_size)``, the worst case one request reserves.
    capacity_pages:
        Resident-block budget.  Defaults to ``total_pages - max_pages_per_req``,
        the largest cache that still leaves one worst-case request admissible.
    hash_fn:
        Block hash, overridable so tests can force bucket collisions.
    """

    def __init__(
        self,
        page_size: int,
        total_pages: int,
        max_pages_per_req: int,
        capacity_pages: Optional[int] = None,
        stall_ticks: int = DEFAULT_STALL_TICKS,
        hash_fn=chain_hash,
    ) -> None:
        self._page_size = page_size
        self._total_pages = total_pages
        self._max_pages_per_req = max_pages_per_req
        if capacity_pages is None:
            capacity_pages = max(0, total_pages - max_pages_per_req)
        self._capacity_pages = capacity_pages
        self._stall_ticks = stall_ticks
        self._hash = hash_fn
        self._index: Dict[int, List[Block]] = {}
        self._blocks: Set[Block] = set()
        self._by_page: Dict[int, Block] = {}
        self._clock = 0
        self._stall_key = None
        self._stalled_for = 0
        self._counts: Dict[str, int] = collections.Counter()

    # ── match at publish ──────────────────────────────────────────────────

    def match(
        self, prompt_ids: Sequence[int], pin_budget_remaining: Optional[int] = None
    ) -> PrefixMatch:
        """Longest verified cached prefix of *prompt_ids*, pinned.

        The match is capped so ``initial_step = npp * page_size`` leaves at
        least one prompt token for the kernel to process, and by
        *pin_budget_remaining* (see :meth:`publish_pin_budget`).  The caller
        must publish ``initial_step = npp * page_size`` in the same locked
        publish (v2.1-0) and must eventually release the pins, either through
        :meth:`insert`/:meth:`on_export` when the request completes or through
        :meth:`release` if it is never published.
        """
        prompt_len = len(prompt_ids)
        limit = (prompt_len - 1) // self._page_size
        if pin_budget_remaining is not None:
            limit = min(limit, max(0, pin_budget_remaining))
        blocks = self._lookup(prompt_ids, limit)
        if not blocks:
            self._counts["misses"] += 1
            return EMPTY_MATCH
        for block in blocks:
            block.refcount += 1
            self._clock += 1
            block.lru_tick = self._clock
        self._counts["hits"] += 1
        self._counts["blocks_matched"] += len(blocks)
        return PrefixMatch(
            len(blocks), tuple(b.page_id for b in blocks), tuple(blocks)
        )

    def release(self, blocks: Iterable[Block]) -> None:
        """Drop pins taken by :meth:`match` without an export (publish aborted)."""
        for block in blocks:
            self._unpin(block)

    def publish_pin_budget(self, unadmitted_npps: Sequence[int]) -> int:
        """Pages a new publish may pin, given the published-unadmitted npps.

        A published request pins its prefix pages until it completes, and it
        cannot complete before it is admitted.  The GPU admits the ring head
        first and needs ``max_pages_per_req - npp_head`` free pages, so the pins
        held by every *other* unadmitted request must leave that many pages
        reachable (§6.4 v2.1-a).  Without the bound, published requests can pin
        each other out of admission forever: the GPU ledger keeps that state
        safe, but not live.  *unadmitted_npps* is oldest-first, head included.
        """
        if not unadmitted_npps:
            return self._max_pages_per_req  # this request becomes the ring head
        others = sum(unadmitted_npps[1:])
        return max(0, self._total_pages - self._max_pages_per_req - others)

    # ── completion accounting ─────────────────────────────────────────────

    def on_export(self, page_ids: Iterable[int]) -> List[int]:
        """Reconcile one completion's exported pages against the cache.

        Every exported page the cache owns was imported by that request, so its
        pin drops here; the rest belong to nobody and are returned by the
        caller.  Export accounting is unconditional — an abandoned completion
        still returns its pages (§6.4).
        """
        free: List[int] = []
        for page in page_ids:
            block = self._by_page.get(page)
            if block is None:
                free.append(page)
            else:
                self._unpin(block)
        return free

    def insert(
        self,
        prompt_ids: Sequence[int],
        page_ids: Sequence[int],
        final_kv_len: int,
        gen_tail_len: int,
        num_prefix_pages: int = 0,
    ) -> List[int]:
        """Account one completion: cache what is cacheable, free the rest.

        *page_ids* is the request's exported page table in page order, so page
        ``j`` holds prompt tokens ``[j*PS, (j+1)*PS)``.  Pages below
        *num_prefix_pages* were imported and are already cached (their pin drops
        here, as in :meth:`on_export`); fresh pages fully inside the cacheable
        range extend the chain, deduping against an identical block that another
        request cached first.  Returns the pages to hand the return ring.
        """
        page_size = self._page_size
        cacheable = min(len(prompt_ids), final_kv_len) - gen_tail_len
        full_pages = max(0, cacheable) // page_size
        free: List[int] = []
        parent: Optional[Block] = None
        chain_ok = True
        if num_prefix_pages > 0:
            # The chain continues from the last imported block; if that block is
            # gone the tail cannot be linked to a verified parent, and inserting
            # it under the wrong one would be a silently wrong future hit.
            if len(page_ids) >= num_prefix_pages:
                parent = self._by_page.get(page_ids[num_prefix_pages - 1])
            chain_ok = parent is not None
        for idx, page in enumerate(page_ids):
            block = self._by_page.get(page)
            if block is not None:
                self._unpin(block)
                continue
            if chain_ok and num_prefix_pages <= idx < full_pages:
                ids = tuple(prompt_ids[idx * page_size : (idx + 1) * page_size])
                existing = self._find(parent, ids)
                if existing is None:
                    parent = self._insert_block(parent, ids, page)
                    self._counts["inserted"] += 1
                else:
                    parent = existing
                    free.append(page)
                    self._counts["deduped"] += 1
                continue
            if idx >= full_pages:
                chain_ok = False  # nothing past a partial page is a stable prefix
            free.append(page)
        return free

    # ── eviction ──────────────────────────────────────────────────────────

    def evict(self, count: int) -> List[int]:
        """Evict up to *count* blocks, LRU first among unpinned leaves.

        Leaf-first keeps every resident block's parent chain resident, which is
        what makes a chain lookup able to reach the block at all.
        """
        pages: List[int] = []
        while len(pages) < count:
            victim = None
            for block in self._blocks:
                if block.refcount or block.children:
                    continue
                if victim is None or block.lru_tick < victim.lru_tick:
                    victim = block
            if victim is None:
                break
            self._remove_block(victim)
            pages.append(victim.page_id)
        self._counts["evicted"] += len(pages)
        return pages

    def tick(
        self,
        oldest_unadmitted=None,
        free_estimate: Optional[int] = None,
        head_worst_need: Optional[int] = None,
    ) -> List[int]:
        """One owner-thread cache tick; returns pages for the return ring.

        Always trims to the capacity budget.  On top of that it force-evicts one
        extra block per tick while **both** halves of the §6.4 v2.1-(c′) rule
        hold:

        1. *oldest_unadmitted* names the same published request for
           ``stall_ticks`` consecutive ticks.  Resident pages hold the GPU's
           ``avail_uncommitted`` down 1:1 and the GPU has no "I am starving"
           channel, so a cache that never reacts to a stalled ring head can park
           admission indefinitely without breaking any invariant (v2.1-c).
        2. The stall is **attributable to pages**: *free_estimate* — the pages
           the caller believes the GPU could still hand out — is below
           *head_worst_need*, the pages the GPU's admission gate wants before it
           will admit that head.

        Condition 1 alone cannot tell page starvation from ROW/batch starvation,
        and the difference matters: no amount of eviction buys a free row.  P2
        measured the misfire on hardware — 12 in-flight requests on 4 rows with
        47–56 of 64 pages free fired backpressure 1061 times in 0.8 s and
        force-evicted 163 blocks, stripping the cache to the pinned chains for
        nothing.  Condition 2 suppresses exactly that case and counts it as
        ``backpressure_suppressed``.

        *free_estimate* is expected to be an upper bound on what the GPU can
        allocate (a free-page mirror plus in-flight returns ignores the
        reservations already made for admitted requests), so the caller should
        pair it with an upper bound on the need — ``max_pages_per_req``, the
        need of a head that matched nothing — rather than a tight one.  Failing
        to fire when the stall really is page-caused is a wedged pipeline;
        firing once too often only costs a block.  That is also why omitting
        *free_estimate* (a caller with no view of the pool) keeps the
        unattributed v2.1-c behaviour of firing on the stall alone; passing
        *free_estimate* without *head_worst_need* compares against
        ``max_pages_per_req``.
        """
        if oldest_unadmitted is not None and oldest_unadmitted == self._stall_key:
            self._stalled_for += 1
        else:
            self._stall_key = oldest_unadmitted
            self._stalled_for = 0
        over = len(self._blocks) - self._capacity_pages
        if oldest_unadmitted is not None and self._stalled_for >= self._stall_ticks:
            if self._page_starved(free_estimate, head_worst_need):
                over = max(over, 1)
                self._counts["backpressure_ticks"] += 1
            else:
                self._counts["backpressure_suppressed"] += 1
        if over <= 0:
            return []
        return self.evict(min(over, MAX_EVICTIONS_PER_TICK))

    def _page_starved(
        self, free_estimate: Optional[int], head_worst_need: Optional[int]
    ) -> bool:
        """Is the pool, not the batch, what the stalled ring head is waiting on?"""
        if free_estimate is None:
            return True  # no attribution channel: fall back to v2.1-c
        need = self._max_pages_per_req if head_worst_need is None else head_worst_need
        return free_estimate < need

    def reset(self) -> List[int]:
        """Drop the whole index (kernel restart) and return every resident page."""
        pages = sorted(self._by_page)
        self._index.clear()
        self._blocks.clear()
        self._by_page.clear()
        self._counts.clear()
        self._stall_key = None
        self._stalled_for = 0
        return pages

    # ── introspection ─────────────────────────────────────────────────────

    def stats(self) -> Dict[str, int]:
        """Counter snapshot; ``resident_pages``/``pinned_pages`` are live sizes.

        ``hits``/``misses`` count publishes, not pages: a match clamped to zero
        by the pin budget counts as a miss, since that is what was published.
        ``backpressure_ticks`` and ``backpressure_suppressed`` split the
        stalled-head ticks by page attribution (§6.4 v2.1-c′): the first evicted
        to unblock admission, the second decided the head was waiting on a row
        rather than a page and left the cache alone.  A ratio that leans hard on
        ``backpressure_suppressed`` is healthy — it is the misfire not happening.
        """
        snapshot = {
            "hits": 0,
            "misses": 0,
            "blocks_matched": 0,
            "inserted": 0,
            "deduped": 0,
            "evicted": 0,
            "backpressure_ticks": 0,
            "backpressure_suppressed": 0,
            "unpin_underflow": 0,
        }
        snapshot.update(self._counts)
        snapshot["resident_pages"] = len(self._by_page)
        snapshot["pinned_pages"] = sum(
            1 for block in self._blocks if block.refcount > 0
        )
        return snapshot

    @property
    def resident_pages(self) -> int:
        return len(self._by_page)

    @property
    def capacity_pages(self) -> int:
        return self._capacity_pages

    # ── index internals ───────────────────────────────────────────────────

    def _lookup(self, prompt_ids: Sequence[int], limit: int) -> List[Block]:
        page_size = self._page_size
        blocks: List[Block] = []
        parent: Optional[Block] = None
        for j in range(min(limit, len(prompt_ids) // page_size)):
            ids = tuple(prompt_ids[j * page_size : (j + 1) * page_size])
            block = self._find(parent, ids)
            if block is None:
                break
            blocks.append(block)
            parent = block
        return blocks

    def _find(
        self, parent: Optional[Block], ids: Tuple[int, ...]
    ) -> Optional[Block]:
        """The cached block holding *ids* directly after *parent*, if any.

        The parent identity check is what makes the per-block id compare
        equivalent to a full-prefix compare (§7 S6).
        """
        bucket = self._index.get(self._hash(parent.chain_hash if parent else 0, ids))
        if not bucket:
            return None
        for block in bucket:
            if block.token_ids == ids and block.parent is parent:
                return block
        return None

    def _insert_block(
        self, parent: Optional[Block], ids: Tuple[int, ...], page: int
    ) -> Block:
        digest = self._hash(parent.chain_hash if parent else 0, ids)
        self._clock += 1
        block = Block(
            page_id=page,
            token_ids=ids,
            parent=parent,
            chain_hash=digest,
            lru_tick=self._clock,
        )
        self._index.setdefault(digest, []).append(block)
        self._blocks.add(block)
        self._by_page[page] = block
        if parent is not None:
            parent.children.add(block)
        return block

    def _remove_block(self, block: Block) -> None:
        bucket = self._index[block.chain_hash]
        bucket.remove(block)
        if not bucket:
            del self._index[block.chain_hash]
        self._blocks.discard(block)
        del self._by_page[block.page_id]
        if block.parent is not None:
            block.parent.children.discard(block)

    def _unpin(self, block: Block) -> None:
        if block.refcount > 0:
            block.refcount -= 1
        else:
            # A page exported by a request that never imported it: keep the
            # ledger monotone rather than letting a negative pin free it early.
            self._counts["unpin_underflow"] += 1


def compute_gen_tail_len(tokenizer, probe_replies=None, **template_kwargs) -> int:
    """Tokens at the end of a rendered prompt that the next turn does not keep.

    Qwen3's template renders a stable cross-turn prefix under the default
    ``enable_thinking=True``, but the hard switch ``enable_thinking=False``
    force-inserts ``<think>\\n\\n</think>`` into the generation prompt and strips
    it again next turn (§11-1).  Rather than special-case a template, this
    measures the tail once at startup: render turn *k*'s prompt and turn *k+1*'s
    prompt and report how much of the former the latter dropped.  The caller
    passes the result to :meth:`PrefixCache.insert`, which keeps that many
    tokens out of the cacheable range.

    Returns 0 for a tokenizer with no chat template (raw prompts are stable).
    """
    if probe_replies is None:
        probe_replies = ("ok", "<think>\nreasoning\n</think>\n\nok")
    if not getattr(tokenizer, "chat_template", None):
        return 0

    def render(messages) -> List[int]:
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **template_kwargs
            )
        except TypeError:
            # An older template rejects the keyword it does not implement.
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return list(tokenizer(text).input_ids)

    turn1 = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    prompt1 = render(turn1)
    tail = 0
    for reply in probe_replies:
        prompt2 = render(
            turn1
            + [
                {"role": "assistant", "content": reply},
                {"role": "user", "content": "And of Germany?"},
            ]
        )
        shared = 0
        for a, b in zip(prompt1, prompt2):
            if a != b:
                break
            shared += 1
        tail = max(tail, len(prompt1) - shared)
    return tail
