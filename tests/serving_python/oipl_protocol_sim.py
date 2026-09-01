#!/usr/bin/env python3
"""Executable model of the OIPL page-lifecycle protocol.

Spec: docs/mpk/online_serving_and_prefix_cache.md (§6 protocol, §7 invariants).

The design's hard part is not any single step but the interleaving of three
concurrent actors: the GPU scheduler lane (``prepare_next_batch``), the CPU
owner thread, and request-submitting threads.  This module models them as
atomic steps over the same shared pinned channels the real protocol uses, and a
randomized scheduler interleaves those steps while a checker re-validates the
protocol invariants after every single step.

Modelled design = the amended (v2) one from §6 of the design doc:

  GPU (single scheduler lane, one atomic step per sub-step of an iteration)
    reclaim  row lease: reclaim a row only after the CPU acked with step=-1
    step0    bounded drain of the page-return ring into the free queue
    step1    completion detect -> export the row's pages to the completion
             ring (no self-free) + refund the unused page reservation
    step23   compact the batch and grow page tables from the free queue
    step4    admission gated by the page ledger, two-phase page-table fill
             (imported prefix pages, then fresh pops), late ring-slot clear
    step56   clear unused slots, write cursors back, publish the return-head
             and free-page-count mirrors
    exec     the "forward pass": KV writes + one output token per input token

  CPU (single owner thread, one atomic step per stage, round-robin)
    drain    one completion: unconditional page accounting, then the row ack
             is queued (the row lease is held until the ack stage)
    evict    LRU / refcount==0 / leaf-first eviction into the return ring;
             force-eviction needs a stalled ring head AND a page attribution
    publish  match-at-publish: lookup + refcount++ + prefix array + ready=1 as
             ONE atomic step; parked (unpublished) requests hold nothing
    ack      write pinned_step[row] = -1 after re-reading the leased row

The invariants re-checked after every atomic step are documented on
``Sim.check``; a violation raises with the full state and the last events.
``Config.bug`` injects a deliberate deviation from the design (see ``BUGS``)
so the checker itself can be shown to bite.

Standalone use::

    python3 oipl_protocol_sim.py --sweep              # sweep + adversarial runs
    python3 oipl_protocol_sim.py --events 300000 --seed 7 --pool 12
"""

from __future__ import annotations

import argparse
import collections
import random
import sys
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Set, Tuple

EOS = 999_999
JUNK = -1  # model output for a position whose token is discarded

# Page ownership states.  Every physical page id is in exactly one of them.
FREE = "free"  # GPU free page queue
ROW = "row"  # exclusively owned (writable) by one active row
COMP = "comp"  # exported, in the completion ring, not yet accounted by CPU
CACHE = "cache"  # frozen, owned by the CPU prefix cache (read-shareable)
RET = "ret"  # in the page-return ring, not yet drained by the GPU


class InvariantViolation(AssertionError):
    """Raised with a full event trace when a protocol invariant breaks."""


class LivenessViolation(AssertionError):
    """Raised when the protocol stops making progress with work outstanding."""


@dataclass
class Config:
    name: str = "default"
    page_size: int = 8
    max_seq: int = 64
    total_pages: int = 24
    rows: int = 4  # total_inflight == MPK_MAX_NUM_BATCHED_REQUESTS
    max_batched_tokens: int = 8
    ring_cap: int = 8
    return_ring_cap: int = 0  # 0 -> next power of two >= total_pages
    drain_limit: int = 4  # Step 0 per-iteration drain budget
    min_prompt: int = 1
    max_prompt: int = 40
    max_pending_submits: int = 6
    cache_budget: int = -1  # -1 -> max(0, total_pages - pages_per_req)
    stall_ticks: int = 3  # owner ticks before eviction turns aggressive
    publish_clamp: bool = True  # bound prefix pins held by unadmitted requests
    stall_limit: int = 30000  # events without progress -> LivenessViolation
    trace_len: int = 400
    bug: str = ""  # deliberate deviation, to prove the checker bites

    @property
    def pages_per_req(self) -> int:
        return (self.max_seq + self.page_size - 1) // self.page_size

    @property
    def ret_cap(self) -> int:
        if self.return_ring_cap:
            return self.return_ring_cap
        cap = 1
        while cap < self.total_pages:
            cap *= 2
        return cap

    def validate(self) -> None:
        assert self.ring_cap & (self.ring_cap - 1) == 0
        assert self.ret_cap & (self.ret_cap - 1) == 0
        assert self.ret_cap >= self.total_pages  # structurally cannot overflow
        assert self.ring_cap >= self.rows  # S4: comp occupancy <= total_inflight
        assert 1 <= self.min_prompt <= self.max_prompt <= self.max_seq
        assert self.total_pages >= self.pages_per_req  # else nothing is admissible


BUGS = {
    "": "no injected bug (the amended v2 design)",
    "no_ledger": "Step 4 admits without the reservation gate (CPU watermark only)",
    "self_free": "Step 1 also self-frees the exported pages (upstream behaviour)",
    "no_publish_refcount": "match-at-publish does not take a refcount",
    "no_refund": "completion zeroes reserved_remaining without refunding avail",
    "no_publish_clamp": "no bound on prefix pins held by unadmitted requests",
    "raw_last_page_len": "last_page_len via a bare % (missing the A10.1 fix)",
    "ids_only_match": "cache lookup compares block ids without the parent link",
    "unattributed_backpressure": "backpressure fires on any stalled ring head "
                                 "(the pre-v2.1-c′ rule, misfires on row starvation)",
    "no_backpressure": "the cache never reacts to a stalled ring head",
}


def kv_digest(prev: int, token: int) -> int:
    """Fold a token into a running digest of everything written before it.

    A KV entry is not a function of its own token alone: layer L>0 keys and
    values depend on the whole preceding context.  Modelling the KV of a
    position as (position, token, digest-of-the-prefix) is what lets the
    checker catch a hit on a block whose tokens match but whose *context*
    does not.
    """
    return (prev * 1_000_003 + token + 1) % ((1 << 61) - 1)


@dataclass
class Request:
    rid: int
    tokens: Tuple[int, ...]
    eos_at: Optional[int]  # absolute position of the EOS token, or None

    def __post_init__(self) -> None:
        digests = [0]
        for token in self.tokens:
            digests.append(kv_digest(digests[-1], token))
        self.digests: Tuple[int, ...] = tuple(digests)

    @property
    def prompt_len(self) -> int:
        return len(self.tokens)

    def model_output(self, pos: int) -> int:
        """The token the model predicts for absolute position ``pos``."""
        if pos < self.prompt_len:
            return JUNK  # discarded by Step 1's `pos >= prompt_len` gate
        if self.eos_at is not None and pos == self.eos_at:
            return EOS
        return 100_000 + (self.rid % 500) * 100 + (pos % 100)


class Fabric:
    """The pinned CPU<->GPU channels (§6.2), the only shared state."""

    def __init__(self, cfg: Config):
        cap = cfg.ring_cap
        self.mask = cap - 1
        # Request ring (CPU -> GPU), extended with the prefix-import channel.
        self.req_ready = [0] * cap
        self.req_rid = [-1] * cap
        self.req_prompt_len = [0] * cap
        self.req_npp = [0] * cap
        self.req_prefix_pages: List[List[int]] = [[] for _ in range(cap)]
        self.inbox: List[List[int]] = [[] for _ in range(cap)]
        # Completion ring (GPU -> CPU), extended with the page-export channel.
        self.comp_ready = [0] * cap
        self.comp_rid = [-1] * cap
        self.comp_row = [-1] * cap
        self.comp_final_step = [0] * cap
        self.comp_pages: List[List[int]] = [[] for _ in range(cap)]
        # Row lease / progress channel.
        self.pinned_step = [0] * cfg.rows
        self.rid_at_row = [-1] * cfg.rows
        # Page-return ring (CPU -> GPU, SPSC).
        self.ret_cap = cfg.ret_cap
        self.ret_mask = self.ret_cap - 1
        self.ret_ring = [-1] * self.ret_cap
        self.ret_tail = 0
        self.ret_head_mirror = 0
        # Free-page count mirror (GPU -> CPU): page_queue_tail - page_queue_head
        # as of the end of the GPU's last iteration.  With ret_head_mirror this
        # is the CPU's only view of the pool, and the input to the page
        # attribution of backpressure (§6.4 v2.1-c′).
        self.free_count_mirror = cfg.total_pages


class GPU:
    """``prepare_next_batch`` (§6.3) plus the forward pass, as atomic steps."""

    SUBSTEPS = ("reclaim", "step0", "step1", "step23", "step4", "step56", "exec")

    def __init__(self, sim: "Sim"):
        self.sim = sim
        cfg = sim.cfg
        self.cfg = cfg
        self.fab = sim.fab
        # gpu_malloc private state (§1.4)
        self.page_queue = list(range(cfg.total_pages))
        self.pq_head = 0
        self.pq_tail = cfg.total_pages
        self.free_rows = list(range(cfg.rows))
        self.free_row_top = cfg.rows
        self.request_ids = [-1] * cfg.rows  # slot -> row
        self.request_rids = [-1] * cfg.rows  # slot -> rid
        self.gpu_req_head = 0
        self.gpu_comp_tail = 0
        self.ret_head = 0  # private head of the page-return ring
        # device tensors (§1.2)
        self.qo_indptr = [0] * (cfg.rows + 1)
        self.kv_indptr = [0] * (cfg.rows + 1)
        self.kv_indices = [-1] * (cfg.rows * cfg.pages_per_req)
        self.last_page_len = [0] * cfg.rows
        self.step = [0] * cfg.rows
        self.prompt_length = [0] * cfg.rows
        self.tokens = [[None] * (cfg.max_seq + 1) for _ in range(cfg.rows)]
        self.output_tokens = [JUNK] * cfg.max_batched_tokens
        # OIPL ledger (§6.3a)
        self.avail_uncommitted = cfg.total_pages
        self.reserved_remaining = [0] * cfg.rows
        # batch accumulators, live across step23 -> step56
        self.num_reqs = 0
        self.num_tokens = 0
        self.num_pages = 0
        self.substep = 0
        self.iterations = 0

    # ── free page queue ───────────────────────────────────────────────────

    def _pop_page(self) -> int:
        if self.pq_tail - self.pq_head <= 0:
            self.sim.violation("I3", "pop from an empty page queue")
        page = self.page_queue[self.pq_head % self.cfg.total_pages]
        self.pq_head += 1
        return page

    def _push_page(self, page: int) -> None:
        self.page_queue[self.pq_tail % self.cfg.total_pages] = page
        self.pq_tail += 1

    # ── one atomic step ───────────────────────────────────────────────────

    def step_once(self) -> str:
        name = self.SUBSTEPS[self.substep]
        getattr(self, "_" + name)()
        self.substep += 1
        if self.substep == len(self.SUBSTEPS):
            self.substep = 0
            self.iterations += 1
        return name

    # ── row lease reclaim (#754) ──────────────────────────────────────────

    def _reclaim(self) -> None:
        for row in range(self.cfg.rows):
            owner = self.fab.rid_at_row[row]
            if owner >= 0 and self.fab.pinned_step[row] == -1:
                self.fab.rid_at_row[row] = -1
                self.free_rows[self.free_row_top] = row
                self.free_row_top += 1
                if self.sim.row_fresh[row]:
                    self.sim.violation(
                        "I1", f"row {row} reclaimed while still owning pages"
                    )
                self.sim.trace_add(f"gpu.reclaim row={row} rid={owner}")

    # ── Step 0: bounded drain of the page-return ring ─────────────────────

    def _step0(self) -> None:
        tail = self.fab.ret_tail  # ld.acquire.sys
        drained = 0
        while self.ret_head != tail and drained < self.cfg.drain_limit:
            page = self.fab.ret_ring[self.ret_head & self.fab.ret_mask]
            self._push_page(page)
            self.sim.move_page(page, RET, FREE)
            self.avail_uncommitted += 1
            self.ret_head += 1
            drained += 1
        if drained:
            self.sim.trace_add(
                f"gpu.step0 drained={drained} avail={self.avail_uncommitted}"
            )

    # ── Step 1: finalize the previous batch ───────────────────────────────

    def _step1(self) -> None:
        cfg = self.cfg
        for i in range(cfg.rows):
            row = self.request_ids[i]
            if row == -1:
                continue
            step = self.step[row]
            qo = self.qo_indptr[i]
            num_tokens = self.qo_indptr[i + 1] - qo
            plen = self.prompt_length[row]
            for j in range(num_tokens):
                pos = step + j + 1
                if plen <= pos < cfg.max_seq:
                    self.tokens[row][pos] = self.output_tokens[qo + j]
            self.step[row] = step + num_tokens
            self.fab.pinned_step[row] = step + num_tokens  # st.release.sys
            done = (step + num_tokens + 1 >= cfg.max_seq) or (
                self.tokens[row][step + num_tokens] == EOS
                and step + num_tokens >= plen
            )
            if done:
                self._complete(i, row, step + num_tokens)

    def _complete(self, slot: int, row: int, final_step: int) -> None:
        fab = self.fab
        rid = self.request_rids[slot]
        comp_slot = self.gpu_comp_tail & fab.mask
        if fab.comp_ready[comp_slot] != 0:
            # S4 proves this cannot happen (occupancy <= total_inflight <= cap);
            # the real kernel spins here, a spin would mean the proof is wrong.
            self.sim.violation("I5", f"completion ring slot {comp_slot} overwritten")
        kvp = self.kv_indptr[slot]
        num_pages = self.kv_indptr[slot + 1] - kvp
        pages = list(self.kv_indices[kvp : kvp + num_pages])
        # Export the pages; the GPU does NOT free them (ownership inversion).
        fab.comp_rid[comp_slot] = rid
        fab.comp_row[comp_slot] = row
        fab.comp_final_step[comp_slot] = final_step
        fab.comp_pages[comp_slot] = pages
        fab.comp_ready[comp_slot] = 1  # st.release.sys, after the payload
        self.gpu_comp_tail += 1
        fresh = self.sim.row_fresh[row]
        exported_fresh = [p for p in pages if p in fresh]
        if len(exported_fresh) != len(fresh):
            self.sim.violation("I1", f"row {row} owns pages outside its page table")
        for page in exported_fresh:
            self.sim.move_page(page, ROW, COMP)
        fresh.clear()
        self.sim.comp_fresh[comp_slot] = exported_fresh
        if self.cfg.bug == "self_free":
            for page in pages:
                self._push_page(page)
        # Retain the row until the CPU acks; refund the unused reservation.
        self.sim.row_imported[row] = []
        self.request_ids[slot] = -1
        self.request_rids[slot] = -1
        if self.cfg.bug != "no_refund":
            self.avail_uncommitted += self.reserved_remaining[row]
        self.reserved_remaining[row] = 0
        self.sim.stats["completions"] += 1
        self.sim.progress()
        self.sim.trace_add(
            f"gpu.step1 complete rid={rid} row={row} final_step={final_step} "
            f"pages={pages} avail={self.avail_uncommitted}"
        )

    # ── Steps 2+3: snapshot page table, compact the batch, grow pages ─────

    def _step23(self) -> None:
        cfg = self.cfg
        snapshot = list(self.kv_indices[: self.kv_indptr[cfg.rows]])
        num_reqs = num_tokens = num_pages = 0
        for i in range(cfg.rows):
            row = self.request_ids[i]
            if row == -1:
                continue
            kvp = self.kv_indptr[i]
            num_old_pages = self.kv_indptr[i + 1] - kvp
            self.request_ids[num_reqs] = row
            self.request_rids[num_reqs] = self.request_rids[i]
            self.qo_indptr[num_reqs] = num_tokens
            self.kv_indptr[num_reqs] = num_pages
            step = self.step[row]
            remaining = self.prompt_length[row] - step
            budget = cfg.max_batched_tokens - num_tokens
            num_new_tokens = min(remaining, budget) if remaining > 0 else min(1, budget)
            num_new_pages = (step + num_new_tokens + cfg.page_size - 1) // cfg.page_size
            self.last_page_len[num_reqs] = self._last_page_len(step + num_new_tokens)
            for j in range(num_old_pages):
                self.kv_indices[num_pages + j] = snapshot[kvp + j]
            for j in range(num_old_pages, num_new_pages):
                page = self._pop_page()
                self.reserved_remaining[row] -= 1
                if self.reserved_remaining[row] < 0:
                    self.sim.violation("I3", f"reserved_remaining[{row}] < 0")
                self.sim.move_page(page, FREE, ROW)
                self.sim.row_fresh[row].add(page)
                self.kv_indices[num_pages + j] = page
            num_pages += num_new_pages
            num_tokens += num_new_tokens
            num_reqs += 1
        self.num_reqs, self.num_tokens, self.num_pages = num_reqs, num_tokens, num_pages

    def _last_page_len(self, seq_len: int) -> int:
        rem = seq_len % self.cfg.page_size
        if self.cfg.bug == "raw_last_page_len":
            return rem  # the online path's bare %, see §1.2 / §8 A10.1
        return self.cfg.page_size if rem == 0 else rem

    # ── Step 4: admission, gated by the page ledger ───────────────────────

    def _step4(self) -> None:
        cfg = self.cfg
        fab = self.fab
        while (
            self.num_reqs < cfg.rows
            and self.num_tokens < cfg.max_batched_tokens
            and self.free_row_top > 0
        ):
            req_slot = self.gpu_req_head & fab.mask
            if fab.req_ready[req_slot] == 0:  # ld.acquire.sys
                break
            rid = fab.req_rid[req_slot]
            plen = fab.req_prompt_len[req_slot]
            npp = fab.req_npp[req_slot]
            worst_need = cfg.pages_per_req - npp
            if cfg.bug != "no_ledger" and self.avail_uncommitted < worst_need:
                self.sim.stats["admission_deferred"] += 1
                self.sim.trace_add(
                    f"gpu.step4 defer rid={rid} npp={npp} need={worst_need} "
                    f"avail={self.avail_uncommitted}"
                )
                break  # leave ready=1: the request is retried next iteration
            self.avail_uncommitted -= worst_need
            row = self.free_rows[self.free_row_top - 1]
            self.free_row_top -= 1
            if self.sim.row_fresh[row]:
                self.sim.violation("I1", f"row {row} leased while still owning pages")
            self.reserved_remaining[row] = worst_need
            for j in range(plen):
                self.tokens[row][j] = fab.inbox[req_slot][j]
            self.prompt_length[row] = plen
            initial_step = npp * cfg.page_size
            self.step[row] = initial_step
            fab.pinned_step[row] = initial_step  # st.release.sys, before the owner
            fab.rid_at_row[row] = rid  # st.release.sys
            self.request_ids[self.num_reqs] = row
            self.request_rids[self.num_reqs] = rid
            self.qo_indptr[self.num_reqs] = self.num_tokens
            self.kv_indptr[self.num_reqs] = self.num_pages
            remaining = plen - initial_step
            num_new_tokens = min(
                remaining if remaining > 0 else 1,
                cfg.max_batched_tokens - self.num_tokens,
            )
            num_new_pages = (
                initial_step + num_new_tokens + cfg.page_size - 1
            ) // cfg.page_size
            self.last_page_len[self.num_reqs] = self._last_page_len(
                initial_step + num_new_tokens
            )
            imported = []
            for j in range(num_new_pages):
                if j < npp:  # phase 1: imported prefix pages
                    page = fab.req_prefix_pages[req_slot][j]
                    imported.append(page)
                else:  # phase 2: fresh pages
                    page = self._pop_page()
                    self.reserved_remaining[row] -= 1
                    if self.reserved_remaining[row] < 0:
                        self.sim.violation("I3", f"reserved_remaining[{row}] < 0")
                    self.sim.move_page(page, FREE, ROW)
                    self.sim.row_fresh[row].add(page)
                self.kv_indices[self.num_pages + j] = page
            # Late slot clear: the prefix array must be consumed first (§10-2).
            fab.req_ready[req_slot] = 0  # st.release.sys
            self.gpu_req_head += 1
            self.sim.on_admit(rid, row, npp, imported, initial_step)
            self.num_tokens += num_new_tokens
            self.num_pages += num_new_pages
            self.num_reqs += 1
            self.sim.stats["admissions"] += 1
            self.sim.progress()
            self.sim.trace_add(
                f"gpu.step4 admit rid={rid} row={row} npp={npp} "
                f"init_step={initial_step} imported={imported} "
                f"avail={self.avail_uncommitted}"
            )

    # ── Steps 5+6: clear unused slots, write cursors back, publish mirrors ─

    def _step56(self) -> None:
        cfg = self.cfg
        for i in range(self.num_reqs, cfg.rows):
            self.request_ids[i] = -1
            self.request_rids[i] = -1
        for i in range(self.num_reqs, cfg.rows + 1):
            self.qo_indptr[i] = self.num_tokens
            self.kv_indptr[i] = self.num_pages
        self.fab.ret_head_mirror = self.ret_head  # published for CPU accounting
        self.fab.free_count_mirror = self.pq_tail - self.pq_head

    # ── the forward pass: KV writes and one output token per input token ──

    def _exec(self) -> None:
        cfg = self.cfg
        for i in range(self.num_reqs):
            row = self.request_ids[i]
            rid = self.request_rids[i]
            req = self.sim.requests[rid]
            base = self.step[row]
            qo = self.qo_indptr[i]
            n = self.qo_indptr[i + 1] - qo
            kvp = self.kv_indptr[i]
            # What attention derives from the page table alone (§1.2) has to be
            # the real sequence length -- this is what makes a forged prefix
            # page table a correct history.
            num_pages = self.kv_indptr[i + 1] - kvp
            derived = (num_pages - 1) * cfg.page_size + self.last_page_len[i]
            if derived != base + n:
                self.sim.violation(
                    "I8",
                    f"slot {i}: page table implies seq_len {derived}, "
                    f"true length is {base + n}",
                )
            for j in range(n):
                pos = base + j
                page = self.kv_indices[kvp + pos // cfg.page_size]
                self.sim.write_kv(row, page, pos, self.tokens[row][pos])
                self.output_tokens[qo + j] = req.model_output(pos + 1)


@dataclass
class Block:
    """One cache block = one physical page = page_size prompt tokens."""

    h: int  # chained hash h_j = H(h_{j-1}, ids)
    parent: Optional["Block"]
    tokens: Tuple[int, ...]
    page: int
    refcount: int = 0
    ts: int = 0
    children: Set[int] = field(default_factory=set)  # ids of child blocks

    def __hash__(self) -> int:
        return id(self)


@dataclass
class Published:
    rid: int
    slot: int
    npp: int
    blocks: List[Block]
    held_pages: List[int]


def weak_hash(prev: int, ids: Sequence[int]) -> int:
    """Deliberately collision-prone: hits must be decided by the id compare."""
    return (prev * 31 + sum(ids)) % 97


class CPU:
    """The single owner thread (§6.4) plus the request-submitting threads."""

    STAGES = ("drain", "evict", "publish", "ack")

    def __init__(self, sim: "Sim"):
        self.sim = sim
        self.cfg = sim.cfg
        self.fab = sim.fab
        self.pending_submits: Deque[Request] = collections.deque()
        self.published: Dict[int, Published] = {}
        self.unadmitted: Deque[int] = collections.deque()
        self.pending_acks: Deque[Tuple[int, int]] = collections.deque()
        self.buckets: Dict[int, List[Block]] = {}
        self.blocks: Set[Block] = set()
        self.page_to_block: Dict[int, Block] = {}
        self.cpu_req_tail = 0
        self.cpu_comp_head = 0
        self.clock = 0
        self.stage = 0
        self._stall_head = -1
        self._stall_ticks = 0

    @property
    def cache_budget(self) -> int:
        if self.cfg.cache_budget >= 0:
            return self.cfg.cache_budget
        return max(0, self.cfg.total_pages - self.cfg.pages_per_req)

    # ── submit (any request thread; holds nothing) ────────────────────────

    def submit(self, req: Request) -> None:
        self.sim.requests[req.rid] = req
        self.pending_submits.append(req)
        self.sim.trace_add(f"cpu.submit rid={req.rid} plen={req.prompt_len}")

    # ── one atomic owner-thread stage ─────────────────────────────────────

    def tick(self) -> str:
        name = self.STAGES[self.stage]
        self.clock += 1
        self._observe_admissions()
        getattr(self, "_" + name)()
        self.stage = (self.stage + 1) % len(self.STAGES)
        return name

    def _observe_admissions(self) -> None:
        # The GPU drains the request ring in FIFO order, so ready==0 on the
        # oldest published slot means exactly "that request was admitted".
        while self.unadmitted:
            info = self.published[self.unadmitted[0]]
            if self.fab.req_ready[info.slot] != 0:
                break
            self.unadmitted.popleft()

    # ── stage: drain one completion ───────────────────────────────────────

    def _drain(self) -> None:
        fab = self.fab
        slot = self.cpu_comp_head & fab.mask
        if fab.comp_ready[slot] == 0:  # ld.acquire.sys
            return
        rid = fab.comp_rid[slot]
        row = fab.comp_row[slot]
        final_step = fab.comp_final_step[slot]
        pages = fab.comp_pages[slot]
        self.sim.final_steps[rid] = final_step
        self.sim.check_output(rid, row, final_step, "drain")
        self._account_pages(rid, final_step, pages, slot)
        # Slot reusable only after the bookkeeping succeeded (#754 discipline).
        fab.comp_ready[slot] = 0
        self.cpu_comp_head += 1
        self.pending_acks.append((row, rid))
        self.sim.trace_add(f"cpu.drain rid={rid} row={row} pages={pages}")

    def _account_pages(
        self, rid: int, final_step: int, pages: List[int], comp_slot: int
    ) -> None:
        """Unconditional page accounting for one completion (§6.4)."""
        cfg = self.cfg
        info = self.published.pop(rid)
        req = self.sim.requests[rid]
        prompt = req.tokens
        # Only pages whose whole token block was actually written are insertable.
        full_blocks = min(len(prompt), final_step) // cfg.page_size
        parent = info.blocks[-1] if info.blocks else None
        chain_ok = True
        for idx, page in enumerate(pages):
            block = self.page_to_block.get(page)
            if block is not None:  # a cached page this request imported
                if page not in info.held_pages:
                    self.sim.violation(
                        "I1", f"rid={rid} returns cached page {page} it never held"
                    )
                info.held_pages.remove(page)
                block.refcount -= 1
                if block.refcount < 0:
                    self.sim.violation("I1", f"refcount < 0 on page {page}")
                continue
            if chain_ok and info.npp <= idx < full_blocks:
                ids = tuple(prompt[idx * cfg.page_size : (idx + 1) * cfg.page_size])
                existing = self._find_block(parent, ids)
                if existing is not None:  # dedupe: identical block already cached
                    parent = existing
                    self._push_return(page)
                    self.sim.stats["dedupes"] += 1
                else:
                    parent = self._insert_block(parent, ids, page)
                    self.sim.stats["inserts"] += 1
                continue
            if idx >= full_blocks:
                chain_ok = False  # nothing past a partial page can be cached
            self._push_return(page)
        if info.held_pages:
            self.sim.violation(
                "I1",
                f"rid={rid} completed without exporting held pages "
                f"{info.held_pages}",
            )
        self.sim.comp_fresh[comp_slot] = []

    # ── stage: eviction ───────────────────────────────────────────────────

    def _evict(self) -> None:
        # The watermark is a throughput heuristic only; correctness is the
        # GPU ledger's job.
        if self.unadmitted:
            head = self.unadmitted[0]
            if head == self._stall_head:
                self._stall_ticks += 1
            else:
                self._stall_head, self._stall_ticks = head, 0
        else:
            self._stall_head, self._stall_ticks = -1, 0
        forced = self._stall_ticks >= self.cfg.stall_ticks and self._backpressure()
        if forced:
            # Backpressure: the ring head is not being admitted *and the pool is
            # why*, so release it one block at a time until it is (the CPU sees
            # only a lagging free-count mirror, so it walks down gradually).
            target = max(0, len(self.blocks) - 1)
        else:
            target = self.cache_budget
        evicted = 0
        while len(self.blocks) > target and evicted < 4:
            candidates = [b for b in self.blocks if b.refcount == 0 and not b.children]
            if not candidates:
                break
            victim = min(candidates, key=lambda b: (b.ts, b.page))
            self._remove_block(victim)
            self._push_return(victim.page)
            evicted += 1
            self.sim.stats["evictions"] += 1
            if forced:
                self.sim.stats["forced_evictions"] += 1
        if evicted:
            self.sim.trace_add(f"cpu.evict n={evicted} cached={len(self.blocks)}")

    def _backpressure(self) -> bool:
        """Is the stalled ring head stalled on *pages*? (§6.4 v2.1-c′)

        A stall alone does not justify force-evicting: the head can equally be
        waiting for a free row, and no amount of eviction produces one.  P2
        measured that misfire on hardware — 12 in-flight requests on 4 rows,
        47–56 of 64 pages free, 1061 backpressure ticks in 0.8 s, 163 blocks
        force-evicted for nothing.  So the trigger also requires the CPU's own
        estimate of the pool to be short of what the GPU's admission gate wants
        for the head.

        Both inputs are deliberately loose in the same direction.  The estimate
        (free-count mirror + pages still in the return ring) is an *upper* bound
        on ``avail_uncommitted``: it cannot see the reservations held by
        admitted rows.  The need is therefore also taken at its upper bound —
        the whole ``pages_per_req``, i.e. the need of a head that matched
        nothing — even though the CPU published the head's npp and could
        subtract it.  Under-firing wedges admission; over-firing costs a block.
        """
        cfg = self.cfg
        if cfg.bug == "no_backpressure":
            return False
        if cfg.bug == "unattributed_backpressure":
            return True  # the pre-v2.1-c′ rule: any stalled head fires
        fab = self.fab
        free_estimate = fab.free_count_mirror + (fab.ret_tail - fab.ret_head_mirror)
        if free_estimate < cfg.pages_per_req:
            self.sim.stats["backpressure_fired"] += 1
            return True
        self.sim.stats["backpressure_suppressed"] += 1
        return False

    # ── stage: match-at-publish (one atomic step) ─────────────────────────

    def _publish(self) -> None:
        cfg, fab = self.cfg, self.fab
        if not self.pending_submits:
            return
        slot = self.cpu_req_tail & fab.mask
        if fab.req_ready[slot] != 0:  # ring full, request stays parked
            return
        req = self.pending_submits[0]
        matched = self._lookup(req.tokens)
        # initial_step = npp*PS must leave at least one token to process.
        npp = min(len(matched), (req.prompt_len - 1) // cfg.page_size)
        npp = min(npp, self._publish_pin_budget())
        blocks = matched[:npp]
        if cfg.bug != "no_publish_refcount":
            for block in blocks:
                block.refcount += 1
                block.ts = self.clock
        pages = [b.page for b in blocks]
        fab.inbox[slot] = list(req.tokens)
        fab.req_rid[slot] = req.rid
        fab.req_prompt_len[slot] = req.prompt_len
        fab.req_npp[slot] = npp
        fab.req_prefix_pages[slot] = pages
        fab.req_ready[slot] = 1  # store_i32_release, after the whole payload
        self.published[req.rid] = Published(req.rid, slot, npp, blocks, list(pages))
        self.unadmitted.append(req.rid)
        self.cpu_req_tail += 1
        self.pending_submits.popleft()
        self.sim.stats["publishes"] += 1
        if npp:
            self.sim.stats["hits"] += 1
            self.sim.stats["hit_tokens"] += npp * cfg.page_size
        self.sim.trace_add(
            f"cpu.publish rid={req.rid} slot={slot} npp={npp} prefix={pages}"
        )

    def _publish_pin_budget(self) -> int:
        """Liveness bound on prefix pins held by not-yet-admitted requests.

        A published request pins its matched blocks until it completes, and it
        cannot complete before it is admitted.  The GPU admits the ring head
        first and needs pages_per_req - npp_head free pages, so the pins held
        by every *other* unadmitted request must leave that many pages
        reachable: sum(pins of unadmitted except head) <= total - pages_per_req.
        Without this bound two published requests can pin each other out of
        admission forever (the ledger keeps it safe, but not live).
        """
        cfg = self.cfg
        if not cfg.publish_clamp or cfg.bug == "no_publish_clamp":
            return cfg.pages_per_req
        if not self.unadmitted:
            return cfg.pages_per_req  # this request becomes the ring head
        others = sum(self.published[r].npp for r in list(self.unadmitted)[1:])
        return max(0, cfg.total_pages - cfg.pages_per_req - others)

    # ── stage: row ack (releases the #754 row lease) ──────────────────────

    def _ack(self) -> None:
        if not self.pending_acks:
            return
        row, rid = self.pending_acks.popleft()
        # The row lease guarantees the data is still the completed request's.
        self.sim.check_output(rid, row, self.sim.final_steps[rid], "ack")
        self.fab.pinned_step[row] = -1  # store_i32_release: row is reclaimable
        self.sim.trace_add(f"cpu.ack rid={rid} row={row}")

    # ── cache index ───────────────────────────────────────────────────────

    def _find_block(
        self, parent: Optional[Block], ids: Tuple[int, ...]
    ) -> Optional[Block]:
        """Find the cached block for ``ids`` *following* ``parent``.

        The parent link is what makes the per-block id compare equivalent to a
        full-prefix compare (§7 S6): a bucketed chained hash can put a block
        with identical ids but a different preceding context in the same
        bucket, and its KV is not this request's KV.
        """
        h = weak_hash(parent.h if parent else 0, ids)
        for block in self.buckets.get(h, ()):
            if block.tokens != ids:
                continue
            if block.parent is parent or self.cfg.bug == "ids_only_match":
                return block
        return None

    def _insert_block(
        self, parent: Optional[Block], ids: Tuple[int, ...], page: int
    ) -> Block:
        h = weak_hash(parent.h if parent else 0, ids)
        block = Block(h=h, parent=parent, tokens=ids, page=page, ts=self.clock)
        self.buckets.setdefault(h, []).append(block)
        self.blocks.add(block)
        self.page_to_block[page] = block
        if parent is not None:
            parent.children.add(id(block))
        self.sim.move_page(page, COMP, CACHE)
        return block

    def _remove_block(self, block: Block) -> None:
        self.buckets[block.h].remove(block)
        if not self.buckets[block.h]:
            del self.buckets[block.h]
        self.blocks.discard(block)
        del self.page_to_block[block.page]
        if block.parent is not None:
            block.parent.children.discard(id(block))

    def _lookup(self, prompt: Tuple[int, ...]) -> List[Block]:
        """Chained lookup with a full id compare of every matched block (S6)."""
        out: List[Block] = []
        parent: Optional[Block] = None
        for j in range(len(prompt) // self.cfg.page_size):
            ids = tuple(prompt[j * self.cfg.page_size : (j + 1) * self.cfg.page_size])
            block = self._find_block(parent, ids)
            if block is None:
                break
            out.append(block)
            parent = block
        return out

    def _push_return(self, page: int) -> None:
        fab = self.fab
        if fab.ret_tail - self.sim.gpu.ret_head >= fab.ret_cap:
            self.sim.violation("I5", "page return ring overwrote an undrained slot")
        fab.ret_ring[fab.ret_tail & fab.ret_mask] = page
        fab.ret_tail += 1  # store_i32_release
        self.sim.move_page(page, (COMP, CACHE), RET)
        self.sim.stats["returns"] += 1


class Workload:
    """Random requests with deliberately shared, page-aligned prefixes."""

    def __init__(self, rng: random.Random, cfg: Config):
        self.rng = rng
        self.cfg = cfg
        ps = cfg.page_size
        self.bases = [
            tuple(rng.randrange(100, 999) for _ in range(ps * k)) for k in (1, 2, 3)
        ]
        # Two different heads sharing one block, so the same block content
        # appears under different prefixes (a wrong-hit trap for the cache).
        self.heads = [
            tuple(rng.randrange(100, 999) for _ in range(ps)) for _ in range(2)
        ]
        self.shared = tuple(rng.randrange(100, 999) for _ in range(ps))
        self.recent: List[Tuple[int, ...]] = []
        self.next_rid = 0

    def next_request(self) -> Request:
        cfg = self.cfg
        r = self.rng.random()
        if r < 0.20 and self.recent:  # exact duplicate prompt
            tokens = self.rng.choice(self.recent)
        else:
            if r < 0.60:
                base = self.rng.choice(self.bases)[: cfg.max_prompt]
            elif r < 0.75:
                base = (self.rng.choice(self.heads) + self.shared)[: cfg.max_prompt]
            else:
                base = ()
            tail_len = self.rng.randint(0, max(0, cfg.max_prompt - len(base)))
            tokens = base + tuple(
                self.rng.randrange(100, 999) for _ in range(tail_len)
            )
        tokens = tokens[: cfg.max_prompt]
        if len(tokens) < cfg.min_prompt:
            tokens = tokens + tuple(
                self.rng.randrange(100, 999)
                for _ in range(cfg.min_prompt - len(tokens))
            )
        self.recent.append(tokens)
        if len(self.recent) > 8:
            self.recent.pop(0)
        rid = self.next_rid
        self.next_rid += 1
        plen = len(tokens)
        if plen >= cfg.max_seq or self.rng.random() < 0.2:
            eos_at = None  # runs to the max_seq bound
        else:
            eos_at = self.rng.randint(plen, cfg.max_seq - 1)
        return Request(rid=rid, tokens=tokens, eos_at=eos_at)


class Sim:
    """Owns the shared state, the checker, and the randomized scheduler."""

    def __init__(self, cfg: Config, seed: int = 0):
        cfg.validate()
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.seed = seed
        self.fab = Fabric(cfg)
        self.gpu = GPU(self)
        self.cpu = CPU(self)
        self.workload = Workload(random.Random(seed ^ 0x5EED), cfg)
        self.requests: Dict[int, Request] = {}
        self.final_steps: Dict[int, int] = {}
        # Checker-owned ground truth (instrumentation, not protocol state).
        self.page_state: List[str] = [FREE] * cfg.total_pages
        self.page_tokens: List[List[object]] = [
            [None] * cfg.page_size for _ in range(cfg.total_pages)
        ]
        self.row_fresh: List[Set[int]] = [set() for _ in range(cfg.rows)]
        self.row_digest: List[int] = [0] * cfg.rows
        self.row_imported: List[List[int]] = [[] for _ in range(cfg.rows)]
        self.comp_fresh: List[List[int]] = [[] for _ in range(cfg.ring_cap)]
        self.trace: Deque[str] = collections.deque(maxlen=cfg.trace_len)
        self.event_no = 0
        self.events_since_progress = 0
        self.max_stall = 0
        self.stats: Dict[str, int] = collections.Counter()

    # ── trace / failure reporting ─────────────────────────────────────────

    def trace_add(self, text: str) -> None:
        self.trace.append(f"#{self.event_no:<7} {text}")

    def violation(self, inv: str, msg: str) -> None:
        raise InvariantViolation(f"{inv}: {msg}\n{self.dump()}")

    def dump(self) -> str:
        cfg = self.cfg
        gpu, cpu = self.gpu, self.cpu
        free_q = [
            gpu.page_queue[i % cfg.total_pages] for i in range(gpu.pq_head, gpu.pq_tail)
        ]
        ret_pending = [
            self.fab.ret_ring[i & self.fab.ret_mask]
            for i in range(gpu.ret_head, self.fab.ret_tail)
        ]
        cache_view = [
            (b.page, b.refcount, b.tokens[:2])
            for b in sorted(cpu.blocks, key=lambda b: b.page)
        ]
        head = [
            "",
            f"config    = {cfg}",
            f"seed      = {self.seed}   event = {self.event_no}",
            f"stats     = {dict(sorted(self.stats.items()))}",
            f"pages     = {self.page_state}",
            f"free_q    = {free_q}  avail={gpu.avail_uncommitted} "
            f"reserved={gpu.reserved_remaining}",
            f"rows      = ids={gpu.request_ids} rids={gpu.request_rids} "
            f"step={gpu.step} owner={self.fab.rid_at_row} "
            f"pinned_step={self.fab.pinned_step}",
            f"row_fresh = {[sorted(s) for s in self.row_fresh]}  "
            f"imported={self.row_imported}",
            f"req_ring  = ready={self.fab.req_ready} head={gpu.gpu_req_head} "
            f"tail={cpu.cpu_req_tail}",
            f"comp_ring = ready={self.fab.comp_ready} tail={gpu.gpu_comp_tail} "
            f"head={cpu.cpu_comp_head}",
            f"ret_ring  = head={gpu.ret_head} tail={self.fab.ret_tail} "
            f"pending={ret_pending} head_mirror={self.fab.ret_head_mirror} "
            f"free_mirror={self.fab.free_count_mirror}",
            f"cache     = {cache_view}",
            "--- last events ---",
        ]
        return "\n".join(head + list(self.trace))

    # ── page-ownership bookkeeping ────────────────────────────────────────

    def move_page(self, page: int, frm, to: str) -> None:
        expected = frm if isinstance(frm, tuple) else (frm,)
        if self.page_state[page] not in expected:
            self.violation(
                "I1",
                f"page {page} is {self.page_state[page]}, expected one of {expected} "
                f"for the {expected}->{to} transition",
            )
        self.page_state[page] = to

    def write_kv(self, row: int, page: int, pos: int, token: int) -> None:
        if self.page_state[page] != ROW or page not in self.row_fresh[row]:
            self.violation(
                "I1",
                f"row {row} writes KV for pos {pos} into page {page} "
                f"(state {self.page_state[page]}) it does not exclusively own",
            )
        self.row_digest[row] = kv_digest(self.row_digest[row], token)
        self.page_tokens[page][pos % self.cfg.page_size] = (
            pos,
            token,
            self.row_digest[row],
        )

    def on_admit(
        self, rid: int, row: int, npp: int, imported: List[int], initial_step: int
    ) -> None:
        """I4: an admitted request's imported pages must hold its own prefix."""
        cfg = self.cfg
        req = self.requests[rid]
        if initial_step != npp * cfg.page_size:
            self.violation("I4", f"rid={rid} initial_step {initial_step} != npp*PS")
        if npp and initial_step > req.prompt_len - 1:
            self.violation(
                "I4", f"rid={rid} initial_step {initial_step} > prompt_len-1"
            )
        for k, page in enumerate(imported):
            if self.page_state[page] != CACHE:
                self.violation(
                    "I4",
                    f"rid={rid} imports page {page} in state {self.page_state[page]}",
                )
            block = self.cpu.page_to_block.get(page)
            if block is None or block.refcount <= 0:
                self.violation(
                    "I4", f"rid={rid} imports page {page} with no live refcount"
                )
            if any(page in self.row_imported[r] for r in range(cfg.rows) if r != row):
                self.stats["shared_imports"] += 1
            want = [
                (pos, req.tokens[pos], req.digests[pos + 1])
                for pos in range(k * cfg.page_size, (k + 1) * cfg.page_size)
            ]
            if self.page_tokens[page] != want:
                self.violation(
                    "I4",
                    f"rid={rid} block {k}: page {page} holds KV "
                    f"{self.page_tokens[page]} but this request's prefix needs "
                    f"{want}",
                )
        self.row_imported[row] = list(imported)
        # The row inherits the context digest of everything it imported.
        self.row_digest[row] = req.digests[initial_step]

    def check_output(self, rid: int, row: int, final_step: int, when: str) -> None:
        """The CPU reads the leased row; the data must still be this request's."""
        cfg = self.cfg
        req = self.requests[rid]
        if self.fab.rid_at_row[row] != rid:
            self.violation(
                "I7",
                f"{when}: row {row} owner is {self.fab.rid_at_row[row]}, not {rid}",
            )
        got = self.gpu.tokens[row]
        if tuple(got[: req.prompt_len]) != req.tokens:
            self.violation("I7", f"{when}: rid={rid} prompt in row {row} was clobbered")
        for pos in range(req.prompt_len, min(final_step, cfg.max_seq - 1) + 1):
            if got[pos] != req.model_output(pos):
                self.violation(
                    "I7", f"{when}: rid={rid} generated token at {pos} is wrong"
                )

    def progress(self) -> None:
        self.events_since_progress = 0

    # ── invariants, checked after every atomic step ───────────────────────

    def check(self) -> None:
        """The protocol invariants (§7), re-validated after every atomic step.

        I1  every physical page has exactly one owner state (free queue / one
            row's page table / completion-ring in transit / cache / return-ring
            in transit) and the owning container really holds it: no page in
            two places, none lost, no write to a page the row does not own.
        I2  avail_uncommitted == (queue_tail - queue_head) - sum(reserved).
        I3  the page queue never underflows; the ledger never goes negative.
        I4  an admitted request with npp>0 has initial_step == npp*PS <=
            prompt_len-1, and its imported pages hold the KV of exactly its own
            token prefix (token-level compare) with a live refcount.
        I5  completion-ring occupancy <= total_inflight; no ring ever
            overwrites a slot the other side has not consumed.
        I6  conservation at quiesce (see check_quiesced).
        I7  the row lease: while the CPU reads a completed row, the row still
            belongs to that rid and holds its prompt and its generated tokens.
        I8  the attention-visible sequence length derived from the page table
            equals the true KV length (checked in GPU._exec).
        I9  the CPU's free-page estimate (free-count mirror + pages still in
            the return ring) is an upper bound on avail_uncommitted -- which is
            what makes the page attribution of backpressure conservative in the
            safe direction (§6.4 v2.1-c′).
        """
        cfg, gpu, cpu, fab = self.cfg, self.gpu, self.cpu, self.fab

        # I1: every physical page has exactly one owner state, and the owning
        # container really holds it: no page in two places, none lost.
        owner: Dict[int, str] = {}

        def claim(page: int, state: str, where: str) -> None:
            if page in owner:
                self.violation(
                    "I1", f"page {page} owned twice: {owner[page]} and {where}"
                )
            owner[page] = where
            if self.page_state[page] != state:
                self.violation(
                    "I1",
                    f"page {page} sits in {where} but its state is "
                    f"{self.page_state[page]}",
                )

        for i in range(gpu.pq_head, gpu.pq_tail):
            claim(gpu.page_queue[i % cfg.total_pages], FREE, "free-queue")
        for row in range(cfg.rows):
            for page in self.row_fresh[row]:
                claim(page, ROW, f"row{row}")
        for slot in range(cfg.ring_cap):
            if fab.comp_ready[slot]:
                for page in self.comp_fresh[slot]:
                    claim(page, COMP, f"comp{slot}")
        for block in cpu.blocks:
            claim(block.page, CACHE, "cache")
        for i in range(gpu.ret_head, fab.ret_tail):
            claim(fab.ret_ring[i & fab.ret_mask], RET, "return-ring")
        if len(owner) != cfg.total_pages:
            missing = [p for p in range(cfg.total_pages) if p not in owner]
            self.violation("I1", f"pages lost: {missing} (states {self.page_state})")

        # I2: the ledger identity is exact at all times.
        free_count = gpu.pq_tail - gpu.pq_head
        reserved = sum(gpu.reserved_remaining)
        if gpu.avail_uncommitted != free_count - reserved:
            self.violation(
                "I2",
                f"avail_uncommitted={gpu.avail_uncommitted} != free({free_count}) "
                f"- reserved({reserved})",
            )
        # I3: the queue never underflows and reservations never go negative.
        if free_count < 0 or free_count > cfg.total_pages:
            self.violation("I3", f"page queue occupancy {free_count} out of range")
        if gpu.avail_uncommitted < 0 or min(gpu.reserved_remaining) < 0:
            self.violation(
                "I3",
                f"ledger went negative: avail={gpu.avail_uncommitted} "
                f"reserved={gpu.reserved_remaining}",
            )
        # I4 (continuous half): a live row's imported pages stay cached and pinned.
        for row in range(cfg.rows):
            if fab.rid_at_row[row] < 0:
                continue
            for page in self.row_imported[row]:
                block = cpu.page_to_block.get(page)
                bad = self.page_state[page] != CACHE
                if bad or block is None or block.refcount <= 0:
                    self.violation(
                        "I4",
                        f"row {row} holds imported page {page} in state "
                        f"{self.page_state[page]} with no live refcount",
                    )
        # I5: ring occupancy bounds; no slot is overwritten while unconsumed.
        comp_occ = gpu.gpu_comp_tail - cpu.cpu_comp_head
        if comp_occ > cfg.rows or comp_occ > cfg.ring_cap:
            self.violation(
                "I5", f"completion ring occupancy {comp_occ} exceeds total_inflight"
            )
        if sum(1 for s in fab.comp_ready if s) != comp_occ:
            self.violation("I5", "completion ring ready flags disagree with cursors")
        req_occ = cpu.cpu_req_tail - gpu.gpu_req_head
        if req_occ > cfg.ring_cap or sum(1 for s in fab.req_ready if s) != req_occ:
            self.violation("I5", f"request ring occupancy {req_occ} inconsistent")
        ret_occ = fab.ret_tail - gpu.ret_head
        if ret_occ > fab.ret_cap:
            self.violation("I5", f"return ring occupancy {ret_occ} exceeds capacity")
        # The head mirror is a lagging observation of the GPU's private head,
        # so the CPU's occupancy estimate is always an upper bound.
        if not fab.ret_head_mirror <= gpu.ret_head <= fab.ret_tail:
            self.violation("I5", "return-ring head mirror ran ahead of the head")
        # I9: since the last mirror publish the free queue can only have grown
        # by pages the GPU drained out of the return ring -- which the CPU's
        # (tail - head_mirror) term already counts -- so the estimate never
        # undercounts what the GPU could still hand out.  Backpressure suppresses
        # itself on this estimate, and suppressing a real page stall would wedge
        # admission, so the bound has to hold at every step, not on average.
        estimate = fab.free_count_mirror + (fab.ret_tail - fab.ret_head_mirror)
        if estimate < gpu.avail_uncommitted:
            self.violation(
                "I9",
                f"CPU free estimate {estimate} < avail_uncommitted "
                f"{gpu.avail_uncommitted} (mirror={fab.free_count_mirror})",
            )

    def check_quiesced(self) -> None:
        """I6: conservation once everything has drained."""
        cfg, gpu, cpu, fab = self.cfg, self.gpu, self.cpu, self.fab
        free_count = gpu.pq_tail - gpu.pq_head
        cached = len(cpu.blocks)
        in_transit = (fab.ret_tail - gpu.ret_head) + sum(
            len(self.comp_fresh[s]) for s in range(cfg.ring_cap) if fab.comp_ready[s]
        )
        owned = sum(len(s) for s in self.row_fresh)
        if free_count + cached + in_transit + owned != cfg.total_pages:
            self.violation(
                "I6",
                f"conservation: free={free_count} cache={cached} "
                f"in_transit={in_transit} rows={owned} != {cfg.total_pages}",
            )
        if owned or in_transit:
            self.violation("I6", "quiesce reached with pages still in flight")

    def outstanding(self) -> bool:
        return bool(
            self.cpu.pending_submits
            or self.cpu.published
            or self.cpu.pending_acks
            or any(r >= 0 for r in self.fab.rid_at_row)
            or any(self.fab.comp_ready)
            or self.fab.ret_tail != self.gpu.ret_head
        )

    # ── the randomized scheduler ──────────────────────────────────────────

    def _tick_guard(self) -> None:
        self.check()
        self.events_since_progress += 1
        self.max_stall = max(self.max_stall, self.events_since_progress)
        if self.events_since_progress > self.cfg.stall_limit and self.outstanding():
            raise LivenessViolation(
                f"no admission or completion for {self.events_since_progress} "
                f"events with work outstanding\n{self.dump()}"
            )

    def step_gpu(self) -> str:
        self.event_no += 1
        name = self.gpu.step_once()
        self._tick_guard()
        return name

    def step_cpu(self, stage: Optional[str] = None) -> str:
        self.event_no += 1
        if stage is None:
            name = self.cpu.tick()
        else:
            self.cpu.stage = self.cpu.STAGES.index(stage)
            name = self.cpu.tick()
        self._tick_guard()
        return name

    def step_submit(self, req: Optional[Request] = None) -> None:
        self.event_no += 1
        self.cpu.submit(req if req is not None else self.workload.next_request())
        self._tick_guard()

    def make_request(
        self, tokens: Sequence[int], eos_at: Optional[int] = None
    ) -> Request:
        rid = self.workload.next_rid
        self.workload.next_rid += 1
        return Request(rid=rid, tokens=tuple(tokens), eos_at=eos_at)

    def gpu_run_to(self, substep: str) -> None:
        """Advance the GPU until ``substep`` is the next thing it will run."""
        for _ in range(2 * len(GPU.SUBSTEPS)):
            if GPU.SUBSTEPS[self.gpu.substep] == substep:
                return
            self.step_gpu()
        raise AssertionError(f"substep {substep} not reached")

    def run_random(self, events: int) -> "Sim":
        cfg = self.cfg
        for _ in range(events):
            roll = self.rng.random()
            if roll < 0.50:
                self.step_gpu()
            elif roll < 0.85:
                self.step_cpu()
            elif len(self.cpu.pending_submits) < cfg.max_pending_submits:
                self.step_submit()
            else:
                self.step_gpu()
        return self

    def drive_until(self, predicate, max_events: int = 200_000) -> "Sim":
        while not predicate():
            if max_events <= 0:
                raise LivenessViolation(f"drive_until gave up\n{self.dump()}")
            max_events -= 1
            if self.rng.random() < 0.55:
                self.step_gpu()
            else:
                self.step_cpu()
        return self

    def quiesce(self, max_events: int = 400_000) -> "Sim":
        self.drive_until(lambda: not self.outstanding(), max_events)
        self.check_quiesced()
        return self


# ── configuration sweep ───────────────────────────────────────────────────

def sweep_configs() -> List[Config]:
    return [
        # Tight pools on purpose: pool == one worst-case request.
        Config(name="pool8", total_pages=8, max_prompt=40),
        Config(name="pool12", total_pages=12, max_prompt=40),
        Config(name="pool24", total_pages=24, max_prompt=64),
        # Degenerate: every prompt is shorter than one page, so npp is always 0.
        Config(name="short-prompts", total_pages=12, max_prompt=7),
        # Slow return-ring drain: pages take several iterations to come back.
        Config(name="slow-drain", total_pages=12, drain_limit=1, max_prompt=40),
        # Roomy pool: all four rows run concurrently, so the same cached page
        # can sit in two rows' page tables at once (§7-S2 read sharing), and
        # ring_cap == total_inflight puts the completion ring at its S4 bound.
        Config(name="roomy", total_pages=40, ring_cap=4, max_prompt=40),
    ]


def run_config(cfg: Config, seed: int, events: int, verbose: bool = True) -> Sim:
    sim = Sim(cfg, seed=seed)
    sim.run_random(events)
    sim.quiesce()
    if verbose:
        print(
            f"  {cfg.name:<14} seed={seed:<4} events={events:<7} "
            f"iters={sim.gpu.iterations:<6} "
            f"adm={sim.stats['admissions']:<5} done={sim.stats['completions']:<5} "
            f"hit={sim.stats['hits']:<4} hit_tok={sim.stats['hit_tokens']:<5} "
            f"ins={sim.stats['inserts']:<4} dedup={sim.stats['dedupes']:<4} "
            f"evict={sim.stats['evictions']:<4} "
            f"forced={sim.stats['forced_evictions']:<4} "
            f"bp={sim.stats['backpressure_fired']}/"
            f"{sim.stats['backpressure_suppressed']:<6} "
            f"defer={sim.stats['admission_deferred']:<5} "
            f"max_stall={sim.max_stall}"
        )
    return sim


# ── handwritten adversarial schedules ─────────────────────────────────────

def _aligned_prompt(sim: Sim, blocks: int, extra: int = 1, tag: int = 0) -> List[int]:
    ps = sim.cfg.page_size
    return [200 + tag * 50 + i for i in range(blocks * ps + extra)]


def scenario_toctou(rounds: int = 16, seed: int = 1) -> Sim:
    """Review item §10-1: evict -> return -> publish lands between Step 0 and
    Step 4, so the newly published request is financed by a page the GPU has
    not drained yet.  With the ledger this must be SAFE: admission defers."""
    cfg = Config(
        name="toctou", total_pages=10, max_prompt=24, drain_limit=1, cache_budget=4
    )
    sim = Sim(cfg, seed=seed)
    for round_no in range(rounds):
        # Odd rounds reuse one two-block prefix (so the publish that lands
        # mid-iteration carries imported pages); even rounds are cold prompts
        # that keep the pool under pressure and force eviction + deferral.
        if round_no % 2:
            tokens = _aligned_prompt(sim, blocks=2, extra=1 + round_no % 3, tag=0)
        else:
            tokens = _aligned_prompt(sim, blocks=2, extra=1, tag=2 + round_no)
        req = sim.make_request(tokens, eos_at=len(tokens) + 2)
        sim.step_submit(req)
        # Park the GPU right after Step 0 for this iteration ...
        sim.gpu_run_to("step1")
        # ... then let the CPU evict (page -> return ring, NOT yet visible to
        # the free queue) and publish a request financed by exactly that page.
        sim.step_cpu("drain")
        sim.step_cpu("evict")
        sim.step_cpu("publish")
        sim.step_cpu("ack")
        # ... and only now run Step 1 / compaction / admission.
        sim.gpu_run_to("exec")
        sim.drive_until(lambda: not sim.cpu.pending_submits and not sim.cpu.published,
                        max_events=40_000)
    sim.quiesce()
    # The schedule is only interesting if it actually hit all three paths.
    assert sim.stats["completions"] == rounds
    assert sim.stats["admission_deferred"] > 0, "no admission ever deferred"
    assert sim.stats["evictions"] > 0, "no page ever evicted into the return ring"
    assert sim.stats["hits"] > 0, "no publish ever carried imported prefix pages"
    return sim


def scenario_duplicate_prompts(seed: int = 2) -> Sim:
    """Two identical prompts in flight: the second completion must dedupe,
    and a third submission must hit the single cached copy."""
    cfg = Config(name="dupes", total_pages=24, max_prompt=40)
    sim = Sim(cfg, seed=seed)
    tokens = _aligned_prompt(sim, blocks=2, extra=1)
    for _ in range(2):
        sim.step_submit(sim.make_request(tokens, eos_at=len(tokens) + 1))
    sim.drive_until(lambda: sim.stats["completions"] == 2 and not sim.cpu.published)
    assert sim.stats["dedupes"] >= 1, "identical concurrent prompts were not deduped"
    assert len(sim.cpu.blocks) == 2, f"cache holds {len(sim.cpu.blocks)} blocks, want 2"
    sim.step_submit(sim.make_request(tokens, eos_at=len(tokens) + 1))
    sim.drive_until(lambda: sim.stats["completions"] == 3 and not sim.cpu.published)
    assert sim.stats["hits"] >= 1, "the third identical prompt did not hit"
    sim.quiesce()
    return sim


def scenario_evict_after_match(seed: int = 3) -> Sim:
    """A block matched at publish must survive eviction until its holder
    completes (match-at-publish takes the refcount atomically)."""
    cfg = Config(name="evict-race", total_pages=24, max_prompt=40, cache_budget=0)
    sim = Sim(cfg, seed=seed)
    tokens = _aligned_prompt(sim, blocks=2, extra=1)
    sim.step_submit(sim.make_request(tokens, eos_at=len(tokens) + 1))
    sim.drive_until(lambda: sim.stats["completions"] == 1 and not sim.cpu.published)
    sim.step_submit(sim.make_request(tokens, eos_at=len(tokens) + 1))
    sim.step_cpu("publish")
    assert sim.stats["hits"] == 1, "prefix was not matched at publish"
    pinned = [b for b in sim.cpu.blocks if b.refcount > 0]
    assert pinned, "match-at-publish did not pin its blocks"
    for _ in range(6):  # cache_budget=0: eviction fires every tick
        sim.step_cpu("evict")
    assert all(b in sim.cpu.blocks for b in pinned), "a pinned block was evicted"
    sim.drive_until(lambda: sim.stats["completions"] == 2 and not sim.cpu.published)
    sim.drive_until(lambda: not any(b.refcount for b in sim.cpu.blocks))
    for _ in range(8):
        sim.step_cpu("evict")
    assert not sim.cpu.blocks, "unpinned blocks were not evicted at budget 0"
    sim.quiesce()
    return sim


def scenario_full_pool_churn(requests: int = 24, seed: int = 4) -> Sim:
    """Pool == one worst-case request: every admission must be financed by a
    page that came back through the return ring."""
    cfg = Config(name="churn", total_pages=8, max_prompt=32, drain_limit=2)
    sim = Sim(cfg, seed=seed)
    for i in range(requests):
        tokens = _aligned_prompt(sim, blocks=1 + i % 3, extra=1, tag=i % 4)
        sim.step_submit(sim.make_request(tokens, eos_at=len(tokens) + (i % 5)))
        sim.drive_until(lambda: len(sim.cpu.pending_submits) <= 1, max_events=60_000)
    sim.drive_until(lambda: sim.stats["completions"] == requests)
    sim.quiesce()
    return sim


def scenario_publish_pin_deadlock(bug: str = "", seed: int = 5) -> Sim:
    """Two published-but-unadmitted requests pin enough cached pages that the
    ring head (which matched nothing) can never be admitted.  The ledger keeps
    this SAFE; only the publish-time pin budget keeps it LIVE."""
    cfg = Config(
        name="pin-deadlock",
        total_pages=12,
        max_prompt=32,
        cache_budget=6,
        stall_limit=4000,
        bug=bug,
    )
    sim = Sim(cfg, seed=seed)
    chains = [_aligned_prompt(sim, blocks=3, extra=1, tag=t) for t in (0, 1)]
    for i, tokens in enumerate(chains):  # warm the cache: two 3-block chains
        sim.step_submit(sim.make_request(tokens, eos_at=len(tokens) + 1))
        sim.drive_until(
            lambda i=i: sim.stats["completions"] == i + 1 and not sim.cpu.published,
            max_events=60_000,
        )
    assert len(sim.cpu.blocks) == 6, f"warm-up cached {len(sim.cpu.blocks)} blocks"
    cold = [700 + i for i in range(20)]
    sim.step_submit(sim.make_request(cold, eos_at=len(cold) + 1))
    sim.step_cpu("publish")  # ring head, npp=0, needs the whole worst case
    for tokens in chains:
        sim.step_submit(sim.make_request(tokens, eos_at=len(tokens) + 1))
        sim.step_cpu("publish")
    sim.drive_until(lambda: sim.stats["completions"] == 5, max_events=80_000)
    sim.quiesce()
    return sim


def scenario_wrong_prefix_hit(bug: str = "", seed: int = 6) -> Sim:
    """A block whose ids match but whose context does not must never be
    imported.  Two heads are built to collide in the chained hash, so the
    per-block id compare alone would accept the wrong block; only the parent
    link rejects it (§7 S6)."""
    cfg = Config(
        name="wrong-prefix",
        total_pages=40,
        ring_cap=4,
        cache_budget=12,
        max_prompt=40,
        bug=bug,
    )
    sim = Sim(cfg, seed=seed)
    ps = cfg.page_size
    head_a = [200 + i for i in range(ps)]
    head_b = [head_a[0] + 97] + head_a[1:]  # same chained-hash bucket, other ids
    shared = [300 + i for i in range(ps)]
    other = [400 + i for i in range(ps)]
    assert weak_hash(weak_hash(0, head_a), shared) == weak_hash(
        weak_hash(0, head_b), shared
    ), "the scenario needs the two chains to collide"
    warm = [head_a + shared + [500], head_b + other + [501]]
    for i, tokens in enumerate(warm):
        sim.step_submit(sim.make_request(tokens, eos_at=len(tokens) + 1))
        sim.drive_until(
            lambda i=i: sim.stats["completions"] == i + 1 and not sim.cpu.published
        )
    assert len(sim.cpu.blocks) == 4, f"warm-up cached {len(sim.cpu.blocks)} blocks"
    probe = head_b + shared + [502]  # head_b's context, head_a's second block
    req = sim.make_request(probe, eos_at=len(probe) + 1)
    sim.step_submit(req)
    sim.step_cpu("publish")
    npp = sim.cpu.published[req.rid].npp
    if not bug:
        assert npp == 1, f"matched {npp} blocks, only the head is really this prefix"
    sim.drive_until(lambda: sim.stats["completions"] == 3 and not sim.cpu.published)
    sim.quiesce()
    return sim


def scenario_row_starvation(bug: str = "", seed: int = 7) -> Sim:
    """Rows exhausted, pages plentiful: backpressure must NOT force-evict.

    One row, a 64-page pool and a request that runs to max_seq, so the next
    request sits on the ring head unadmitted for as long as the batch is full
    while ~7/8 of the pool is free.  This is the shape P2 measured on hardware
    (12 in flight on 4 rows, 47-56/64 pages free, 1061 backpressure ticks in
    0.8 s, 163 blocks force-evicted): eviction cannot buy a row, so the only
    thing the pre-v2.1-c′ rule bought was an empty cache.  With the page
    attribution the whole stall must pass without a single forced eviction.
    """
    cfg = Config(
        name="row-starve",
        rows=1,
        total_pages=64,
        max_prompt=40,
        stall_ticks=3,
        bug=bug,
    )
    sim = Sim(cfg, seed=seed)
    # Warm two independent chains so force-eviction would have victims to take.
    for tag in (0, 1):
        tokens = _aligned_prompt(sim, blocks=2, extra=1, tag=tag)
        sim.step_submit(sim.make_request(tokens, eos_at=len(tokens) + 1))
        sim.drive_until(
            lambda: not sim.cpu.published and not sim.cpu.pending_submits,
            max_events=60_000,
        )
    assert len(sim.cpu.blocks) == 4, f"warm-up cached {len(sim.cpu.blocks)} blocks"
    warm_pages = {b.page for b in sim.cpu.blocks}

    # A long-runner takes the only row; a cold request then heads the ring.
    hog = _aligned_prompt(sim, blocks=2, extra=1, tag=5)
    sim.step_submit(sim.make_request(hog, eos_at=None))  # runs to max_seq
    sim.drive_until(lambda: sim.gpu.free_row_top == 0, max_events=60_000)
    cold = _aligned_prompt(sim, blocks=1, extra=1, tag=6)
    sim.step_submit(sim.make_request(cold, eos_at=len(cold) + 1))
    sim.drive_until(lambda: sim.stats["completions"] == 4, max_events=200_000)
    sim.quiesce()
    assert sim.stats["backpressure_suppressed"] > 0 or bug, (
        "the schedule never stalled the ring head, so it proves nothing"
    )
    if not bug:
        assert sim.stats["forced_evictions"] == 0, "row starvation force-evicted"
        assert sim.stats["evictions"] == 0, "a page was evicted with the pool idle"
        assert warm_pages <= {b.page for b in sim.cpu.blocks}, "the cache was stripped"
    return sim


def scenario_page_starvation(bug: str = "", seed: int = 8) -> Sim:
    """Rows free, pool exhausted: backpressure MUST force-evict.

    The pool is exactly one worst-case request, so the cached blocks of the
    first request are precisely what keeps the second (which matched nothing,
    and so needs the whole worst case) below the GPU's admission gate.  Nothing
    completes to release them and the capacity budget is not exceeded, so the
    only thing that can unblock admission is the stalled-head rule firing.
    """
    cfg = Config(
        name="page-starve",
        total_pages=8,
        max_prompt=32,
        cache_budget=4,
        stall_ticks=3,
        stall_limit=4000,
        bug=bug,
    )
    sim = Sim(cfg, seed=seed)
    warm = _aligned_prompt(sim, blocks=2, extra=1, tag=0)
    sim.step_submit(sim.make_request(warm, eos_at=len(warm) + 1))
    sim.drive_until(
        lambda: sim.stats["completions"] == 1 and not sim.cpu.published,
        max_events=60_000,
    )
    assert len(sim.cpu.blocks) == 2, f"warm-up cached {len(sim.cpu.blocks)} blocks"
    cold = [700 + i for i in range(20)]  # npp=0: needs the whole pool
    sim.step_submit(sim.make_request(cold, eos_at=len(cold) + 1))
    sim.drive_until(lambda: sim.stats["completions"] == 2, max_events=200_000)
    sim.quiesce()
    assert sim.stats["forced_evictions"] > 0, "admission unblocked without eviction"
    assert sim.stats["backpressure_fired"] > 0
    return sim


SCENARIOS = {
    "toctou": scenario_toctou,
    "duplicates": scenario_duplicate_prompts,
    "evict-after-match": scenario_evict_after_match,
    "full-pool-churn": scenario_full_pool_churn,
    "publish-pin-deadlock": scenario_publish_pin_deadlock,
    "wrong-prefix-hit": scenario_wrong_prefix_hit,
    "row-starvation": scenario_row_starvation,
    "page-starvation": scenario_page_starvation,
}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sweep", action="store_true", help="run the config sweep")
    parser.add_argument("--events", type=int, default=300_000)
    parser.add_argument("--seeds", type=int, default=1, help="seeds per config")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pool", type=int, default=24)
    parser.add_argument("--page-size", type=int, default=8)
    parser.add_argument("--max-seq", type=int, default=64)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--ring-cap", type=int, default=8)
    parser.add_argument("--max-prompt", type=int, default=40)
    parser.add_argument("--bug", default="", choices=sorted(BUGS))
    parser.add_argument(
        "--scenarios", action="store_true", help="run the adversarial schedules"
    )
    args = parser.parse_args(argv)

    failures = 0
    if args.scenarios or args.sweep:
        print("adversarial schedules:")
        for name, fn in SCENARIOS.items():
            try:
                sim = fn()
                print(
                    f"  {name:<22} OK  (completions={sim.stats['completions']}, "
                    f"hits={sim.stats['hits']}, "
                    f"defers={sim.stats['admission_deferred']})"
                )
            except AssertionError as exc:
                failures += 1
                print(f"  {name:<22} FAILED\n{exc}")
    if args.sweep:
        print(f"config sweep ({args.events} events x {args.seeds} seed(s)):")
        for cfg in sweep_configs():
            for s in range(args.seeds):
                try:
                    run_config(cfg, seed=args.seed + s, events=args.events)
                except AssertionError as exc:
                    failures += 1
                    print(f"  {cfg.name} seed={args.seed + s} FAILED\n{exc}")
    if not args.sweep and not args.scenarios:
        cfg = Config(
            name="cli",
            page_size=args.page_size,
            max_seq=args.max_seq,
            total_pages=args.pool,
            rows=args.rows,
            ring_cap=args.ring_cap,
            max_prompt=args.max_prompt,
            bug=args.bug,
        )
        try:
            run_config(cfg, seed=args.seed, events=args.events)
        except AssertionError as exc:
            failures += 1
            print(f"  FAILED\n{exc}")
    print(f"FAILURES: {failures}" if failures else "all green")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
