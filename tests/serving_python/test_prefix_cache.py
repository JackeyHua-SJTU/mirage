"""Unit tests for the OIPL prompt-prefix page cache.

Ports the adversarial scenarios the protocol simulator
(``oipl_protocol_sim.py``) uses to justify the §6.4 v2.1 rules — wrong-prefix
hit, dedupe, evict-after-match, publish pin budget — onto the real cache
index, and adds a randomized property run that re-checks the structural
invariants after every operation.  Pure stdlib: the cache imports no torch, so
this runs anywhere, including without a built ``mirage``.
"""

import importlib.util
import math
import pathlib
import random
import sys
import zlib

_SRC = (
    pathlib.Path(__file__).resolve().parents[2]
    / "python"
    / "mirage"
    / "mpk"
    / "prefix_cache.py"
)
# Loaded by path: the cache is stdlib-only, so these tests do not need a built
# ``mirage`` (importing the package would pull in torch and the compiler).
_spec = importlib.util.spec_from_file_location("mpk_prefix_cache", _SRC)
prefix_cache = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = prefix_cache  # dataclasses resolves annotations here
_spec.loader.exec_module(prefix_cache)

PrefixCache = prefix_cache.PrefixCache
PS = 4


def weak_hash(parent_hash, ids):
    """Deliberately collision-prone: hits must be decided by the id + parent."""
    return (parent_hash * 31 + sum(ids)) % 97


class Pool:
    """Stand-in for the GPU page allocator, so every page has exactly one home."""

    def __init__(self, total):
        self.free = list(range(total))
        self.out = set()

    def take(self, count):
        assert count <= len(self.free)
        pages = [self.free.pop(0) for _ in range(count)]
        self.out.update(pages)
        return pages

    def give(self, pages):
        for page in pages:
            assert page in self.out, f"page {page} returned twice"
            self.out.discard(page)
            self.free.append(page)


def make_cache(
    total_pages=64,
    max_pages_per_req=8,
    capacity_pages=None,
    hash_fn=None,
    stall_ticks=prefix_cache.DEFAULT_STALL_TICKS,
):
    kwargs = {} if hash_fn is None else {"hash_fn": hash_fn}
    return PrefixCache(
        page_size=PS,
        total_pages=total_pages,
        max_pages_per_req=max_pages_per_req,
        capacity_pages=capacity_pages,
        stall_ticks=stall_ticks,
        **kwargs,
    )


def prompt_of(pages, extra=1, tag=0):
    """A page-aligned prompt with *extra* trailing tokens (so npp < pages)."""
    return [1000 * (tag + 1) + i for i in range(pages * PS + extra)]


def run_request(cache, pool, prompt, final_kv_len=None, gen_tail_len=0, budget=None):
    """One full request lifecycle: match at publish, export + insert at drain."""
    if final_kv_len is None:
        final_kv_len = len(prompt)
    match = cache.match(prompt, budget)
    num_pages = math.ceil(final_kv_len / PS)
    pages = list(match.page_ids) + pool.take(num_pages - match.num_prefix_pages)
    free = cache.insert(
        prompt,
        pages,
        final_kv_len,
        gen_tail_len,
        num_prefix_pages=match.num_prefix_pages,
    )
    pool.give(free)
    return match, pages, free


# ── basic behaviour ───────────────────────────────────────────────────────


def test_cold_then_warm():
    """A cold prompt caches its full pages; the same prompt then hits them."""
    cache, pool = make_cache(), Pool(64)
    prompt = prompt_of(3)
    cold, cold_pages, freed = run_request(cache, pool, prompt)
    assert cold.num_prefix_pages == 0
    assert cache.stats()["inserted"] == 3
    assert freed == cold_pages[3:]  # the partial tail page is not cacheable

    warm, _, _ = run_request(cache, pool, prompt)
    # initial_step = npp*PS must leave at least one prompt token to process.
    assert warm.num_prefix_pages == 3
    assert list(warm.page_ids) == cold_pages[:3]
    assert cache.stats()["hits"] == 1


def test_npp_never_consumes_the_whole_prompt():
    """A page-aligned prompt caps at prompt_len-1, so the kernel always steps."""
    cache, pool = make_cache(), Pool(64)
    prompt = prompt_of(2, extra=0)
    run_request(cache, pool, prompt)
    match = cache.match(prompt)
    assert match.num_prefix_pages == 1


def test_partial_prefix_extends_the_chain():
    """A longer prompt sharing a prefix hits it and caches only its new pages."""
    cache, pool = make_cache(), Pool(64)
    base = prompt_of(2)
    run_request(cache, pool, base)
    longer = base[: 2 * PS] + [7000 + i for i in range(2 * PS + 1)]
    match, _, _ = run_request(cache, pool, longer)
    assert match.num_prefix_pages == 2
    assert cache.stats()["inserted"] == 4
    assert cache.resident_pages == 4


# ── §6.4 v2.1-b: the parent link is load-bearing ──────────────────────────


def test_wrong_prefix_hit_rejected_by_parent_link():
    """Two chains built to collide: only the parent link rejects the impostor.

    ``head_a`` and ``head_b`` hash into the same bucket, so the block holding
    ``shared`` under ``head_a`` and the one holding it under ``head_b`` are
    indistinguishable by bucket + id compare — but their KV is different
    context.  Importing the wrong one is a silent mis-hit, not a miss (§7 S6).
    """
    cache, pool = make_cache(hash_fn=weak_hash), Pool(64)
    head_a = [200 + i for i in range(PS)]
    head_b = [head_a[0] + 97] + head_a[1:]
    shared = [300 + i for i in range(PS)]
    other = [400 + i for i in range(PS)]
    assert weak_hash(0, head_a) == weak_hash(0, head_b)
    assert weak_hash(weak_hash(0, head_a), shared) == weak_hash(
        weak_hash(0, head_b), shared
    )

    _, a_pages, _ = run_request(cache, pool, head_a + shared + [500])
    _, b_pages, _ = run_request(cache, pool, head_b + other + [501])
    assert cache.resident_pages == 4

    # The bucket really does hold two same-id blocks under different parents,
    # so an id-only compare would have accepted the wrong one here.
    bucket = cache._index[weak_hash(weak_hash(0, head_b), tuple(shared))]
    same_ids = [b for b in bucket if b.token_ids == tuple(shared)]
    assert len(same_ids) == 1  # head_b's chain never cached `shared`
    collided = cache._index[weak_hash(0, tuple(head_b))]
    assert {b.token_ids for b in collided} == {tuple(head_a), tuple(head_b)}

    probe = head_b + shared + [502]
    match = cache.match(probe)
    assert match.num_prefix_pages == 1, "matched a block from the other chain"
    assert list(match.page_ids) == b_pages[:1]
    cache.release(match.blocks)

    # The same second block *is* reachable from its own context.
    match = cache.match(head_a + shared + [503])
    assert list(match.page_ids) == a_pages[:2]


# ── dedupe ────────────────────────────────────────────────────────────────


def test_duplicate_prompts_dedupe_on_insert():
    """Two identical prompts in flight: the loser's pages go straight back."""
    cache, pool = make_cache(), Pool(64)
    prompt = prompt_of(2)
    first_match = cache.match(prompt)
    second_match = cache.match(prompt)
    assert first_match.num_prefix_pages == second_match.num_prefix_pages == 0

    first_pages = pool.take(3)
    second_pages = pool.take(3)
    pool.give(cache.insert(prompt, first_pages, len(prompt), 0))
    freed = cache.insert(prompt, second_pages, len(prompt), 0)
    pool.give(freed)

    assert cache.stats()["deduped"] == 2
    assert cache.stats()["inserted"] == 2
    assert cache.resident_pages == 2
    assert freed == second_pages, "the duplicate kept a page out of the ring"
    assert pool.out == set(first_pages[:2])


# ── §11-1 / v2.1-d: what is inside the cacheable range ────────────────────


def test_generation_tail_is_excluded():
    """The template's per-turn tail is not a stable prefix, so it is not cached."""
    cache, pool = make_cache(), Pool(64)
    prompt = prompt_of(3, extra=0)  # exactly 3 pages
    _, pages, freed = run_request(cache, pool, prompt, gen_tail_len=1)
    assert cache.stats()["inserted"] == 2
    assert freed == pages[2:]


def test_insert_range_clamped_by_final_kv_len():
    """A completion before prefill finished must not cache half-written pages."""
    cache, pool = make_cache(), Pool(64)
    prompt = prompt_of(4)
    pages = pool.take(2)
    pool.give(cache.insert(prompt, pages, final_kv_len=2 * PS - 1, gen_tail_len=0))
    assert cache.stats()["inserted"] == 1
    assert cache.resident_pages == 1


def test_tail_pages_all_return_to_the_ring():
    """Every page past the cacheable range is freed, not just the first one."""
    cache, pool = make_cache(), Pool(64)
    prompt = prompt_of(4, extra=2)  # 18 tokens -> 5 exported pages
    _, pages, freed = run_request(cache, pool, prompt, gen_tail_len=2 * PS + 1)
    assert cache.stats()["inserted"] == 2
    assert freed == pages[2:]
    assert len(freed) == 3


# ── refcount lifecycle ────────────────────────────────────────────────────


def test_refcount_lifecycle_and_double_export():
    """Pins follow match -> export, and a repeated export never frees a page."""
    cache, pool = make_cache(), Pool(64)
    prompt = prompt_of(2)
    _, cold_pages, _ = run_request(cache, pool, prompt)
    blocks = [cache._by_page[p] for p in cold_pages[:2]]

    first = cache.match(prompt)
    second = cache.match(prompt)
    assert [b.refcount for b in blocks] == [2, 2]
    assert cache.stats()["pinned_pages"] == 2

    # A published request that never made it to the ring gives its pins back.
    cache.release(second.blocks)
    assert [b.refcount for b in blocks] == [1, 1]

    exported = list(first.page_ids) + pool.take(1)
    pool.give(cache.insert(prompt, exported, len(prompt), 0, num_prefix_pages=2))
    assert [b.refcount for b in blocks] == [0, 0]
    assert cache.resident_pages == 2

    # A second export of the same pages must not push a pin negative or hand a
    # still-cached page to the return ring.
    assert cache.on_export(exported[:2]) == []
    assert [b.refcount for b in blocks] == [0, 0]
    assert cache.stats()["unpin_underflow"] == 2
    assert cache.resident_pages == 2


def test_pinned_blocks_survive_eviction():
    """Match-at-publish takes the pin atomically, so eviction cannot race it."""
    cache, pool = make_cache(capacity_pages=0), Pool(64)
    prompt = prompt_of(2)
    _, pages, _ = run_request(cache, pool, prompt)
    match = cache.match(prompt)
    assert match.num_prefix_pages == 2

    # capacity_pages=0, so every tick wants the whole cache gone.
    for _ in range(4):
        assert cache.tick() == [], "a pinned block was evicted"
    assert cache.resident_pages == 2

    cache.release(match.blocks)
    evicted = cache.tick()
    assert evicted == [pages[1], pages[0]]
    assert cache.resident_pages == 0
    pool.give(evicted)


# ── eviction order ────────────────────────────────────────────────────────


def test_evict_is_lru_leaf_first():
    """Children before parents, LRU among the leaves."""
    cache, pool = make_cache(), Pool(64)
    chain_a = prompt_of(3, tag=0)
    chain_b = prompt_of(2, tag=1)
    _, a_pages, _ = run_request(cache, pool, chain_a)
    _, b_pages, _ = run_request(cache, pool, chain_b)
    assert cache.resident_pages == 5

    order = cache.evict(5)
    assert order == [a_pages[2], a_pages[1], a_pages[0], b_pages[1], b_pages[0]]
    assert cache.stats()["evicted"] == 5
    assert cache.resident_pages == 0
    pool.give(order)


def test_match_refreshes_recency():
    """A matched chain moves to the back of the LRU order."""
    cache, pool = make_cache(), Pool(64)
    chain_a = prompt_of(1, tag=0)
    chain_b = prompt_of(1, tag=1)
    _, a_pages, _ = run_request(cache, pool, chain_a)
    _, b_pages, _ = run_request(cache, pool, chain_b)
    cache.release(cache.match(chain_a).blocks)
    assert cache.evict(1) == [b_pages[0]]


def test_tick_trims_to_capacity_and_backpressures():
    """Steady state trims to the budget; a stalled ring head keeps giving back."""
    cache, pool = make_cache(capacity_pages=4, max_pages_per_req=8), Pool(64)
    for tag in range(3):
        run_request(cache, pool, prompt_of(2, tag=tag))
    assert cache.resident_pages == 6
    pool.give(cache.tick())
    assert cache.resident_pages == 4
    assert cache.tick() == []

    # v2.1-c: the oldest published request is not being admitted, so the cache
    # walks the pool back one block per tick even though it is within budget.
    for _ in range(prefix_cache.DEFAULT_STALL_TICKS):
        assert cache.tick(oldest_unadmitted=7) == []
    for expected in (3, 2, 1, 0):
        pool.give(cache.tick(oldest_unadmitted=7))
        assert cache.resident_pages == expected
    # A different head resets the stall counter.
    assert cache.tick(oldest_unadmitted=8) == []


# ── §6.4 v2.1-a: publish pin budget ───────────────────────────────────────


def test_publish_pin_budget_bound():
    """sum(npp of unadmitted except the head) <= total - max_pages_per_req."""
    cache = make_cache(total_pages=12, max_pages_per_req=4)
    assert cache.publish_pin_budget([]) == 4
    assert cache.publish_pin_budget([3]) == 8  # the head's own pins do not count
    assert cache.publish_pin_budget([3, 5]) == 3
    assert cache.publish_pin_budget([3, 5, 4]) == 0
    assert cache.publish_pin_budget([3, 9, 9]) == 0


def test_match_clamped_by_pin_budget():
    """Over budget the publish sheds prefix pages rather than pinning the pool."""
    cache, pool = make_cache(total_pages=12, max_pages_per_req=4), Pool(64)
    prompt = prompt_of(3)
    _, pages, _ = run_request(cache, pool, prompt)
    blocks = [cache._by_page[p] for p in pages[:3]]

    clamped = cache.match(prompt, 2)
    assert clamped.num_prefix_pages == 2
    assert [b.refcount for b in blocks] == [1, 1, 0]
    cache.release(clamped.blocks)

    starved = cache.match(prompt, 0)
    assert starved.num_prefix_pages == 0
    assert [b.refcount for b in blocks] == [0, 0, 0]
    assert cache.stats()["misses"] == 2  # the cold run plus the starved match


# ── randomized property run ───────────────────────────────────────────────


def check_invariants(cache, pool, inflight):
    """Structural invariants that must hold after every single operation."""
    resident = set(cache._by_page)
    assert len(resident) == len(cache._blocks) == cache.stats()["resident_pages"]
    indexed = [b for bucket in cache._index.values() for b in bucket]
    assert len(indexed) == len(resident)
    assert set(indexed) == cache._blocks

    pins = {}
    for flight in inflight:
        for block in flight.blocks:
            pins[block] = pins.get(block, 0) + 1
    for block in cache._blocks:
        assert cache._by_page[block.page_id] is block
        parent_hash = block.parent.chain_hash if block.parent else 0
        assert block.chain_hash == cache._hash(parent_hash, block.token_ids)
        assert block in cache._index[block.chain_hash]
        if block.parent is not None:
            assert block.parent in cache._blocks, "a parent was evicted before it"
            assert block in block.parent.children
        for child in block.children:
            assert child in cache._blocks and child.parent is block
        assert block.refcount == pins.get(block, 0), "refcount != outstanding matches"

    # Every page is in exactly one place: free in the pool, or held by the cache
    # and/or by a request that has not exported it yet.
    held = set(resident)
    for flight in inflight:
        held.update(flight.pages)
    assert pool.out == held
    assert not pool.out.intersection(pool.free)
    assert len(pool.free) == len(set(pool.free))


class Flight:
    """A published-but-not-yet-drained request."""

    def __init__(self, rid, prompt, match, pages, final_kv_len):
        self.rid = rid
        self.prompt = prompt
        self.blocks = match.blocks
        self.num_prefix_pages = match.num_prefix_pages
        self.pages = pages
        self.final_kv_len = final_kv_len


def test_random_ops_preserve_invariants():
    """Random publish / complete / tick interleavings against every invariant."""
    rng = random.Random(20260901)
    cache = make_cache(
        total_pages=96, max_pages_per_req=6, capacity_pages=10, stall_ticks=3
    )
    pool = Pool(96)
    bases = [tuple(rng.randrange(100, 999) for _ in range(PS * k)) for k in (1, 2, 3)]
    inflight = []
    published = 0
    completed = 0

    for _ in range(1500):
        roll = rng.random()
        if roll < 0.45 and len(inflight) < 4:
            base = rng.choice(bases)
            tail = [rng.randrange(1, 9) for _ in range(rng.randint(1, 5))]
            prompt = list(base) + tail
            final_kv_len = len(prompt) + rng.randint(0, 6)
            num_pages = math.ceil(final_kv_len / PS)
            if num_pages > len(pool.free):
                continue
            budget = cache.publish_pin_budget([f.num_prefix_pages for f in inflight])
            match = cache.match(prompt, budget)
            pages = list(match.page_ids) + pool.take(num_pages - match.num_prefix_pages)
            inflight.append(Flight(published, prompt, match, pages, final_kv_len))
            published += 1
        elif roll < 0.9 and inflight:
            flight = inflight.pop(rng.randrange(len(inflight)))
            free = cache.insert(
                flight.prompt,
                flight.pages,
                flight.final_kv_len,
                gen_tail_len=rng.choice((0, 1)),
                num_prefix_pages=flight.num_prefix_pages,
            )
            kept = set(flight.pages) - set(free)
            pool.give(free)
            assert kept <= set(cache._by_page), "a kept page left the index"
            completed += 1
        else:
            # The owner tick sees the oldest published request; while that rid
            # stops changing the cache force-evicts on top of the budget.
            pool.give(cache.tick(inflight[0].rid if inflight else None))
        check_invariants(cache, pool, inflight)

    while inflight:
        flight = inflight.pop()
        pool.give(
            cache.insert(
                flight.prompt,
                flight.pages,
                flight.final_kv_len,
                0,
                num_prefix_pages=flight.num_prefix_pages,
            )
        )
        completed += 1
    check_invariants(cache, pool, inflight)
    while True:
        freed = cache.tick()
        if not freed:
            break
        pool.give(freed)
    assert cache.resident_pages <= cache.capacity_pages

    stats = cache.stats()
    assert completed == published > 100
    assert stats["hits"] > 20 and stats["inserted"] > 20
    assert stats["deduped"] > 0 and stats["evicted"] > 0
    assert stats["backpressure_ticks"] > 0
    assert stats["unpin_underflow"] == 0

    pool.give(cache.reset())
    assert cache.resident_pages == 0
    assert not pool.out
    assert sorted(pool.free) == list(range(96))


def test_gen_tail_len_without_chat_template():
    """A tokenizer with no template renders raw prompts: nothing to exclude."""

    class Raw:
        chat_template = None

    assert prefix_cache.compute_gen_tail_len(Raw()) == 0


class FakeChatTokenizer:
    """The structure of a Qwen3-style template, without transformers.

    Stable per-turn rendering, a ``<think>`` block stripped out of assistant
    history, and — with thinking disabled — an empty think block force-inserted
    into the generation prompt and dropped again on the next turn.  That last
    part is the only thing that makes a turn stop being a prefix of the next
    (§11-1), and it is what ``compute_gen_tail_len`` has to measure.
    """

    chat_template = "fake"

    def __init__(self, enable_thinking=True):
        self.default_thinking = enable_thinking

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=None
    ):
        if enable_thinking is None:
            enable_thinking = self.default_thinking
        out = []
        for message in messages:
            content = message["content"]
            if message["role"] == "assistant" and "</think>" in content:
                content = content.split("</think>", 1)[1].strip()
            out.append(f"<|im_start|> {message['role']} {content} <|im_end|>")
        if add_generation_prompt:
            out.append("<|im_start|> assistant")
            if not enable_thinking:
                out.append("<think> </think>")
        return " ".join(out)

    def __call__(self, text):
        ids = [zlib.crc32(word.encode()) for word in text.split()]
        return type("Encoding", (), {"input_ids": ids})()


def test_gen_tail_len_measures_the_forced_think_block():
    """Thinking on: a stable prefix.  Thinking off: exactly the forced tail."""
    assert prefix_cache.compute_gen_tail_len(FakeChatTokenizer()) == 0
    tail = prefix_cache.compute_gen_tail_len(
        FakeChatTokenizer(enable_thinking=False)
    )
    assert tail == 2  # "<think>" "</think>"

    # The property the cache depends on: turn k is a prefix of turn k+1 once
    # the measured tail is excluded.
    tokenizer = FakeChatTokenizer(enable_thinking=False)
    turn1 = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    def render(messages):
        return tokenizer(tokenizer.apply_chat_template(messages)).input_ids

    first = render(turn1)
    second = render(
        turn1
        + [
            {"role": "assistant", "content": "<think> hmm </think> Paris."},
            {"role": "user", "content": "And of Germany?"},
        ]
    )
    assert second[: len(first)] != first
    assert second[: len(first) - tail] == first[: len(first) - tail]
