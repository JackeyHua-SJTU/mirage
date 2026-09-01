"""In-repo re-confirmation of the chat-template prefix stability (§11 item 1).

The OIPL prefix cache only ever hits across turns if the chat template renders
turn *k*'s prompt as a token-exact prefix of turn *k+1*'s prompt.  That was
measured on a newer transformers than the one this repo pins, so the design doc
left it open pending an in-repo test — this is that test.  It runs wherever a
Qwen3 tokenizer is present locally (the GPU box) and skips elsewhere; set
``MPK_PREFIX_CACHE_TEST_MODEL`` to point it at another model or path.
"""

import importlib.util
import os
import pathlib
import sys

import pytest

transformers = pytest.importorskip("transformers")

_SRC = (
    pathlib.Path(__file__).resolve().parents[2]
    / "python"
    / "mirage"
    / "mpk"
    / "prefix_cache.py"
)
_spec = importlib.util.spec_from_file_location("mpk_prefix_cache", _SRC)
prefix_cache = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = prefix_cache
_spec.loader.exec_module(prefix_cache)

CANDIDATE_MODELS = ("Qwen/Qwen3-8B", "Qwen/Qwen3-0.6B", "Qwen/Qwen3-4B")

TURN1 = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
]
# Long enough that whole 64-token pages fall inside the cacheable range.
LONG_TURN1 = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain paged attention. " * 40},
]
REPLIES = (
    "Paris.",
    "<think>\nThe user asks about France.\n</think>\n\nParis.",
)


@pytest.fixture(scope="module")
def tokenizer():
    names = [os.environ["MPK_PREFIX_CACHE_TEST_MODEL"]] if (
        "MPK_PREFIX_CACHE_TEST_MODEL" in os.environ
    ) else list(CANDIDATE_MODELS)
    for name in names:
        try:
            tok = transformers.AutoTokenizer.from_pretrained(
                name, local_files_only=True
            )
        except Exception:
            continue
        if getattr(tok, "chat_template", None):
            return tok
    pytest.skip(f"no local chat-template tokenizer among {names}")


def render(tokenizer, messages, **kwargs):
    """Tokenize a rendered prompt exactly the way TokenizerManager does."""
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **kwargs
    )
    return list(tokenizer(text).input_ids)


def next_turn(reply, turn1=TURN1):
    return turn1 + [
        {"role": "assistant", "content": reply},
        {"role": "user", "content": "And of Germany?"},
    ]


def shared_prefix_len(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


@pytest.mark.parametrize("reply", REPLIES, ids=("plain", "with-think"))
def test_default_mode_prefix_is_token_exact(tokenizer, reply):
    """Under the default (thinking) mode turn k is a prefix of turn k+1.

    Including when the history holds a ``<think>`` block: the template strips
    it identically on every render, so the token stream does not shift.
    """
    turn1 = render(tokenizer, TURN1)
    turn2 = render(tokenizer, next_turn(reply))
    assert turn2[: len(turn1)] == turn1, (
        f"turn 2 diverges at token {shared_prefix_len(turn1, turn2)} of "
        f"{len(turn1)} (transformers {transformers.__version__})"
    )


def test_default_mode_needs_no_tail_exclusion(tokenizer):
    assert prefix_cache.compute_gen_tail_len(tokenizer, probe_replies=REPLIES) == 0


def test_no_think_switch_only_costs_the_forced_tail(tokenizer):
    """``enable_thinking=False`` breaks the prefix, and only at the very end.

    The template force-inserts an empty ``<think></think>`` block into the
    generation prompt and drops it again next turn.  The damage must be a short
    suffix, which is exactly what ``compute_gen_tail_len`` measures and what
    the cache keeps out of the cacheable range.
    """
    try:
        turn1 = render(tokenizer, TURN1, enable_thinking=False)
    except TypeError:
        pytest.skip("template has no enable_thinking switch")
    turn2 = render(tokenizer, next_turn(REPLIES[0]), enable_thinking=False)
    tail = prefix_cache.compute_gen_tail_len(
        tokenizer, probe_replies=REPLIES, enable_thinking=False
    )
    assert tail == len(turn1) - shared_prefix_len(turn1, turn2)
    assert 0 < tail <= 8, f"unexpected generation-prompt tail of {tail} tokens"
    assert turn2[: len(turn1) - tail] == turn1[: len(turn1) - tail]


@pytest.mark.parametrize("enable_thinking", (True, False))
def test_cacheable_range_is_a_stable_prefix(tokenizer, enable_thinking):
    """The rule the cache actually applies: every page it may insert from turn
    k still holds the same tokens in turn k+1's prompt."""
    page_size = 64
    kwargs = {}
    if not enable_thinking:
        try:
            render(tokenizer, TURN1, enable_thinking=False)
        except TypeError:
            pytest.skip("template has no enable_thinking switch")
        kwargs["enable_thinking"] = False
    tail = prefix_cache.compute_gen_tail_len(
        tokenizer, probe_replies=REPLIES, **kwargs
    )
    turn1 = render(tokenizer, LONG_TURN1, **kwargs)
    cacheable = ((len(turn1) - tail) // page_size) * page_size
    assert cacheable >= page_size, "the probe prompt is shorter than one page"
    for reply in REPLIES:
        turn2 = render(tokenizer, next_turn(reply, LONG_TURN1), **kwargs)
        assert turn2[:cacheable] == turn1[:cacheable]
