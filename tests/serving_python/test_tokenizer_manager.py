"""Unit tests for the chat-template rendering behind multi-turn serving.

``TokenizerManager.tokenize_messages`` is what turns an OpenAI messages list
into the token ids the engine publishes, so what it renders decides whether
turn *k*'s prompt is a token prefix of turn *k+1*'s -- the shape the OIPL
prefix cache matches on.  The tests run against a fake tokenizer (no torch, no
transformers) plus, where a Qwen3 tokenizer is present locally, the real one::

    pytest tests/serving_python/test_tokenizer_manager.py
"""

import importlib.util
import os
import pathlib
import sys

import pytest

_SRC = (
    pathlib.Path(__file__).resolve().parents[2]
    / "python"
    / "mirage"
    / "engine"
    / "tokenizer_manager.py"
)
_spec = importlib.util.spec_from_file_location("mpk_tokenizer_manager", _SRC)
tokenizer_manager = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = tokenizer_manager
_spec.loader.exec_module(tokenizer_manager)

ChatTemplateError = tokenizer_manager.ChatTemplateError
DEFAULT_SYSTEM_PROMPT = tokenizer_manager.DEFAULT_SYSTEM_PROMPT
TokenizerManager = tokenizer_manager.TokenizerManager

CANDIDATE_MODELS = ("Qwen/Qwen3-8B", "Qwen/Qwen3-0.6B", "Qwen/Qwen3-4B")


class _Ids(list):
    def tolist(self):
        return list(self)


class _Encoding:
    def __init__(self, ids):
        self.input_ids = [_Ids(ids)]


class FakeTokenizer:
    """Chat-template-shaped renderer with a one-token-per-character encoder."""

    chat_template = "fake"

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True):
        assert tokenize is False
        text = "".join(
            f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
            for m in messages
        )
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return text

    def __call__(self, texts, return_tensors=None):
        return _Encoding([ord(char) for char in texts[0]])


@pytest.fixture
def manager():
    return TokenizerManager(FakeTokenizer())


def ids_of(text):
    return [ord(char) for char in text]


# ── rendering ─────────────────────────────────────────────────────────────────


def test_matches_a_manual_render_of_the_whole_conversation(manager):
    """The engine's ids are exactly those of a hand-rendered messages list."""
    conversation = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "Paris."},
        {"role": "user", "content": "And of Germany?"},
    ]
    expected = FakeTokenizer().apply_chat_template(
        [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}] + conversation,
        tokenize=False, add_generation_prompt=True,
    )
    assert manager.tokenize_messages(conversation) == ids_of(expected)


def test_turn_k_is_a_token_prefix_of_turn_k_plus_1(manager):
    """The property the prefix cache exists to exploit, minus the gen prompt."""
    turn1 = [{"role": "user", "content": "What is the capital of France?"}]
    turn2 = turn1 + [
        {"role": "assistant", "content": "Paris."},
        {"role": "user", "content": "And of Germany?"},
    ]
    ids1 = manager.tokenize_messages(turn1)
    ids2 = manager.tokenize_messages(turn2)
    shared = ids1[: len(ids1) - len(ids_of("<|im_start|>assistant\n"))]
    assert ids2[: len(shared)] == shared


def test_client_system_message_is_not_doubled(manager):
    messages = [
        {"role": "system", "content": "You are a pirate."},
        {"role": "user", "content": "hi"},
    ]
    rendered = "".join(chr(token) for token in manager.tokenize_messages(messages))
    assert rendered.count("<|im_start|>system") == 1
    assert "You are a pirate." in rendered
    assert DEFAULT_SYSTEM_PROMPT not in rendered


def test_default_system_message_is_injected_when_absent(manager):
    rendered = "".join(chr(token) for token in manager.tokenize_messages(
        [{"role": "user", "content": "hi"}]))
    assert rendered.startswith(
        f"<|im_start|>system\n{DEFAULT_SYSTEM_PROMPT}<|im_end|>\n")


def test_a_system_turn_anywhere_counts_as_client_supplied(manager):
    """A conversation replayed with its system turn keeps exactly that one."""
    rendered = "".join(chr(token) for token in manager.tokenize_messages([
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "Be terse."},
    ]))
    assert rendered.count("<|im_start|>system") == 1


def test_array_content_is_flattened_to_text(manager):
    array = manager.tokenize_messages([{"role": "user", "content": [
        {"type": "text", "text": "hello "},
        {"type": "text", "text": "world"},
    ]}])
    assert array == manager.tokenize_messages(
        [{"role": "user", "content": "hello world"}])


def test_prompt_path_still_wraps_a_single_turn(manager):
    """``submit(prompt=...)`` keeps the pre-A3 rendering (demo back-compat)."""
    assert manager.tokenize("hi") == manager.tokenize_messages(
        [{"role": "user", "content": "hi"}])


def test_raw_prompt_path_applies_no_template(manager):
    assert manager.tokenize("hi", use_template=False) == ids_of("hi")


# ── malformed input ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("messages", [
    None,
    [],
    "hello",
    [{"role": "user"}],
    [{"content": "hi"}],
    ["hi"],
    [{"role": "user", "content": 7}],
    [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}],
    [{"role": "user", "content": [{"type": "text"}]}],
])
def test_malformed_messages_raise_a_typed_error(manager, messages):
    with pytest.raises(ChatTemplateError):
        manager.tokenize_messages(messages)


def test_missing_content_names_the_offending_message(manager):
    with pytest.raises(ChatTemplateError) as excinfo:
        manager.tokenize_messages([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user"},
        ])
    assert "messages[2]" in str(excinfo.value)
    assert "content" in str(excinfo.value)


def test_normalize_messages_is_idempotent():
    """The HTTP layer normalizes to answer 400; the engine normalizes again."""
    once = tokenizer_manager.normalize_messages(
        [{"role": "user", "content": [{"type": "text", "text": "hi"}]}])
    assert tokenizer_manager.normalize_messages(once) == once


# ── the real template ─────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def hf_tokenizer():
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("torch")  # tokenize_messages asks for "pt" tensors
    names = [os.environ["MPK_PREFIX_CACHE_TEST_MODEL"]] if (
        "MPK_PREFIX_CACHE_TEST_MODEL" in os.environ
    ) else list(CANDIDATE_MODELS)
    for name in names:
        try:
            tok = transformers.AutoTokenizer.from_pretrained(
                name, local_files_only=True)
        except Exception:
            continue
        if getattr(tok, "chat_template", None):
            return tok
    pytest.skip(f"no local chat-template tokenizer among {names}")


def test_real_template_matches_a_manual_render(hf_tokenizer):
    conversation = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "Paris."},
        {"role": "user", "content": "And of Germany?"},
    ]
    text = hf_tokenizer.apply_chat_template(
        [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}] + conversation,
        tokenize=False, add_generation_prompt=True,
    )
    expected = list(hf_tokenizer(text).input_ids)
    assert TokenizerManager(hf_tokenizer).tokenize_messages(
        conversation) == expected
