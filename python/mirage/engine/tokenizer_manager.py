"""Thread-safe tokenizer/detokenizer wrapper."""

from __future__ import annotations

import threading
from typing import Any, Sequence

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


class ChatTemplateError(ValueError):
    """A messages list the chat template cannot be asked to render.

    Raised instead of letting a missing key surface as a ``KeyError``, so the
    HTTP layer can answer 400 for a malformed request body rather than 500.
    """


def _flatten_content(content: Any, where: str) -> str:
    """Collapse OpenAI content parts into the plain text a template renders."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
                continue
            if isinstance(part, dict) and part.get("type", "text") == "text":
                text = part.get("text")
                if not isinstance(text, str):
                    raise ChatTemplateError(
                        f"{where}: text content part has no 'text' string")
                parts.append(text)
                continue
            kind = part.get("type") if isinstance(part, dict) else type(part).__name__
            raise ChatTemplateError(
                f"{where}: unsupported content part of type {kind!r}")
        return "".join(parts)
    raise ChatTemplateError(
        f"{where}: 'content' must be a string or a list of content parts")


def normalize_messages(messages: Any) -> list[dict]:
    """Validate an OpenAI messages list and render it template-ready.

    Content parts are flattened to text and the default system message is
    prepended only when the client sent none, so a client that supplies its own
    system turn keeps a prompt that is byte-identical to what it asked for.
    Idempotent: a list this returned normalizes to itself.
    """
    if not isinstance(messages, list) or not messages:
        raise ChatTemplateError("'messages' must be a non-empty list")
    out: list[dict] = []
    has_system = False
    for index, message in enumerate(messages):
        where = f"messages[{index}]"
        if not isinstance(message, dict):
            raise ChatTemplateError(f"{where} is not an object")
        role = message.get("role")
        if not isinstance(role, str) or not role:
            raise ChatTemplateError(f"{where} has no 'role' string")
        if "content" not in message:
            raise ChatTemplateError(f"{where} ({role}) has no 'content'")
        out.append({
            "role": role,
            "content": _flatten_content(message["content"], where),
        })
        has_system = has_system or role == "system"
    if not has_system:
        out.insert(0, {"role": "system", "content": DEFAULT_SYSTEM_PROMPT})
    return out


class TokenizerManager:
    """Wraps a HuggingFace tokenizer for thread-safe tokenization/detokenization."""

    def __init__(self, tokenizer) -> None:
        self._tokenizer = tokenizer
        self._lock = threading.Lock()

    def tokenize_messages(
        self, messages: Sequence[dict], add_generation_prompt: bool = True
    ) -> list[int]:
        """Render a whole conversation through the chat template and tokenize.

        Rendering the full list — not just its last user turn — is what makes
        turn *k*'s prompt a token prefix of turn *k+1*'s, which is the shape the
        OIPL prefix cache matches on (§11-1).
        """
        rendered = normalize_messages(messages)
        text = self._tokenizer.apply_chat_template(
            rendered, tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        return self._encode(text)

    def tokenize(self, prompt: str, use_template: bool = True) -> list[int]:
        """Apply chat template (if requested) and return token IDs."""
        if use_template:
            return self.tokenize_messages([{"role": "user", "content": prompt}])
        return self._encode(prompt)

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs to text, skipping special tokens."""
        with self._lock:
            return self._tokenizer.decode(token_ids, skip_special_tokens=True)

    def decode_single(self, token_id: int) -> str:
        """Decode a single token ID to text."""
        with self._lock:
            return self._tokenizer.decode([token_id], skip_special_tokens=True)

    def _encode(self, text: str) -> list[int]:
        with self._lock:
            return self._tokenizer([text], return_tensors="pt").input_ids[0].tolist()
