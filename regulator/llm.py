"""Thin Anthropic client wrapper used during regulatory parsing.

The LLM's job is strictly to propose *structure* (a list of node ids with a
verbatim start anchor per node). It never authors body text. This module keeps
that contract narrow: send a stable instruction/schema block plus the volatile
chunk text, and return parsed JSON.

Prompt caching: the byte-stable instruction/rubric/schema block is sent as a
``system`` block with ``cache_control: {"type": "ephemeral"}`` so repeated
chunk calls reuse it. Only the per-chunk text varies, and it lives in the user
message after the cached prefix.
"""

from __future__ import annotations

import json
import os
from typing import Any

import anthropic

DEFAULT_MODEL = "claude-haiku-4-5"


def _strip_code_fences(text: str) -> str:
    """Remove a surrounding Markdown code fence if the model added one."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    # Drop the opening fence line (``` or ```json) and a trailing fence line.
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


class StructureLLM:
    """Anthropic-backed helper that returns parsed JSON structure proposals."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        # Anthropic() resolves ANTHROPIC_API_KEY from the environment when no
        # explicit key is passed; python-dotenv loads .env upstream.
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model or os.environ.get("PARSE_MODEL", DEFAULT_MODEL)

    @property
    def model(self) -> str:
        return self._model

    def propose_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 8000,
    ) -> dict[str, Any]:
        """Send ``system_prompt`` + ``user_prompt`` and return parsed JSON.

        The system block is marked for ephemeral prompt caching. Malformed JSON
        is retried once with a corrective nudge before giving up.
        """
        system_blocks = [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]

        raw = self._complete(system_blocks, user_prompt, max_tokens)
        try:
            return json.loads(_strip_code_fences(raw))
        except json.JSONDecodeError:
            # One corrective retry: hand the bad output back and ask for pure
            # JSON. Keeps a transient formatting slip from failing the parse.
            retry_prompt = (
                user_prompt
                + "\n\nYour previous reply was not valid JSON. Reply with ONLY "
                + "a single JSON object, no prose and no code fences."
            )
            raw = self._complete(system_blocks, retry_prompt, max_tokens)
            return json.loads(_strip_code_fences(raw))

    def _complete(
        self,
        system_blocks: list[dict[str, Any]],
        user_prompt: str,
        max_tokens: int,
    ) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system_blocks,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text")
