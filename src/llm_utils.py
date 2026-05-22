"""LLM calling utilities: retry wrappers, message sanitisation, event helpers.

This module has **no** imports from the rest of the agent stack so it can be
used freely by any layer without creating circular dependencies.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import litellm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate-limit retry config
# ---------------------------------------------------------------------------

_RATE_LIMIT_MAX_RETRIES = int(os.getenv("AGENT_RATE_LIMIT_RETRIES", "2"))
_RATE_LIMIT_BASE_DELAY = float(os.getenv("AGENT_RATE_LIMIT_BASE_DELAY", "15.0"))


# ---------------------------------------------------------------------------
# LLM completion wrappers
# ---------------------------------------------------------------------------

def _make_final_answer(reason: str, content: str) -> dict[str, Any]:
    """Build a ``final_answer`` event that permanently closes the execution lane.

    ``reason`` is one of ``"iteration_limit"``, ``"exception"``,
    ``"rate_limited"``, or ``"critical_failure"``.
    The frontend should treat this event as a terminal signal -- no further
    events will follow from the same generator invocation.
    """
    return {"type": "final_answer", "reason": reason, "content": content}


async def _acompletion_with_retry(**kwargs: Any) -> Any:
    """Call litellm.acompletion with bounded retries for provider rate limits."""
    for attempt in range(_RATE_LIMIT_MAX_RETRIES + 1):
        try:
            return await litellm.acompletion(**kwargs)
        except Exception as exc:
            if not _is_rate_limit_error(exc) or attempt >= _RATE_LIMIT_MAX_RETRIES:
                raise
            delay = _RATE_LIMIT_BASE_DELAY * (attempt + 1)
            logger.warning(
                "LLM rate limited; retrying in %.1fs (%d/%d)",
                delay,
                attempt + 1,
                _RATE_LIMIT_MAX_RETRIES,
            )
            await asyncio.sleep(delay)

    raise RuntimeError("unreachable retry state")


async def _acompletion_stream_with_retry(**kwargs: Any) -> Any:
    """Call litellm.acompletion with stream=True, retrying on rate limits.

    Returns the async-iterable stream object so the caller can iterate chunks.
    """
    kwargs = {**kwargs, "stream": True}
    for attempt in range(_RATE_LIMIT_MAX_RETRIES + 1):
        try:
            return await litellm.acompletion(**kwargs)
        except Exception as exc:
            if not _is_rate_limit_error(exc) or attempt >= _RATE_LIMIT_MAX_RETRIES:
                raise
            delay = _RATE_LIMIT_BASE_DELAY * (attempt + 1)
            logger.warning(
                "LLM rate limited; retrying in %.1fs (%d/%d)",
                delay,
                attempt + 1,
                _RATE_LIMIT_MAX_RETRIES,
            )
            await asyncio.sleep(delay)
    raise RuntimeError("unreachable retry state")


def _is_rate_limit_error(exc: Exception) -> bool:
    rate_limit_cls = getattr(litellm, "RateLimitError", None)
    if rate_limit_cls is not None and isinstance(exc, rate_limit_cls):
        return True
    label = f"{type(exc).__name__}: {exc}".lower()
    return "ratelimit" in label or "rate_limit" in label or "rate limit" in label


def _rate_limit_user_message() -> str:
    return (
        "The model provider rate limit was reached for this minute. I paused this "
        "turn before doing more work; wait a short moment and send the message "
        "again. I also reduced the agent's per-call token budget so the next run "
        "should put less pressure on the limit."
    )


def _is_async_iterable(value: Any) -> bool:
    return hasattr(value, "__aiter__")


# ---------------------------------------------------------------------------
# Message preparation / sanitisation
# ---------------------------------------------------------------------------

def _prepare_llm_request_messages(
    messages: list[dict[str, Any]], original_prompt: str | None = None
) -> list[dict[str, Any]]:
    """Return provider-safe messages with at least one non-system turn."""
    prepared = _sanitize_messages_for_llm(messages)
    if any(msg.get("role") != "system" for msg in prepared):
        return prepared

    prompt = " ".join(str(original_prompt or "").split())
    fallback = (
        f"Continue working on the current task: {prompt[:1000]}"
        if prompt
        else "Please continue."
    )
    logger.warning(
        "LLM request contained only system messages after sanitization; "
        "injecting a transient user continuation message."
    )
    return [
        *prepared,
        {"role": "user", "content": fallback},
    ]


def _sanitize_messages_for_llm(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove provider-invalid empty text blocks without mutating history."""
    sanitized: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        cleaned = dict(msg)

        if isinstance(content, str):
            if content.strip():
                sanitized.append(cleaned)
                continue
            if role == "assistant" and cleaned.get("tool_calls"):
                cleaned["content"] = None
                sanitized.append(cleaned)
                continue
            if role == "tool":
                cleaned["content"] = "(empty result)"
                sanitized.append(cleaned)
            continue

        if isinstance(content, list):
            blocks: list[Any] = []
            for block in content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "text"
                    and not str(block.get("text") or "").strip()
                ):
                    continue
                blocks.append(block)
            if blocks:
                cleaned["content"] = blocks
                sanitized.append(cleaned)
                continue
            if role == "assistant" and cleaned.get("tool_calls"):
                cleaned["content"] = None
                sanitized.append(cleaned)
                continue
            if role == "tool":
                cleaned["content"] = "(empty result)"
                sanitized.append(cleaned)
            continue

        if content is None:
            if role == "assistant" and cleaned.get("tool_calls"):
                sanitized.append(cleaned)
            elif role == "tool":
                cleaned["content"] = "(empty result)"
                sanitized.append(cleaned)
            continue

        sanitized.append(cleaned)

    return sanitized
