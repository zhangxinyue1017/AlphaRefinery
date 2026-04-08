from __future__ import annotations

import os
import time
from typing import Sequence

from .models import LLMProviderConfig


class OpenAICompatProvider:
    """Thin provider wrapper around an OpenAI-compatible chat endpoint."""

    def __init__(self, config: LLMProviderConfig) -> None:
        self.config = config

    def generate(self, messages: Sequence[dict[str, str]]) -> str:
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("openai package is required for OpenAI-compatible providers") from exc

        # The workspace commonly uses HTTP(S) proxies and an optional SOCKS ALL_PROXY.
        # httpx can use HTTP proxies without extra deps, but SOCKS requires `socksio`.
        # To keep llm_refine usable in this environment, we temporarily drop ALL_PROXY
        # while preserving HTTP_PROXY / HTTPS_PROXY.
        removed_proxy_env = {
            key: os.environ.pop(key)
            for key in ("ALL_PROXY", "all_proxy")
            if os.environ.get(key)
        }
        try:
            client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
            )
            request_max_tokens = int(self.config.max_tokens)
            last_error: Exception | None = None
            for attempt in range(3):
                try:
                    response = client.chat.completions.create(
                        model=self.config.model,
                        messages=list(messages),
                        temperature=self.config.temperature,
                        max_tokens=request_max_tokens,
                    )
                except Exception as exc:
                    last_error = exc
                    if not self._is_retryable_error(exc) or attempt >= 2:
                        raise
                    time.sleep(1.5 * (attempt + 1))
                    continue

                choice = response.choices[0]
                message = choice.message
                content = self._extract_content(message.content)
                if content:
                    return content

                reasoning = getattr(message, "reasoning_content", None)
                refusal = getattr(message, "refusal", None)
                finish_reason = getattr(choice, "finish_reason", None)
                reasoning_text = self._extract_content(reasoning)
                refusal_text = self._extract_content(refusal)
                if (
                    finish_reason == "length"
                    and not reasoning_text
                    and not refusal_text
                    and attempt < 2
                ):
                    request_max_tokens = max(request_max_tokens + 1000, int(request_max_tokens * 1.5), 3000)
                    time.sleep(0.8 * (attempt + 1))
                    continue
                raise RuntimeError(
                    "provider returned empty content "
                    f"(finish_reason={finish_reason!r}, "
                    f"reasoning_present={bool(reasoning_text)}, "
                    f"refusal_present={bool(refusal_text)}, "
                    f"attempt={attempt + 1}, "
                    f"max_tokens={request_max_tokens})"
                )
            if last_error is not None:
                raise last_error
        finally:
            for key, value in removed_proxy_env.items():
                os.environ[key] = value
        raise RuntimeError("provider request loop exited without a response")

    @staticmethod
    def _extract_content(payload) -> str:
        if payload is None:
            return ""
        if isinstance(payload, str):
            return payload.strip()
        if isinstance(payload, list):
            parts: list[str] = []
            for item in payload:
                if isinstance(item, str):
                    text = item.strip()
                    if text:
                        parts.append(text)
                    continue
                if isinstance(item, dict):
                    text = str(item.get("text") or item.get("content") or "").strip()
                    if text:
                        parts.append(text)
                    continue
                text = str(getattr(item, "text", "") or getattr(item, "content", "") or "").strip()
                if text:
                    parts.append(text)
            return "\n".join(part for part in parts if part).strip()
        return str(payload).strip()

    @staticmethod
    def _is_retryable_error(exc: Exception) -> bool:
        text = str(exc).lower()
        retryable_markers = (
            "429",
            "ratelimiterror",
            "rate limit",
            "too many requests",
            "负载已饱和",
            "稍后再试",
            "502",
            "503",
            "504",
            "bad gateway",
            "gateway timeout",
            "temporarily unavailable",
            "internalservererror",
            "apiconnectionerror",
            "timeout",
        )
        return any(marker in text for marker in retryable_markers)
