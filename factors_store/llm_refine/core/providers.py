'''Provider adapters for OpenAI-compatible LLM calls.

Wraps chat completion generation using configured base URL, API key, model, and timeout.
'''

from __future__ import annotations

import json
import os
import subprocess
import time
from typing import Sequence

from .models import LLMProviderConfig


class OpenAICompatProvider:
    """Thin provider wrapper around an OpenAI-compatible chat endpoint."""

    def __init__(self, config: LLMProviderConfig) -> None:
        self.config = config
        self.request_count = 0
        self.last_usage: dict[str, object] | None = None
        self.last_response_id = ""
        self.retries_disabled = os.getenv("LLM_DISABLE_PROVIDER_RETRIES", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def telemetry(self) -> dict[str, object]:
        return {
            "provider": self.config.name,
            "model": self.config.model,
            "token_limit_parameter": self._token_limit_parameter(),
            "temperature_sent": self._supports_configured_temperature(),
            "request_count": int(self.request_count),
            "retries_disabled": bool(self.retries_disabled),
            "usage": self.last_usage,
            "response_id": self.last_response_id or None,
        }

    def _token_limit_parameter(self) -> str:
        provider = self.config.name.strip().lower()
        model = self.config.model.strip().lower()
        if provider == "openai" and (model.startswith("gpt-5") or model.startswith("o")):
            return "max_completion_tokens"
        return "max_tokens"

    def _supports_configured_temperature(self) -> bool:
        provider = self.config.name.strip().lower()
        model = self.config.model.strip().lower()
        return not (provider == "openai" and (model.startswith("gpt-5") or model.startswith("o")))

    def _record_response(self, response) -> None:
        self.last_response_id = str(getattr(response, "id", "") or "")
        usage = getattr(response, "usage", None)
        if usage is None:
            self.last_usage = None
        elif hasattr(usage, "model_dump"):
            self.last_usage = dict(usage.model_dump())
        elif isinstance(usage, dict):
            self.last_usage = dict(usage)
        else:
            self.last_usage = {
                key: getattr(usage, key)
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                if getattr(usage, key, None) is not None
            }

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
                max_retries=0 if self.retries_disabled else 2,
            )
            request_max_tokens = int(self.config.max_tokens)
            last_error: Exception | None = None
            max_attempts = 1 if self.retries_disabled else 3
            for attempt in range(max_attempts):
                try:
                    self.request_count += 1
                    token_limit = {self._token_limit_parameter(): request_max_tokens}
                    sampling = (
                        {"temperature": self.config.temperature}
                        if self._supports_configured_temperature()
                        else {}
                    )
                    response = client.chat.completions.create(
                        model=self.config.model,
                        messages=list(messages),
                        **sampling,
                        **token_limit,
                    )
                except Exception as exc:
                    last_error = exc
                    if not self._is_retryable_error(exc) or attempt >= max_attempts - 1:
                        if not self.retries_disabled and self._should_try_curl_fallback(exc):
                            return self._generate_via_curl(
                                messages=messages,
                                max_tokens=request_max_tokens,
                                restored_proxy_env=removed_proxy_env,
                            )
                        raise
                    time.sleep(1.5 * (attempt + 1))
                    continue

                self._record_response(response)
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
                    and attempt < max_attempts - 1
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
                if not self.retries_disabled and self._should_try_curl_fallback(last_error):
                    return self._generate_via_curl(
                        messages=messages,
                        max_tokens=request_max_tokens,
                        restored_proxy_env=removed_proxy_env,
                    )
                raise last_error
        finally:
            for key, value in removed_proxy_env.items():
                os.environ[key] = value
        raise RuntimeError("provider request loop exited without a response")

    def _generate_via_curl(
        self,
        *,
        messages: Sequence[dict[str, str]],
        max_tokens: int,
        restored_proxy_env: dict[str, str],
    ) -> str:
        payload = {
            "model": self.config.model,
            "messages": list(messages),
            self._token_limit_parameter(): int(max_tokens),
        }
        if self._supports_configured_temperature():
            payload["temperature"] = self.config.temperature
        endpoint = self.config.base_url.rstrip("/") + "/chat/completions"
        env = os.environ.copy()
        for key, value in restored_proxy_env.items():
            env[key] = value
        command = [
            "curl",
            "-sS",
            endpoint,
            "-H",
            f"Authorization: Bearer {self.config.api_key}",
            "-H",
            "Content-Type: application/json",
            "-d",
            json.dumps(payload, ensure_ascii=False),
        ]
        self.request_count += 1
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=max(30.0, float(self.config.timeout)),
            env=env,
            check=False,
        )
        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            stdout = (completed.stdout or "").strip()
            detail = stderr or stdout or f"curl exited with code {completed.returncode}"
            raise RuntimeError(f"curl fallback failed: {detail}")

        try:
            response = json.loads(completed.stdout)
        except Exception as exc:
            preview = (completed.stdout or "").strip()[:400]
            raise RuntimeError(f"curl fallback returned invalid JSON: {preview}") from exc

        error_payload = response.get("error")
        if error_payload:
            raise RuntimeError(f"curl fallback provider error: {error_payload}")

        choices = response.get("choices") or []
        if not choices:
            raise RuntimeError("curl fallback returned no choices")
        message = choices[0].get("message") or {}
        self.last_response_id = str(response.get("id") or "")
        usage = response.get("usage")
        self.last_usage = dict(usage) if isinstance(usage, dict) else None
        content = self._extract_content(message.get("content"))
        if content:
            return content
        reasoning = self._extract_content(message.get("reasoning_content"))
        refusal = self._extract_content(message.get("refusal"))
        raise RuntimeError(
            "curl fallback returned empty content "
            f"(reasoning_present={bool(reasoning)}, refusal_present={bool(refusal)})"
        )

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

    @staticmethod
    def _should_try_curl_fallback(exc: Exception) -> bool:
        text = str(exc).lower()
        fallback_markers = (
            "apiconnectionerror",
            "apitimeouterror",
            "connection error",
            "timed out",
            "timeout",
            "request timed out",
            "ssl",
            "unexpected eof",
            "name resolution",
            "temporary failure in name resolution",
            "proxyerror",
            "couldn't connect to server",
            "failed to resolve",
        )
        return any(marker in text for marker in fallback_markers)
