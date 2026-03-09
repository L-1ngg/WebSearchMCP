from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx


_NON_RETRYABLE_STATUS_CODES = {400, 404, 422}


class TavilyClient:
    def __init__(self, api_url: str, api_keys: list[str], cooldown_seconds: int = 60):
        self.api_url = api_url.rstrip("/")
        self.api_keys = [key.strip() for key in api_keys if key and key.strip()]
        self.cooldown_seconds = max(int(cooldown_seconds), 0)
        self._lock = asyncio.Lock()
        self._next_index = 0
        self._cooldowns: dict[str, float] = {}

    @property
    def is_configured(self) -> bool:
        return bool(self.api_keys)

    async def extract(self, url: str) -> str | None:
        if not self.is_configured:
            return None

        data = await self._post_json(
            "/extract",
            {"urls": [url], "format": "markdown"},
            timeout=60.0,
        )
        if not data:
            return None

        results = data.get("results", [])
        if not results:
            return None
        content = (results[0].get("raw_content") or "").strip()
        return content or None

    async def search(self, query: str, max_results: int = 6) -> list[dict] | None:
        if not self.is_configured:
            return None

        data = await self._post_json(
            "/search",
            {
                "query": query,
                "max_results": max_results,
                "search_depth": "advanced",
                "include_raw_content": False,
                "include_answer": False,
            },
            timeout=90.0,
        )
        if not data:
            return None

        results = data.get("results", [])
        if not results:
            return None
        return [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
                "score": item.get("score", 0),
            }
            for item in results
        ]

    async def map(
        self,
        url: str,
        instructions: str = "",
        max_depth: int = 1,
        max_breadth: int = 20,
        limit: int = 50,
        timeout: int = 150,
    ) -> dict[str, Any] | None:
        if not self.is_configured:
            return None

        body: dict[str, Any] = {
            "url": url,
            "max_depth": max_depth,
            "max_breadth": max_breadth,
            "limit": limit,
            "timeout": timeout,
        }
        if instructions:
            body["instructions"] = instructions

        data = await self._post_json("/map", body, timeout=float(timeout + 10))
        if not data:
            return None
        return {
            "base_url": data.get("base_url", ""),
            "results": data.get("results", []),
            "response_time": data.get("response_time", 0),
        }

    async def _post_json(self, path: str, body: dict[str, Any], timeout: float) -> dict[str, Any] | None:
        candidate_indices = await self._candidate_indices()
        if not candidate_indices:
            return None

        errors: list[str] = []
        for index in candidate_indices:
            key = self.api_keys[index]
            endpoint = f"{self.api_url}{path}"
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(endpoint, headers=headers, json=body)
                    response.raise_for_status()
                    await self._mark_success(key)
                    return response.json()
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code
                errors.append(f"{status_code}@{self._mask_key(key)}")
                if status_code in _NON_RETRYABLE_STATUS_CODES:
                    raise
                await self._mark_failure(key, aggressive=status_code in (401, 403, 429) or status_code >= 500)
            except (httpx.TimeoutException, httpx.NetworkError, httpx.ConnectError, httpx.RemoteProtocolError) as exc:
                errors.append(f"{type(exc).__name__}@{self._mask_key(key)}")
                await self._mark_failure(key, aggressive=True)
            except Exception as exc:
                errors.append(f"{type(exc).__name__}@{self._mask_key(key)}")
                await self._mark_failure(key, aggressive=True)

        if errors:
            raise RuntimeError(f"All Tavily API keys failed: {', '.join(errors)}")
        return None

    async def _candidate_indices(self) -> list[int]:
        async with self._lock:
            total = len(self.api_keys)
            if total == 0:
                return []

            start = self._next_index % total
            ordered = list(range(start, total)) + list(range(0, start))
            self._next_index = (start + 1) % total

            now = time.monotonic()
            available = [index for index in ordered if self._cooldowns.get(self.api_keys[index], 0.0) <= now]
            return available or ordered

    async def _mark_success(self, key: str) -> None:
        async with self._lock:
            self._cooldowns.pop(key, None)

    async def _mark_failure(self, key: str, aggressive: bool) -> None:
        if not aggressive or self.cooldown_seconds <= 0:
            return
        async with self._lock:
            self._cooldowns[key] = time.monotonic() + self.cooldown_seconds

    @staticmethod
    def _mask_key(key: str) -> str:
        if len(key) <= 8:
            return "***"
        return f"{key[:4]}...{key[-4:]}"
