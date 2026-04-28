from __future__ import annotations

from contextvars import ContextVar

from .config import config
from .logger import log_info
from .providers.tavily import TavilyClient

_TAVILY_CLIENT: TavilyClient | None = None
_TAVILY_CLIENT_FINGERPRINT: tuple[str, tuple[str, ...], int] | None = None
_FETCH_STATUS: ContextVar[str] = ContextVar("web_search_fetch_status", default="unknown")


def _get_tavily_client() -> TavilyClient:
    global _TAVILY_CLIENT, _TAVILY_CLIENT_FINGERPRINT

    fingerprint = (
        config.tavily_api_url,
        tuple(config.tavily_api_keys),
        config.tavily_key_cooldown_seconds,
    )
    if _TAVILY_CLIENT is None or _TAVILY_CLIENT_FINGERPRINT != fingerprint:
        _TAVILY_CLIENT = TavilyClient(
            api_url=fingerprint[0],
            api_keys=list(fingerprint[1]),
            cooldown_seconds=fingerprint[2],
        )
        _TAVILY_CLIENT_FINGERPRINT = fingerprint

    return _TAVILY_CLIENT


async def _call_tavily_extract(url: str) -> str | None:
    client = _get_tavily_client()
    if not client.is_configured:
        return None
    try:
        return await client.extract(url)
    except Exception:
        return None


async def _call_tavily_search(query: str, max_results: int = 6) -> list[dict] | None:
    client = _get_tavily_client()
    if not client.is_configured:
        return None
    try:
        return await client.search(query, max_results)
    except Exception:
        return None


async def _call_firecrawl_search(query: str, limit: int = 14) -> list[dict] | None:
    import httpx

    api_key = config.firecrawl_api_key
    if not api_key:
        return None
    endpoint = f"{config.firecrawl_api_url.rstrip('/')}/search"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {"query": query, "limit": limit}
    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(endpoint, headers=headers, json=body)
            response.raise_for_status()
            data = response.json()
            results = data.get("data", {}).get("web", [])
            return [
                {"title": r.get("title", ""), "url": r.get("url", ""), "description": r.get("description", "")}
                for r in results
            ] if results else None
    except Exception:
        return None


async def _call_firecrawl_scrape(url: str, ctx=None) -> str | None:
    import httpx

    api_url = config.firecrawl_api_url
    api_key = config.firecrawl_api_key
    if not api_key:
        return None
    endpoint = f"{api_url.rstrip('/')}/scrape"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    max_retries = config.retry_max_attempts
    for attempt in range(max_retries):
        body = {
            "url": url,
            "formats": ["markdown"],
            "timeout": 60000,
            "waitFor": (attempt + 1) * 1500,
        }
        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                response = await client.post(endpoint, headers=headers, json=body)
                response.raise_for_status()
                data = response.json()
                markdown = data.get("data", {}).get("markdown", "")
                if markdown and markdown.strip():
                    return markdown
                await log_info(ctx, f"Firecrawl: markdown为空, 重试 {attempt + 1}/{max_retries}", config.debug_enabled)
        except Exception as e:
            await log_info(ctx, f"Firecrawl error: {e}", config.debug_enabled)
            return None
    return None


def _truncate_content(content: str, max_chars: int) -> dict[str, object]:
    normalized_content = content or ""
    content_length = len(normalized_content)
    returned_content = normalized_content[:max_chars]

    return {
        "content": returned_content,
        "truncated": content_length > max_chars,
        "content_length": content_length,
        "returned_length": len(returned_content),
        "max_chars": max_chars,
    }


def _build_tavily_map_payload(data: dict) -> dict:
    return {
        "base_url": data.get("base_url", ""),
        "results": data.get("results", []),
        "response_time": data.get("response_time", 0),
    }


async def _call_tavily_map_structured(
    url: str,
    instructions: str = None,
    max_depth: int = 1,
    max_breadth: int = 20,
    limit: int = 50,
    timeout: int = 150,
) -> dict:
    import httpx

    client = _get_tavily_client()
    if not client.is_configured:
        return {"error": "配置错误: TAVILY_API_KEY / TAVILY_API_KEYS 未配置，请设置环境变量或本地 .env"}
    try:
        data = await client.map(url, instructions or "", max_depth, max_breadth, limit, timeout)
        if not data:
            return {"error": "映射失败: Tavily 未返回可用结果"}
        return _build_tavily_map_payload(data)
    except httpx.TimeoutException:
        return {"error": f"映射超时: 请求超过{timeout}秒"}
    except httpx.HTTPStatusError as e:
        return {"error": f"HTTP错误: {e.response.status_code} - {e.response.text[:200]}"}
    except Exception as e:
        return {"error": f"映射错误: {str(e)}"}


async def _call_tavily_map(
    url: str,
    instructions: str = None,
    max_depth: int = 1,
    max_breadth: int = 20,
    limit: int = 50,
    timeout: int = 150,
) -> str:
    import json

    payload = await _call_tavily_map_structured(url, instructions, max_depth, max_breadth, limit, timeout)
    if "error" in payload:
        return payload["error"]

    return json.dumps(payload, ensure_ascii=False, indent=2)
