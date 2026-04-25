import sys
from contextvars import ContextVar
from pathlib import Path

# 支持直接运行：添加 src 目录到 Python 路径
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from fastmcp import FastMCP, Context
from typing import Annotated, Literal, Optional
from pydantic import Field

# 尝试使用绝对导入（支持 mcp run）
try:
    from web_search import diagnostics
    from web_search.providers.grok import GrokSearchProvider
    from web_search.providers.tavily import TavilyClient
    from web_search.logger import log_info
    from web_search.config import config
    from web_search.sources import SourcesCache, merge_sources, new_session_id, split_answer_and_sources
    from web_search.planning import engine as planning_engine, _split_csv
    from web_search.planning_adapter import (
        _build_planning_context_data,
        _missing_required_phases,
        _normalize_query_text,
        _planning_matches_query,
        _query_fingerprint,
        _resolve_planning_context,
        _sanitize_context_id,
        _sanitize_context_token,
        _sanitize_search_term,
    )
except ImportError:
    from . import diagnostics
    from .providers.grok import GrokSearchProvider
    from .providers.tavily import TavilyClient
    from .logger import log_info
    from .config import config
    from .sources import SourcesCache, merge_sources, new_session_id, split_answer_and_sources
    from .planning import engine as planning_engine, _split_csv
    from .planning_adapter import (
        _build_planning_context_data,
        _missing_required_phases,
        _normalize_query_text,
        _planning_matches_query,
        _query_fingerprint,
        _resolve_planning_context,
        _sanitize_context_id,
        _sanitize_context_token,
        _sanitize_search_term,
    )

import asyncio

mcp = FastMCP("web-search")

_SOURCES_CACHE = SourcesCache(max_size=256)
_AVAILABLE_MODELS_CACHE: dict[tuple[str, str], list[str]] = {}
_AVAILABLE_MODELS_LOCK = asyncio.Lock()
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


async def _fetch_available_models(api_url: str, api_key: str) -> list[str]:
    return await diagnostics.fetch_available_models(api_url, api_key)


async def _get_available_models_cached(api_url: str, api_key: str) -> list[str]:
    key = (api_url, api_key)
    async with _AVAILABLE_MODELS_LOCK:
        if key in _AVAILABLE_MODELS_CACHE:
            return _AVAILABLE_MODELS_CACHE[key]

    try:
        models = await _fetch_available_models(api_url, api_key)
    except Exception:
        models = []

    async with _AVAILABLE_MODELS_LOCK:
        _AVAILABLE_MODELS_CACHE[key] = models
    return models


def _extra_results_to_sources(
    tavily_results: list[dict] | None,
    firecrawl_results: list[dict] | None,
) -> list[dict]:
    sources: list[dict] = []
    seen: set[str] = set()

    if firecrawl_results:
        for r in firecrawl_results:
            url = (r.get("url") or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            item: dict = {"url": url, "provider": "firecrawl"}
            title = (r.get("title") or "").strip()
            if title:
                item["title"] = title
            desc = (r.get("description") or "").strip()
            if desc:
                item["description"] = desc
            sources.append(item)

    if tavily_results:
        for r in tavily_results:
            url = (r.get("url") or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            item: dict = {"url": url, "provider": "tavily"}
            title = (r.get("title") or "").strip()
            if title:
                item["title"] = title
            content = (r.get("content") or "").strip()
            if content:
                item["description"] = content
            sources.append(item)

    return sources


def _build_sources_preview(sources: list[dict], limit: int = 3) -> list[dict]:
    preview: list[dict] = []
    for item in sources[:limit]:
        url = (item.get("url") or "").strip()
        if not url:
            continue

        out: dict = {"url": url}
        for key in ("title", "provider"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                out[key] = value.strip()

        description = item.get("description")
        if isinstance(description, str) and description.strip():
            out["description"] = description.strip()[:240]

        preview.append(out)

    return preview


def _build_web_search_response(
    session_id: str,
    content: str,
    sources: list[dict],
    *,
    status: str = "ok",
    error_code: str = "",
    error_message: str = "",
    retry_same_query: bool = False,
    answer_ready: bool | None = None,
    used_custom_search_prompt: bool = False,
    planning_applied: bool = False,
    planning_status: str = "not_provided",
    warnings: list[str] | None = None,
) -> dict:
    normalized_content = (content or "").strip()
    response = {
        "session_id": session_id,
        "content": normalized_content,
        "sources_count": len(sources),
        "status": status,
        "answer_ready": answer_ready if answer_ready is not None else status == "ok" and bool(normalized_content),
        "used_custom_search_prompt": used_custom_search_prompt,
        "planning_applied": planning_applied,
        "planning_status": planning_status,
    }

    preview = _build_sources_preview(sources)
    if preview:
        response["sources_preview"] = preview

    if warnings:
        response["warnings"] = warnings

    if status != "ok":
        response["error"] = {
            "code": error_code or "web_search_error",
            "message": error_message or normalized_content or "web_search failed",
            "retry_same_query": retry_same_query,
        }

    return response


def _build_get_sources_response(session_id: str, page: dict, error: str = "") -> dict:
    response = {
        "session_id": session_id,
        "sources": page["sources"],
        "sources_count": page["sources_count"],
        "returned_count": len(page["sources"]),
        "next_cursor": page["next_cursor"],
        "has_more": page["has_more"],
    }
    if error:
        response["error"] = error
    return response


def _build_sparse_search_fallback(sources: list[dict]) -> str:
    preview = _build_sources_preview(sources)
    if preview:
        lines = [
            "当前查询没有拿到足够完整的正文输出，但已经检索到一些可供核验的相关来源。",
            "这通常说明主题较冷门、证据分散，或上游模型在证据不足时选择了保守输出。",
            "",
            "可先核验这些来源：",
        ]
        for item in preview:
            label = item.get("title") or item["url"]
            lines.append(f"- [{label}]({item['url']})")
        lines.extend(
            [
                "",
                "如果需要更完整的结论，建议缩小查询范围、补充实体名或时间范围，再继续搜索。",
            ]
        )
        return "\n".join(lines)

    return (
        "当前查询未返回足够完整的正文。"
        "这通常意味着主题较冷门、证据分散，或检索范围仍然过宽。"
        "建议缩小查询范围、补充实体名或时间范围后再试。"
    )


WEB_SEARCH_DESCRIPTION = """
By default, you can call `web_search` directly. The server applies its own bounded search strategy and only uses planning context when you explicitly provide it.

Performs a bounded multi-round web search workflow and returns a synthesized answer. It may use breadth-first exploration followed by depth-first follow-up when the query warrants it, but it must still converge instead of looping indefinitely.

`search_prompt` is optional. Use it when the calling agent wants to provide its own search strategy or answer-style instructions. Internal safety guardrails and fixed-format helper prompts still remain enforced by the server.
If omitted, the server falls back to its default bounded search strategy.

Planning is optional:
- `planning_session_id`: optional planning session from `plan_intent`
- `planning_mode`: `auto` | `require` | `ignore`
- In `auto`, invalid planning is ignored with warnings and the default search path continues.
- In `require`, planning validation failures become terminal errors.
- In `ignore`, planning is skipped entirely.

Structured steering parameters are also available:
- `source_preference`: `auto` | `official` | `community` | `news` | `academic`
- `answer_style`: `auto` | `concise` | `detailed` | `bullet_summary`
- `search_depth`: `auto` | `direct` | `balanced` | `deep`

Example:
{
  "query": "latest FastAPI release notes",
  "planning_session_id": "plan_123",
  "planning_mode": "auto",
  "source_preference": "official",
  "answer_style": "bullet_summary",
  "search_prompt": "Prefer official docs; answer in 3 bullets."
}

Returns:
- session_id: string
- content: string
- sources_count: int
- status: "ok" | "error"
- answer_ready: bool
- used_custom_search_prompt: bool
- planning_applied: bool
- planning_status: string
- sources_preview: lightweight preview of up to 3 cached sources
- warnings: optional non-fatal warnings such as ignored invalid planning
- error: object, only present when status is "error"

Use get_sources only when the full cached source list is needed for citation or verification.
If status is "error", treat it as a terminal result for this exact query. Do not repeat the same query verbatim; answer with the limitation or refine the query first.
"""


@mcp.tool(
    name="web_search",
    output_schema=None,
    description=WEB_SEARCH_DESCRIPTION,
    meta={"version": "2.0.0", "author": "guda.studio"},
)
async def web_search(
    query: Annotated[str, "Clear, self-contained natural-language search query."],
    planning_session_id: Annotated[str, "Optional session ID returned by plan_intent. When present, web_search may apply it as reference context."] = "",
    planning_mode: Annotated[Literal["auto", "require", "ignore"], "How to handle planning_session_id. `auto` applies valid planning and ignores invalid planning with warnings; `require` enforces valid planning; `ignore` skips planning entirely."] = "auto",
    platform: Annotated[str, "Target platform to focus on (e.g., 'Twitter', 'GitHub', 'Reddit'). Leave empty for general web search."] = "",
    model: Annotated[str, "Optional model ID for this request only. This value is used ONLY when user explicitly provided."] = "",
    search_prompt: Annotated[str, "Optional caller-authored search strategy prompt. Use this to steer search depth, source preferences, or answer style while keeping server-side guardrails intact."] = "",
    source_preference: Annotated[Literal["auto", "official", "community", "news", "academic"], "Structured source preference. Use `official` for first-party docs, `community` for practitioner discussion, `news` for current reporting, or `academic` for papers and benchmarks."] = "auto",
    answer_style: Annotated[Literal["auto", "concise", "detailed", "bullet_summary"], "Structured answer style. Use `concise` for short direct answers, `detailed` for fuller explanations, or `bullet_summary` for bullet-led output."] = "auto",
    search_depth: Annotated[Literal["auto", "direct", "balanced", "deep"], "Structured search depth. Use `direct` for minimal verification, `balanced` for limited follow-up, or `deep` for broader exploration before targeted drill-down."] = "auto",
    extra_sources: Annotated[int, "Number of additional reference results from Tavily/Firecrawl. Set 0 to disable. Default 0."] = 0,
    ctx: Context = None,
) -> dict:
    session_id = new_session_id()
    has_custom_search_prompt = bool((search_prompt or "").strip())
    planning_context, planning_status, planning_warnings, planning_error_code, planning_error_message = _resolve_planning_context(
        planning_session_id,
        query,
        planning_mode,
    )
    if planning_error_code:
        await _SOURCES_CACHE.set(session_id, [])
        return _build_web_search_response(
            session_id,
            planning_error_message,
            [],
            status="error",
            error_code=planning_error_code,
            error_message=planning_error_message,
            used_custom_search_prompt=has_custom_search_prompt,
            planning_applied=False,
            planning_status=planning_status,
            warnings=planning_warnings,
        )

    try:
        api_url = config.grok_api_url
        api_key = config.grok_api_key
    except ValueError as e:
        await _SOURCES_CACHE.set(session_id, [])
        message = f"配置错误: {str(e)}"
        return _build_web_search_response(
            session_id,
            message,
            [],
            status="error",
            error_code="config_error",
            error_message=message,
            used_custom_search_prompt=has_custom_search_prompt,
            planning_applied=planning_context is not None,
            planning_status=planning_status,
            warnings=planning_warnings,
        )

    effective_model = config.grok_model
    if model:
        available = await _get_available_models_cached(api_url, api_key)
        if available and model not in available:
            await _SOURCES_CACHE.set(session_id, [])
            message = f"无效模型: {model}"
            return _build_web_search_response(
                session_id,
                message,
                [],
                status="error",
                error_code="invalid_model",
                error_message=message,
                used_custom_search_prompt=has_custom_search_prompt,
                planning_applied=planning_context is not None,
                planning_status=planning_status,
                warnings=planning_warnings,
            )
        effective_model = model

    grok_provider = GrokSearchProvider(api_url, api_key, effective_model)

    # 计算额外信源配额
    has_tavily = config.tavily_enabled and bool(config.tavily_api_keys)
    has_firecrawl = bool(config.firecrawl_api_key)
    firecrawl_count = 0
    tavily_count = 0
    if extra_sources > 0:
        if has_firecrawl and has_tavily:
            firecrawl_count = round(extra_sources * 1)
            tavily_count = extra_sources - firecrawl_count
        elif has_firecrawl:
            firecrawl_count = extra_sources
        elif has_tavily:
            tavily_count = extra_sources

    # 并行执行搜索任务
    async def _safe_tavily() -> list[dict] | None:
        try:
            if tavily_count:
                return await _call_tavily_search(query, tavily_count)
        except Exception:
            return None

    async def _safe_firecrawl() -> list[dict] | None:
        try:
            if firecrawl_count:
                return await _call_firecrawl_search(query, firecrawl_count)
        except Exception:
            return None

    provider_kwargs = {
        "ctx": ctx,
        "planning_context": planning_context,
    }
    if search_prompt.strip():
        provider_kwargs["search_prompt"] = search_prompt
    if source_preference != "auto":
        provider_kwargs["source_preference"] = source_preference
    if answer_style != "auto":
        provider_kwargs["answer_style"] = answer_style
    if search_depth != "auto":
        provider_kwargs["search_depth"] = search_depth

    coros: list = [
        grok_provider.search(
            query,
            platform,
            **provider_kwargs,
        )
    ]
    if tavily_count > 0:
        coros.append(_safe_tavily())
    if firecrawl_count > 0:
        coros.append(_safe_firecrawl())

    gathered = await asyncio.gather(*coros, return_exceptions=True)

    grok_outcome = gathered[0]
    grok_result: str = ""
    tavily_results: list[dict] | None = None
    firecrawl_results: list[dict] | None = None
    idx = 1
    if tavily_count > 0:
        tavily_outcome = gathered[idx]
        tavily_results = None if isinstance(tavily_outcome, Exception) else tavily_outcome
        idx += 1
    if firecrawl_count > 0:
        firecrawl_outcome = gathered[idx]
        firecrawl_results = None if isinstance(firecrawl_outcome, Exception) else firecrawl_outcome

    grok_sources: list[dict] = []
    if not isinstance(grok_outcome, Exception):
        grok_result = grok_outcome or ""
        answer, grok_sources = split_answer_and_sources(grok_result)
    else:
        answer = ""

    extra = _extra_results_to_sources(tavily_results, firecrawl_results)
    all_sources = merge_sources(grok_sources, extra)
    await _SOURCES_CACHE.set(session_id, all_sources)

    if isinstance(grok_outcome, Exception):
        message = f"上游搜索失败: {type(grok_outcome).__name__}: {grok_outcome}"
        await log_info(ctx, message, config.debug_enabled)
        return _build_web_search_response(
            session_id,
            message,
            all_sources,
            status="error",
            error_code="upstream_search_failed",
            error_message=message,
            used_custom_search_prompt=has_custom_search_prompt,
            planning_applied=planning_context is not None,
            planning_status=planning_status,
            warnings=planning_warnings,
        )

    if not answer.strip():
        message = "搜索未返回可用答案正文，已降级返回稀疏结果"
        await log_info(ctx, message, config.debug_enabled)
        return _build_web_search_response(
            session_id,
            _build_sparse_search_fallback(all_sources),
            all_sources,
            answer_ready=False,
            used_custom_search_prompt=has_custom_search_prompt,
            planning_applied=planning_context is not None,
            planning_status=planning_status,
            warnings=planning_warnings,
        )

    return _build_web_search_response(
        session_id,
        answer,
        all_sources,
        used_custom_search_prompt=has_custom_search_prompt,
        planning_applied=planning_context is not None,
        planning_status=planning_status,
        warnings=planning_warnings,
    )


@mcp.tool(
    name="search",
    output_schema=None,
    description="""
    Stable core-tool alias for `web_search`.
    This is a thin non-breaking wrapper that forwards all arguments to `web_search`.
    """,
    meta={"version": "1.4.0", "author": "guda.studio"},
)
async def search(
    query: Annotated[str, "Clear, self-contained natural-language search query."],
    planning_session_id: Annotated[str, "Optional session ID returned by plan_intent. When present, web_search may apply it as reference context."] = "",
    planning_mode: Annotated[Literal["auto", "require", "ignore"], "How to handle planning_session_id. `auto` applies valid planning and ignores invalid planning with warnings; `require` enforces valid planning; `ignore` skips planning entirely."] = "auto",
    platform: Annotated[str, "Target platform to focus on (e.g., 'Twitter', 'GitHub', 'Reddit'). Leave empty for general web search."] = "",
    model: Annotated[str, "Optional model ID for this request only. This value is used ONLY when user explicitly provided."] = "",
    search_prompt: Annotated[str, "Optional caller-authored search strategy prompt. Use this to steer search depth, source preferences, or answer style while keeping server-side guardrails intact."] = "",
    source_preference: Annotated[Literal["auto", "official", "community", "news", "academic"], "Structured source preference. Use `official` for first-party docs, `community` for practitioner discussion, `news` for current reporting, or `academic` for papers and benchmarks."] = "auto",
    answer_style: Annotated[Literal["auto", "concise", "detailed", "bullet_summary"], "Structured answer style. Use `concise` for short direct answers, `detailed` for fuller explanations, or `bullet_summary` for bullet-led output."] = "auto",
    search_depth: Annotated[Literal["auto", "direct", "balanced", "deep"], "Structured search depth. Use `direct` for minimal verification, `balanced` for limited follow-up, or `deep` for broader exploration before targeted drill-down."] = "auto",
    extra_sources: Annotated[int, "Number of additional reference results from Tavily/Firecrawl. Set 0 to disable. Default 0."] = 0,
    ctx: Context = None,
) -> dict:
    return await web_search(
        query=query,
        planning_session_id=planning_session_id,
        planning_mode=planning_mode,
        platform=platform,
        model=model,
        search_prompt=search_prompt,
        source_preference=source_preference,
        answer_style=answer_style,
        search_depth=search_depth,
        extra_sources=extra_sources,
        ctx=ctx,
    )


@mcp.tool(
    name="get_sources",
    description="""
    Retrieve cached sources for a previous web_search call.
    Provide the session_id returned by web_search. If limit is omitted or 0, returns the full cached list.
    If limit is greater than 0, returns a paginated slice and next_cursor metadata for follow-up calls.
    """,
    meta={"version": "1.0.0", "author": "guda.studio"},
)
async def get_sources(
    session_id: Annotated[str, "Session ID from previous web_search call."],
    limit: Annotated[int, "Optional page size. Use 0 or omit it to keep the legacy full-list behavior."] = 0,
    cursor: Annotated[str, "Optional pagination cursor returned by a previous get_sources call."] = "",
) -> dict:
    page = await _SOURCES_CACHE.page(session_id, limit=limit, cursor=cursor)
    if page is None:
        return _build_get_sources_response(
            session_id,
            {
                "sources": [],
                "sources_count": 0,
                "next_cursor": "",
                "has_more": False,
            },
            error="session_id_not_found_or_expired",
        )
    return _build_get_sources_response(session_id, page)


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


@mcp.tool(
    name="web_fetch",
    output_schema=None,
    description="""
    Fetches and extracts complete content from a URL, returning it as a structured Markdown document.

    **Key Features:**
        - **Full Content Extraction:** Retrieves and parses all meaningful content (text, images, links, tables, code blocks).
        - **Markdown Conversion:** Converts HTML structure to well-formatted Markdown with preserved hierarchy.
        - **Content Fidelity:** Maintains 100% content fidelity without summarization or modification.

    **Edge Cases & Best Practices:**
        - Ensure URL is complete and accessible (not behind authentication or paywalls).
        - May not capture dynamically loaded content requiring JavaScript execution.
        - Large pages may take longer to process; consider timeout implications.
    """,
    meta={"version": "1.3.0", "author": "guda.studio"},
)
async def web_fetch(
    url: Annotated[str, "Valid HTTP/HTTPS web address pointing to the target page. Must be complete and accessible."],
    ctx: Context = None
) -> str:
    await log_info(ctx, f"Begin Fetch: {url}", config.debug_enabled)

    result = await _call_tavily_extract(url)
    if result:
        _FETCH_STATUS.set("ok")
        await log_info(ctx, "Fetch Finished (Tavily)!", config.debug_enabled)
        return result

    await log_info(ctx, "Tavily unavailable or failed, trying Firecrawl...", config.debug_enabled)
    result = await _call_firecrawl_scrape(url, ctx)
    if result:
        _FETCH_STATUS.set("ok")
        await log_info(ctx, "Fetch Finished (Firecrawl)!", config.debug_enabled)
        return result

    await log_info(ctx, "Fetch Failed!", config.debug_enabled)
    if not config.tavily_api_keys and not config.firecrawl_api_key:
        _FETCH_STATUS.set("error")
        return "配置错误: TAVILY_API_KEY / TAVILY_API_KEYS 和 FIRECRAWL_API_KEY 均未配置"
    _FETCH_STATUS.set("error")
    return "提取失败: 所有提取服务均未能获取内容"


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


@mcp.tool(
    name="fetch",
    output_schema=None,
    description="""
    Compatibility alias for `web_fetch` that returns a structured object with bounded output.

    Returns:
    - status: `ok` when content is returned, otherwise `error`
    - url: the requested URL
    - content: extracted content truncated to `max_chars`
    - truncated: whether the output was shortened
    - content_length: original content length
    - returned_length: length after truncation
    - max_chars: requested response bound
    """,
    meta={"version": "1.4.0", "author": "guda.studio"},
)
async def fetch(
    url: Annotated[str, "Valid HTTP/HTTPS web address pointing to the target page. Must be complete and accessible."],
    max_chars: Annotated[int, Field(description="Maximum number of characters to return from the fetched content.", ge=1)] = 12000,
    ctx: Context = None,
) -> dict:
    token = _FETCH_STATUS.set("unknown")
    try:
        content = await web_fetch(url, ctx)
        status = _FETCH_STATUS.get()
    finally:
        _FETCH_STATUS.reset(token)

    payload = _truncate_content(content, max_chars)

    if status == "unknown":
        status = "ok" if content else "error"

    return {
        "status": status,
        "url": url,
        **payload,
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


@mcp.tool(
    name="web_map",
    description="""
    Maps a website's structure by traversing it like a graph, discovering URLs and generating a comprehensive site map.

    **Key Features:**
        - **Graph Traversal:** Explores website structure starting from root URL.
        - **Depth & Breadth Control:** Configure traversal limits to balance coverage and performance.
        - **Instruction Filtering:** Use natural language to focus crawler on specific content types.

    **Edge Cases & Best Practices:**
        - Start with low max_depth (1-2) for initial exploration, increase if needed.
        - Use instructions to filter for specific content (e.g., "only documentation pages").
        - Large sites may hit timeout limits; adjust timeout and limit parameters accordingly.
    """,
    meta={"version": "1.3.0", "author": "guda.studio"},
)
async def web_map(
    url: Annotated[str, "Root URL to begin the mapping (e.g., 'https://docs.example.com')."],
    instructions: Annotated[str, "Natural language instructions for the crawler to filter or focus on specific content."] = "",
    max_depth: Annotated[int, Field(description="Maximum depth of mapping from the base URL.", ge=1, le=5)] = 1,
    max_breadth: Annotated[int, Field(description="Maximum number of links to follow per page.", ge=1, le=500)] = 20,
    limit: Annotated[int, Field(description="Total number of links to process before stopping.", ge=1, le=500)] = 50,
    timeout: Annotated[int, Field(description="Maximum time in seconds for the operation.", ge=10, le=150)] = 150
) -> str:
    result = await _call_tavily_map(url, instructions, max_depth, max_breadth, limit, timeout)
    return result


@mcp.tool(
    name="map",
    output_schema=None,
    description="""
    Compatibility alias for `web_map` that returns a structured payload instead of a JSON string.
    """,
    meta={"version": "1.4.0", "author": "guda.studio"},
)
async def map(
    url: Annotated[str, "Root URL to begin the mapping (e.g., 'https://docs.example.com')."],
    instructions: Annotated[str, "Natural language instructions for the crawler to filter or focus on specific content."] = "",
    max_depth: Annotated[int, Field(description="Maximum depth of mapping from the base URL.", ge=1, le=5)] = 1,
    max_breadth: Annotated[int, Field(description="Maximum number of links to follow per page.", ge=1, le=500)] = 20,
    limit: Annotated[int, Field(description="Total number of links to process before stopping.", ge=1, le=500)] = 50,
    timeout: Annotated[int, Field(description="Maximum time in seconds for the operation.", ge=10, le=150)] = 150,
) -> dict:
    return await _call_tavily_map_structured(url, instructions, max_depth, max_breadth, limit, timeout)


@mcp.tool(
    name="doctor",
    output_schema=None,
    description="""
    Runs a compact configuration and connectivity diagnostic for the web-search MCP server.

    Returns:
    - configuration: boolean flags describing whether key settings are present
    - connection_test: Grok API connectivity status without exposing the model list
    - recommended_next_step: one actionable follow-up based on the diagnostic result
    """,
    meta={"version": "1.0.0", "author": "guda.studio"},
)
async def doctor() -> dict:
    return await diagnostics.get_doctor_info()


@mcp.tool(
    name="get_config_info",
    output_schema=None,
    description="""
    Returns current Grok Search MCP server configuration in a structured JSON object.

    **Key Features:**
        - **Structured Diagnostics:** Always returns structured JSON instead of throwing, including when config snapshot gathering fails.
        - **Optional Connection Test:** Set `include_connection_test=true` to probe the `/models` endpoint and validate API access.
        - **Model Discovery:** When the connection test runs successfully, it lists available models from the API.

    **Edge Cases & Best Practices:**
        - Use this tool first when debugging connection or configuration issues.
        - API keys are automatically masked for security in the response.
        - The connection test is skipped by default so callers can inspect configuration without incurring network latency or failures.
        - When enabled, the connection test uses a 10-second timeout and reports failures in-band.
    """,
    meta={"version": "1.4.0", "author": "guda.studio"},
)
async def get_config_info(
    include_connection_test: Annotated[
        bool,
        "Set to true to probe the /models endpoint. Defaults to false so config inspection remains local and reliable.",
    ] = False,
    reason: Annotated[
        str,
        "Optional caller note accepted for compatibility with agent tool-calling patterns. This value is ignored by the server.",
    ] = "",
) -> dict:
    import time
    import httpx

    skipped_connection_test = {
        "status": "skipped",
        "message": "Connection test not run. Pass include_connection_test=true to probe the API.",
        "response_time_ms": None,
        "available_models": [],
    }
    result: dict = {
        "status": "ok",
        "config": {},
        "config_status": "Configuration snapshot not collected.",
        "connection_test": dict(skipped_connection_test),
    }

    try:
        config_snapshot = config.get_config_info()
        result["config"] = config_snapshot
        result.update(config_snapshot)
    except Exception as e:
        result["status"] = "error"
        result["config_status"] = f"❌ Failed to gather configuration snapshot: {str(e)}"
        result["error"] = {
            "code": "config_snapshot_failed",
            "message": f"Failed to gather configuration snapshot: {str(e)}",
        }

    if not include_connection_test:
        return result

    try:
        api_url = config.grok_api_url
        api_key = config.grok_api_key

        start_time = time.perf_counter()
        available_models = await _fetch_available_models(api_url, api_key)
        response_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

        result["connection_test"] = {
            "status": "ok",
            "message": "Connection test succeeded.",
            "response_time_ms": response_time_ms,
            "available_models": available_models,
        }
    except httpx.TimeoutException:
        result["connection_test"] = {
            "status": "error",
            "message": "Connection test timed out after 10 seconds.",
            "response_time_ms": None,
            "available_models": [],
        }
    except httpx.RequestError as e:
        result["connection_test"] = {
            "status": "error",
            "message": f"Connection test failed: {str(e)}",
            "response_time_ms": None,
            "available_models": [],
        }
    except ValueError as e:
        result["connection_test"] = {
            "status": "error",
            "message": f"Connection test could not start: {str(e)}",
            "response_time_ms": None,
            "available_models": [],
        }
    except Exception as e:
        result["connection_test"] = {
            "status": "error",
            "message": f"Connection test failed unexpectedly: {str(e)}",
            "response_time_ms": None,
            "available_models": [],
        }

    return result


@mcp.tool(
    name="switch_model",
    output_schema=None,
    description="""
    Switches the default Grok model used for search and fetch operations, persisting the setting.

    **Key Features:**
        - **Model Selection:** Change the AI model for web search and content fetching.
        - **Persistent Storage:** Model preference saved to ~/.config/web-search/config.json.
        - **Immediate Effect:** New model used for all subsequent operations.

    **Edge Cases & Best Practices:**
        - Use get_config_info to verify available models before switching.
        - Invalid model IDs may cause API errors in subsequent requests.
        - Model changes persist across sessions until explicitly changed again.
    """,
    meta={"version": "1.3.0", "author": "guda.studio"},
)
async def switch_model(
    model: Annotated[str, "Model ID to switch to (e.g., 'grok-4-fast', 'grok-2-latest', 'grok-vision-beta')."]
) -> str:
    import json

    try:
        previous_model = config.grok_model
        config.set_model(model)
        current_model = config.grok_model

        result = {
            "status": "✅ 成功",
            "previous_model": previous_model,
            "current_model": current_model,
            "message": f"模型已从 {previous_model} 切换到 {current_model}",
            "config_file": str(config.config_file)
        }

        return json.dumps(result, ensure_ascii=False, indent=2)

    except ValueError as e:
        result = {
            "status": "❌ 失败",
            "message": f"切换模型失败: {str(e)}"
        }
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        result = {
            "status": "❌ 失败",
            "message": f"未知错误: {str(e)}"
        }
        return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool(
    name="toggle_builtin_tools",
    output_schema=None,
    description="""
    Toggle Claude Code's built-in WebSearch and WebFetch tools on/off.

    **Key Features:**
        - **Tool Control:** Enable or disable Claude Code's native web tools.
        - **Project Scope:** Changes apply to current project's .claude/settings.json.
        - **Status Check:** Query current state without making changes.

    **Edge Cases & Best Practices:**
        - Use "on" to block built-in tools when preferring this MCP server's implementation.
        - Use "off" to restore Claude Code's native tools.
        - Use "status" to check current configuration without modification.
    """,
    meta={"version": "1.3.0", "author": "guda.studio"},
)
async def toggle_builtin_tools(
    action: Annotated[str, "Action to perform: 'on' (block built-in), 'off' (allow built-in), or 'status' (check current state)."] = "status"
) -> str:
    import json

    # Locate project root
    root = Path.cwd()
    while root != root.parent and not (root / ".git").exists():
        root = root.parent

    settings_path = root / ".claude" / "settings.json"
    tools = ["WebFetch", "WebSearch"]

    # Load or initialize
    if settings_path.exists():
        with open(settings_path, 'r', encoding='utf-8') as f:
            settings = json.load(f)
    else:
        settings = {"permissions": {"deny": []}}

    deny = settings.setdefault("permissions", {}).setdefault("deny", [])
    blocked = all(t in deny for t in tools)

    # Execute action
    if action in ["on", "enable"]:
        for t in tools:
            if t not in deny:
                deny.append(t)
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        with open(settings_path, 'w', encoding='utf-8') as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
        msg = "官方工具已禁用"
        blocked = True
    elif action in ["off", "disable"]:
        deny[:] = [t for t in deny if t not in tools]
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        with open(settings_path, 'w', encoding='utf-8') as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
        msg = "官方工具已启用"
        blocked = False
    else:
        msg = f"官方工具当前{'已禁用' if blocked else '已启用'}"

    return json.dumps({
        "blocked": blocked,
        "deny_list": deny,
        "file": str(settings_path),
        "message": msg
    }, ensure_ascii=False, indent=2)


@mcp.tool(
    name="plan_intent",
    output_schema=None,
    description="""
    Phase 1 of search planning: Analyze user intent. Call this FIRST to create a session.
    Returns session_id for subsequent phases. Required flow:
    plan_intent → plan_complexity → plan_sub_query(×N) → plan_search_term(×N) → plan_tool_mapping(×N) → plan_execution

    Required phases depend on complexity: Level 1 = phases 1-3; Level 2 = phases 1-5; Level 3 = all 6.
    """,
)
async def plan_intent(
    thought: Annotated[str, "Reasoning for this phase"],
    original_query: Annotated[str, "Original user query before distillation; required to bind this planning session to web_search."],
    core_question: Annotated[str, "Distilled core question in one sentence"],
    query_type: Annotated[str, "factual | comparative | exploratory | analytical"],
    time_sensitivity: Annotated[str, "realtime | recent | historical | irrelevant"],
    session_id: Annotated[str, "Empty for new session, or existing ID to revise"] = "",
    confidence: Annotated[float, "Confidence 0.0-1.0"] = 1.0,
    domain: Annotated[str, "Specific domain if identifiable"] = "",
    premise_valid: Annotated[Optional[bool], "False if the question contains a flawed assumption"] = None,
    ambiguities: Annotated[str, "Comma-separated unresolved ambiguities"] = "",
    unverified_terms: Annotated[str, "Comma-separated external terms to verify"] = "",
    is_revision: Annotated[bool, "True to overwrite existing intent"] = False,
) -> str:
    import json
    if not original_query.strip():
        return json.dumps({"error": "original_query is required and cannot be blank."}, ensure_ascii=False, indent=2)
    data = {
        "core_question": core_question,
        "query_type": query_type,
        "time_sensitivity": time_sensitivity,
        "query_fingerprint": _query_fingerprint(original_query),
    }
    if domain:
        data["domain"] = domain
    if premise_valid is not None:
        data["premise_valid"] = premise_valid
    if ambiguities:
        data["ambiguities"] = _split_csv(ambiguities)
    if unverified_terms:
        data["unverified_terms"] = _split_csv(unverified_terms)
    return json.dumps(planning_engine.process_phase(
        phase="intent_analysis", thought=thought, session_id=session_id,
        is_revision=is_revision, confidence=confidence, phase_data=data,
    ), ensure_ascii=False, indent=2)


@mcp.tool(
    name="plan_complexity",
    output_schema=None,
    description="Phase 2: Assess search complexity (1-3). Controls required phases: Level 1 = phases 1-3; Level 2 = phases 1-5; Level 3 = all 6.",
)
async def plan_complexity(
    session_id: Annotated[str, "Session ID from plan_intent"],
    thought: Annotated[str, "Reasoning for complexity assessment"],
    level: Annotated[int, "Complexity 1-3"],
    estimated_sub_queries: Annotated[int, "Expected number of sub-queries"],
    estimated_tool_calls: Annotated[int, "Expected total tool calls"],
    justification: Annotated[str, "Why this complexity level"],
    confidence: Annotated[float, "Confidence 0.0-1.0"] = 1.0,
    is_revision: Annotated[bool, "True to overwrite"] = False,
) -> str:
    import json
    if not planning_engine.get_session(session_id):
        return json.dumps({"error": f"Session '{session_id}' not found. Call plan_intent first."})
    return json.dumps(planning_engine.process_phase(
        phase="complexity_assessment", thought=thought, session_id=session_id,
        is_revision=is_revision, confidence=confidence,
        phase_data={"level": level, "estimated_sub_queries": estimated_sub_queries,
                     "estimated_tool_calls": estimated_tool_calls, "justification": justification},
    ), ensure_ascii=False, indent=2)


@mcp.tool(
    name="plan_sub_query",
    output_schema=None,
    description="Phase 3: Add one sub-query. Call once per sub-query; data accumulates across calls. Set is_revision=true to replace all.",
)
async def plan_sub_query(
    session_id: Annotated[str, "Session ID from plan_intent"],
    thought: Annotated[str, "Reasoning for this sub-query"],
    id: Annotated[str, "Unique ID (e.g., 'sq1')"],
    goal: Annotated[str, "Sub-query goal"],
    expected_output: Annotated[str, "What success looks like"],
    boundary: Annotated[str, "What this excludes — mutual exclusion with siblings"],
    confidence: Annotated[float, "Confidence 0.0-1.0"] = 1.0,
    depends_on: Annotated[str, "Comma-separated prerequisite IDs"] = "",
    tool_hint: Annotated[str, "web_search | web_fetch | web_map"] = "",
    is_revision: Annotated[bool, "True to replace all sub-queries"] = False,
) -> str:
    import json
    if not planning_engine.get_session(session_id):
        return json.dumps({"error": f"Session '{session_id}' not found. Call plan_intent first."})
    item = {"id": id, "goal": goal, "expected_output": expected_output, "boundary": boundary}
    if depends_on:
        item["depends_on"] = _split_csv(depends_on)
    if tool_hint:
        item["tool_hint"] = tool_hint
    return json.dumps(planning_engine.process_phase(
        phase="query_decomposition", thought=thought, session_id=session_id,
        is_revision=is_revision, confidence=confidence, phase_data=item,
    ), ensure_ascii=False, indent=2)


@mcp.tool(
    name="plan_search_term",
    output_schema=None,
    description="Phase 4: Add one search term. Call once per term; data accumulates. First call must set approach.",
)
async def plan_search_term(
    session_id: Annotated[str, "Session ID from plan_intent"],
    thought: Annotated[str, "Reasoning for this search term"],
    term: Annotated[str, "Search query (max 8 words)"],
    purpose: Annotated[str, "Sub-query ID this serves (e.g., 'sq1')"],
    round: Annotated[int, "Execution round: 1=broad, 2+=targeted follow-up"],
    confidence: Annotated[float, "Confidence 0.0-1.0"] = 1.0,
    approach: Annotated[str, "broad_first | narrow_first | targeted (required on first call)"] = "",
    fallback_plan: Annotated[str, "Fallback if primary searches fail"] = "",
    is_revision: Annotated[bool, "True to replace all search terms"] = False,
) -> str:
    import json
    if not planning_engine.get_session(session_id):
        return json.dumps({"error": f"Session '{session_id}' not found. Call plan_intent first."})
    data = {"search_terms": [{"term": term, "purpose": purpose, "round": round}]}
    if approach:
        data["approach"] = approach
    if fallback_plan:
        data["fallback_plan"] = fallback_plan
    return json.dumps(planning_engine.process_phase(
        phase="search_strategy", thought=thought, session_id=session_id,
        is_revision=is_revision, confidence=confidence, phase_data=data,
    ), ensure_ascii=False, indent=2)


@mcp.tool(
    name="plan_tool_mapping",
    output_schema=None,
    description="Phase 5: Map a sub-query to a tool. Call once per mapping; data accumulates.",
)
async def plan_tool_mapping(
    session_id: Annotated[str, "Session ID from plan_intent"],
    thought: Annotated[str, "Reasoning for this mapping"],
    sub_query_id: Annotated[str, "Sub-query ID to map"],
    tool: Annotated[str, "web_search | web_fetch | web_map"],
    reason: Annotated[str, "Why this tool for this sub-query"],
    confidence: Annotated[float, "Confidence 0.0-1.0"] = 1.0,
    params_json: Annotated[str, "Optional JSON string for tool-specific params"] = "",
    is_revision: Annotated[bool, "True to replace all mappings"] = False,
) -> str:
    import json
    if not planning_engine.get_session(session_id):
        return json.dumps({"error": f"Session '{session_id}' not found. Call plan_intent first."})
    item = {"sub_query_id": sub_query_id, "tool": tool, "reason": reason}
    if params_json:
        try:
            item["params"] = json.loads(params_json)
        except json.JSONDecodeError:
            pass
    return json.dumps(planning_engine.process_phase(
        phase="tool_selection", thought=thought, session_id=session_id,
        is_revision=is_revision, confidence=confidence, phase_data=item,
    ), ensure_ascii=False, indent=2)


@mcp.tool(
    name="plan_execution",
    output_schema=None,
    description="Phase 6: Define execution order. parallel_groups: semicolon-separated groups of comma-separated IDs (e.g., 'sq1,sq2;sq3').",
)
async def plan_execution(
    session_id: Annotated[str, "Session ID from plan_intent"],
    thought: Annotated[str, "Reasoning for execution order"],
    parallel_groups: Annotated[str, "Parallel batches: 'sq1,sq2;sq3,sq4' (semicolon=groups, comma=IDs)"],
    sequential: Annotated[str, "Comma-separated IDs that must run in order"],
    estimated_rounds: Annotated[int, "Estimated execution rounds"],
    confidence: Annotated[float, "Confidence 0.0-1.0"] = 1.0,
    is_revision: Annotated[bool, "True to overwrite"] = False,
) -> str:
    import json
    if not planning_engine.get_session(session_id):
        return json.dumps({"error": f"Session '{session_id}' not found. Call plan_intent first."})
    parallel = [_split_csv(g) for g in parallel_groups.split(";") if g.strip()] if parallel_groups else []
    seq = _split_csv(sequential)
    return json.dumps(planning_engine.process_phase(
        phase="execution_order", thought=thought, session_id=session_id,
        is_revision=is_revision, confidence=confidence,
        phase_data={"parallel": parallel, "sequential": seq, "estimated_rounds": estimated_rounds},
    ), ensure_ascii=False, indent=2)


def main():
    import signal
    import os
    import threading

    # 信号处理（仅主线程）
    if threading.current_thread() is threading.main_thread():
        def handle_shutdown(signum, frame):
            os._exit(0)
        signal.signal(signal.SIGINT, handle_shutdown)
        if sys.platform != 'win32':
            signal.signal(signal.SIGTERM, handle_shutdown)

    # Windows 父进程监控
    if sys.platform == 'win32':
        import time
        import ctypes
        parent_pid = os.getppid()

        def is_parent_alive(pid):
            """Windows 下检查进程是否存活"""
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not handle:
                return True
            exit_code = ctypes.c_ulong()
            result = kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            kernel32.CloseHandle(handle)
            return result and exit_code.value == STILL_ACTIVE

        def monitor_parent():
            while True:
                if not is_parent_alive(parent_pid):
                    os._exit(0)
                time.sleep(2)

        threading.Thread(target=monitor_parent, daemon=True).start()

    try:
        mcp.run(transport="stdio", show_banner=False)
    except KeyboardInterrupt:
        pass
    finally:
        os._exit(0)


if __name__ == "__main__":
    main()
