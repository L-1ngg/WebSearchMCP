"""
Web Search MCP Server for CherryStudio (Async版 - Cloudflare Worker 支持)

curl_cffi 版本（无 Playwright / cloudscraper）：
- 使用 curl_cffi 发起 HTTP 请求并解析 HTML
- 默认 SSE 便于 CherryStudio 连接（也支持 stdio / streamable-http）
"""

import argparse
import asyncio
import logging
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import quote, quote_plus

from dotenv import load_dotenv

load_dotenv()

from bs4 import BeautifulSoup
from curl_cffi import requests as curl_requests
from mcp.server.fastmcp import FastMCP

# ============================================================================
# Logging
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)

# ============================================================================
# 全局配置
# ============================================================================
parser = argparse.ArgumentParser(description="Web Search MCP Server (curl_cffi)")
parser.add_argument(
    "--proxy",
    type=str,
    default=os.getenv("PROXY"),
    help="本地代理设置，例如: http://127.0.0.1:7890",
)
parser.add_argument(
    "--cf-worker",
    type=str,
    default=os.getenv("CF_WORKER"),
    help="Cloudflare Worker 地址，例如: https://xxx.xxx.workers.dev",
)
parser.add_argument(
    "--timeout",
    type=int,
    default=int(os.getenv("TIMEOUT", "60")),
    help="请求超时时间（秒），默认 60",
)
parser.add_argument(
    "--transport",
    type=str,
    default=os.getenv("TRANSPORT", "sse"),
    choices=["stdio", "sse", "streamable-http"],
    help="MCP 传输方式：stdio / sse / streamable-http（默认 sse，便于 CherryStudio 连接）",
)
parser.add_argument(
    "--host",
    type=str,
    default=os.getenv("HOST", "127.0.0.1"),
    help="SSE/HTTP 监听地址（默认 127.0.0.1）",
)
parser.add_argument(
    "--port",
    type=int,
    default=int(os.getenv("PORT", "8000")),
    help="SSE/HTTP 监听端口（默认 8000）",
)
parser.add_argument(
    "--log-level",
    type=str,
    default=os.getenv("LOG_LEVEL", "INFO"),
    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    help="日志级别（默认 INFO）",
)
parser.add_argument(
    "--http-version",
    type=str,
    default=os.getenv("HTTP_VERSION", "v1"),
    choices=["v1", "v2", "v2tls", "v2_prior_knowledge", "v3", "v3only"],
    help="HTTP 协议版本（默认 v1，代理/Worker 场景更稳）",
)

CLI_ARGS, _ = parser.parse_known_args()
PROXY_CONFIG = CLI_ARGS.proxy
CF_WORKER_URL = CLI_ARGS.cf_worker
REQUEST_TIMEOUT_S = CLI_ARGS.timeout
TRANSPORT = CLI_ARGS.transport
HOST = CLI_ARGS.host
PORT = CLI_ARGS.port
LOG_LEVEL = CLI_ARGS.log_level
HTTP_VERSION = CLI_ARGS.http_version

# 兼容处理：MCP 客户端可能把 "--arg value" 合并成一个字符串
for arg in sys.argv[1:]:
    if arg.startswith("--proxy "):
        PROXY_CONFIG = arg.split(" ", 1)[1]
    elif arg.startswith("--cf-worker "):
        CF_WORKER_URL = arg.split(" ", 1)[1]
    elif arg.startswith("--timeout "):
        try:
            REQUEST_TIMEOUT_S = int(arg.split(" ", 1)[1])
        except Exception:
            pass
    elif arg.startswith("--transport "):
        TRANSPORT = arg.split(" ", 1)[1]
    elif arg.startswith("--host "):
        HOST = arg.split(" ", 1)[1]
    elif arg.startswith("--port "):
        try:
            PORT = int(arg.split(" ", 1)[1])
        except Exception:
            pass
    elif arg.startswith("--log-level "):
        LOG_LEVEL = arg.split(" ", 1)[1]
    elif arg.startswith("--http-version "):
        HTTP_VERSION = arg.split(" ", 1)[1]

logging.getLogger().setLevel(getattr(logging, str(LOG_LEVEL).upper(), logging.INFO))

logger.info(f"[DEBUG] sys.argv: {sys.argv}")
logger.info(f"[DEBUG] PROXY_CONFIG: {PROXY_CONFIG}")
logger.info(f"[DEBUG] CF_WORKER_URL: {CF_WORKER_URL}")
logger.info(f"[DEBUG] REQUEST_TIMEOUT_S: {REQUEST_TIMEOUT_S}")
logger.info(f"[DEBUG] TRANSPORT: {TRANSPORT}")
logger.info(f"[DEBUG] HOST: {HOST}")
logger.info(f"[DEBUG] PORT: {PORT}")
logger.info(f"[DEBUG] LOG_LEVEL: {LOG_LEVEL}")
logger.info(f"[DEBUG] HTTP_VERSION: {HTTP_VERSION}")

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)
MAX_TOKEN_LIMIT = 10000
CURL_IMPERSONATE = "chrome110"


def _get_proxies() -> Optional[Dict[str, str]]:
    if PROXY_CONFIG:
        proxies = {"http": PROXY_CONFIG, "https": PROXY_CONFIG}
        logger.debug("PROXIES enabled: %s", proxies)
        return proxies
    return None


def _get_target_url(original_url: str) -> str:
    if CF_WORKER_URL:
        worker_base = CF_WORKER_URL.rstrip("/")
        encoded_target = quote(original_url)
        wrapped = f"{worker_base}?url={encoded_target}"
        logger.debug("CF_WORKER wrap: original=%s wrapped=%s", original_url, wrapped)
        return wrapped
    return original_url


def _limit_content_length(content: str) -> Tuple[str, bool]:
    estimated_tokens = len(content) // 4
    if estimated_tokens > MAX_TOKEN_LIMIT:
        chars_to_keep = MAX_TOKEN_LIMIT * 4
        return content[:chars_to_keep], True
    return content, False


def _curl_get(
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    timeout_s: Optional[int] = None,
) -> curl_requests.Response:
    timeout_s = int(timeout_s or REQUEST_TIMEOUT_S or 60)
    request_headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "close",
        **(headers or {}),
    }
    logger.debug(
        "HTTP GET url=%s timeout_s=%s proxy=%s impersonate=%s",
        url,
        timeout_s,
        PROXY_CONFIG,
        CURL_IMPERSONATE,
    )
    response = curl_requests.get(
        url,
        headers=request_headers,
        proxies=_get_proxies(),
        timeout=timeout_s,
        allow_redirects=True,
        impersonate=CURL_IMPERSONATE,
        http_version=HTTP_VERSION,
    )
    logger.debug(
        "HTTP RESP url=%s status=%s bytes=%s final_url=%s content_type=%s",
        url,
        getattr(response, "status_code", None),
        len(getattr(response, "content", b"") or b""),
        getattr(response, "url", None),
        (getattr(response, "headers", {}) or {}).get("content-type"),
    )
    response.raise_for_status()
    return response


def _curl_get_body_bytes_streaming(
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    timeout_s: Optional[int] = None,
    max_bytes: int = 512_000,
    stop_when_has_results: Optional[Callable[[bytes], bool]] = None,
) -> tuple[curl_requests.Response, bytes]:
    timeout_s = int(timeout_s or REQUEST_TIMEOUT_S or 60)
    request_headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "close",
        **(headers or {}),
    }

    response: Optional[curl_requests.Response] = None
    buf = bytearray()
    total = 0
    try:
        logger.debug(
            "HTTP STREAM GET url=%s timeout_s=%s proxy=%s impersonate=%s http_version=%s max_bytes=%s",
            url,
            timeout_s,
            PROXY_CONFIG,
            CURL_IMPERSONATE,
            HTTP_VERSION,
            max_bytes,
        )
        response = curl_requests.get(
            url,
            headers=request_headers,
            proxies=_get_proxies(),
            timeout=timeout_s,
            allow_redirects=True,
            impersonate=CURL_IMPERSONATE,
            http_version=HTTP_VERSION,
            stream=True,
        )
        response.raise_for_status()

        for chunk in response.iter_content():
            if not chunk:
                continue
            buf.extend(chunk)
            total = len(buf)

            if total >= max_bytes:
                logger.debug("HTTP STREAM reached max_bytes=%s url=%s", max_bytes, url)
                break

            # 避免每个 chunk 都做全文解析，先做一次轻量 gate
            if stop_when_has_results is not None and total >= 16_000:
                if stop_when_has_results(bytes(buf)):
                    logger.debug("HTTP STREAM early-stop by heuristic url=%s bytes=%s", url, total)
                    break

        body = bytes(buf)
        logger.debug(
            "HTTP STREAM DONE url=%s status=%s bytes=%s final_url=%s content_type=%s",
            url,
            getattr(response, "status_code", None),
            len(body),
            getattr(response, "url", None),
            (getattr(response, "headers", {}) or {}).get("content-type"),
        )
        return response, body
    finally:
        if response is not None:
            try:
                response.close()
            except Exception:
                pass


def _curl_get_with_retries(
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    timeout_s: Optional[int] = None,
    retries: int = 2,
) -> curl_requests.Response:
    timeout_s = int(timeout_s or REQUEST_TIMEOUT_S or 60)
    last_error: Optional[Exception] = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            return _curl_get(url, headers=headers, timeout_s=timeout_s)
        except Exception as e:
            last_error = e
            message = str(e)
            is_timeout = ("curl: (28)" in message) or ("Operation timed out" in message)
            logger.warning(
                "HTTP ERROR attempt=%s/%s url=%s timeout_s=%s is_timeout=%s err=%s",
                attempt,
                retries,
                url,
                timeout_s,
                is_timeout,
                message,
            )
            if attempt >= retries or not is_timeout:
                raise
            timeout_s = max(timeout_s * 2, timeout_s + 30)
            time.sleep(0.4 * attempt)
    assert last_error is not None
    raise last_error


mcp = FastMCP("web-search", host=HOST, port=PORT, log_level=str(LOG_LEVEL).upper())


async def _search_brave_core(query: str, max_results: int = 20) -> List[Dict[str, str]]:
    def _extract_brave_results(html: str, limit: int) -> List[Dict[str, str]]:
        soup = BeautifulSoup(html, "lxml")
        items = soup.select('[data-type="web"]')
        if not items:
            items = soup.select(".snippet")

        if limit and limit > 0:
            items = items[:limit]

        extracted: List[Dict[str, str]] = []
        for item in items:
            link = item.select_one("a[href]")
            if not link:
                continue
            href = link.get("href", "")
            if not href.startswith("http"):
                continue
            if CF_WORKER_URL and CF_WORKER_URL in href:
                continue

            title_elem = item.select_one(".snippet-title, .title")
            desc_elem = item.select_one(".snippet-description, .snippet-content, .description")

            extracted.append(
                {
                    "title": title_elem.get_text(strip=True) if title_elem else "No Title",
                    "url": href,
                    "description": desc_elem.get_text(strip=True) if desc_elem else "",
                }
            )
        return extracted

    def _fetch_and_parse() -> List[Dict[str, str]]:
        results: List[Dict[str, str]] = []
        target_url = f"https://search.brave.com/search?q={quote_plus(query)}"
        visit_url = _get_target_url(target_url)

        logger.info(f"正在搜索: {query}")
        logger.debug("Brave target_url=%s visit_url=%s", target_url, visit_url)
        if CF_WORKER_URL:
            logger.info(f"Via Cloudflare Worker: {visit_url}")
        if PROXY_CONFIG:
            logger.info(f"Via local proxy: {PROXY_CONFIG}")

        def _has_enough_results(body: bytes) -> bool:
            if len(body) < 16_000:
                return False
            # 快速 gate：出现结果结构再解析，避免频繁构建 soup
            if b"data-type" not in body and b"snippet" not in body:
                return False
            try:
                html = body.decode("utf-8", errors="replace")
            except Exception:
                return False
            parsed = _extract_brave_results(html, max_results)
            # 只要已经能提取到结果，就停止继续等网络把尾部传完（Worker/代理场景经常卡在尾部）
            return len(parsed) >= max(1, min(max_results, 3))

        started = time.time()
        last_error: Optional[Exception] = None
        timeout_s = int(REQUEST_TIMEOUT_S or 60)
        for attempt in range(1, 3):
            try:
                _, body = _curl_get_body_bytes_streaming(
                    visit_url,
                    timeout_s=timeout_s,
                    max_bytes=512_000,
                    stop_when_has_results=_has_enough_results,
                )
                logger.debug("Brave fetch elapsed_ms=%s", int((time.time() - started) * 1000))
                html = body.decode("utf-8", errors="replace")
                results = _extract_brave_results(html, max_results)
                logger.info(f"搜索完成，找到 {len(results)} 个结果")
                return results
            except Exception as e:
                last_error = e
                message = str(e)
                is_timeout = ("curl: (28)" in message) or ("Operation timed out" in message)
                logger.warning(
                    "Brave search error attempt=%s/2 url=%s timeout_s=%s is_timeout=%s err=%s",
                    attempt,
                    visit_url,
                    timeout_s,
                    is_timeout,
                    message,
                )
                if attempt >= 2 or not is_timeout:
                    break
                timeout_s = max(timeout_s * 2, timeout_s + 30)
                time.sleep(0.4 * attempt)
        assert last_error is not None
        raise last_error

    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, _fetch_and_parse)
    except Exception as e:
        logger.exception("搜索过程发生错误: %s", e)
        return [{"title": "搜索失败", "url": "", "description": f"错误: {str(e)}"}]


# ============================================================================
# Fetch 工具 (支持 CF Worker)
# ============================================================================
@mcp.tool()
async def fetch_html(url: str, *, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    def _fetch() -> Dict[str, Any]:
        target_url = _get_target_url(url)
        logger.debug("fetch_html url=%s target_url=%s", url, target_url)
        response = _curl_get_with_retries(target_url, headers=headers, timeout_s=REQUEST_TIMEOUT_S, retries=2)

        html_content, was_truncated = _limit_content_length(response.text)
        return {
            "success": True,
            "url": url,
            "via_worker": bool(CF_WORKER_URL),
            "status_code": response.status_code,
            "html": html_content,
            "truncated": was_truncated,
        }

    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, _fetch)
    except Exception as e:
        logger.exception("抓取失败 %s: %s", url, e)
        return {"success": False, "url": url, "error": str(e)}


@mcp.tool()
async def fetch_text(url: str, *, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    def _fetch() -> Dict[str, Any]:
        target_url = _get_target_url(url)
        logger.debug("fetch_text url=%s target_url=%s", url, target_url)
        response = _curl_get_with_retries(target_url, headers=headers, timeout_s=REQUEST_TIMEOUT_S, retries=2)

        soup = BeautifulSoup(response.text, "lxml")
        for tag in soup(["script", "style", "header", "footer", "nav"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        limited_text, was_truncated = _limit_content_length(text)
        return {
            "success": True,
            "url": url,
            "via_worker": bool(CF_WORKER_URL),
            "text": limited_text,
            "truncated": was_truncated,
        }

    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, _fetch)
    except Exception as e:
        logger.exception("抓取文本失败 %s: %s", url, e)
        return {"success": False, "url": url, "error": str(e)}


@mcp.tool()
async def fetch_metadata(url: str, *, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    def _fetch() -> Dict[str, Any]:
        target_url = _get_target_url(url)
        logger.debug("fetch_metadata url=%s target_url=%s", url, target_url)
        response = _curl_get_with_retries(target_url, headers=headers, timeout_s=REQUEST_TIMEOUT_S, retries=2)

        soup = BeautifulSoup(response.text, "lxml")
        title = soup.find("title").get_text(strip=True) if soup.find("title") else ""
        description = ""
        desc_tag = soup.find("meta", attrs={"name": "description"})
        if desc_tag:
            description = desc_tag.get("content", "")

        links = []
        for a in soup.find_all("a", href=True, limit=50):
            links.append({"text": a.get_text(strip=True), "href": a["href"]})

        links_str = str(links)
        _, was_truncated = _limit_content_length(links_str)
        if was_truncated:
            avg_length = len(links_str) / len(links) if links else 0
            keep_count = max(
                1, int(MAX_TOKEN_LIMIT * 4 / avg_length) if avg_length > 0 else 0
            )
            links = links[:keep_count]

        return {
            "success": True,
            "url": url,
            "via_worker": bool(CF_WORKER_URL),
            "title": title,
            "description": description,
            "links": links,
            "truncated": was_truncated,
        }

    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, _fetch)
    except Exception as e:
        logger.exception("提取元数据失败 %s: %s", url, e)
        return {"success": False, "url": url, "error": str(e)}


@mcp.tool()
async def web_search(query: str, max_results: int = 10) -> List[Dict[str, str]]:
    logger.info(f"收到搜索请求: query='{query}'")
    return await _search_brave_core(query=query, max_results=max_results)


def main():
    logger.info("Web Search MCP Server 启动中...")
    if CF_WORKER_URL:
        logger.info(f"启用 Cloudflare Worker 代理: {CF_WORKER_URL}")
        logger.info("注意：流量将通过 Worker 转发，目标网站看到的 IP 为 Cloudflare 节点 IP")

    if PROXY_CONFIG:
        logger.info(f"使用本地代理: {PROXY_CONFIG}")

    logger.info("等待 MCP 客户端连接...")
    transport = TRANSPORT if TRANSPORT in ("stdio", "sse", "streamable-http") else "sse"
    if transport != TRANSPORT:
        logger.warning("未知 transport=%s，回退到 %s", TRANSPORT, transport)

    if transport in ("sse", "streamable-http"):
        logger.info("监听: http://%s:%s", HOST, PORT)
    if transport == "sse":
        logger.info("SSE endpoint: http://%s:%s%s", HOST, PORT, mcp.settings.sse_path)
        logger.info("Messages endpoint: http://%s:%s%s", HOST, PORT, mcp.settings.message_path)
    if transport == "streamable-http":
        logger.info("StreamableHTTP endpoint: http://%s:%s%s", HOST, PORT, mcp.settings.streamable_http_path)

    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
