"""
Web Search MCP Server for CherryStudio (Async版 - Cloudflare Worker 支持)

变更：
- 新增 --cf-worker 参数，支持通过 Cloudflare Worker 代理流量
- 保持原有的 Playwright 和 cloudscraper 逻辑
"""

import argparse
import asyncio
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus, quote

# 设置 Playwright 浏览器路径为项目本地目录
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["PLAYWRIGHT_BROWSERS_PATH"] = os.path.join(_SCRIPT_DIR, ".playwright-browsers")

import cloudscraper
from bs4 import BeautifulSoup
from mcp.server.fastmcp import FastMCP
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright

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
parser = argparse.ArgumentParser(description="Web Search MCP Server")
parser.add_argument(
    "--proxy",
    type=str,
    default=None,
    help="本地代理设置，例如: http://127.0.0.1:7890",
)
parser.add_argument(
    "--cf-worker",
    type=str,
    default=None,
    help="Cloudflare Worker 地址，例如: https://xxx.xxx.workers.dev",
)

CLI_ARGS, _ = parser.parse_known_args()
PROXY_CONFIG = CLI_ARGS.proxy
CF_WORKER_URL = CLI_ARGS.cf_worker

# 兼容处理：MCP 客户端可能把 "--arg value" 合并成一个字符串
if PROXY_CONFIG is None or CF_WORKER_URL is None:
    for arg in sys.argv[1:]:
        if arg.startswith("--proxy ") and PROXY_CONFIG is None:
            PROXY_CONFIG = arg.split(" ", 1)[1]
        elif arg.startswith("--cf-worker ") and CF_WORKER_URL is None:
            CF_WORKER_URL = arg.split(" ", 1)[1]

# 调试：打印接收到的参数
logger.info(f"[DEBUG] sys.argv: {sys.argv}")
logger.info(f"[DEBUG] PROXY_CONFIG: {PROXY_CONFIG}")
logger.info(f"[DEBUG] CF_WORKER_URL: {CF_WORKER_URL}")

# 全局 scraper 实例
_scraper: Optional[cloudscraper.CloudScraper] = None

# Playwright配置
HEADLESS = True
TIMEOUT_MS = 20_000
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)

MAX_TOKEN_LIMIT = 10000


# ============================================================================
# 辅助函数：处理 URL 路由
# ============================================================================
def _get_target_url(original_url: str) -> str:
    """
    如果配置了 CF Worker，则将请求包装到 Worker URL 中。
    否则返回原始 URL。
    """
    if CF_WORKER_URL:
        # 移除末尾斜杠以防万一
        worker_base = CF_WORKER_URL.rstrip("/")
        # 构造类似 https://worker.dev?url=https://google.com 的请求
        encoded_target = quote(original_url) # 使用 quote 而不是 quote_plus 以保持 :// 的完整性兼容性
        return f"{worker_base}?url={encoded_target}"
    return original_url


def _get_scraper() -> cloudscraper.CloudScraper:
    global _scraper
    if _scraper is None:
        _scraper = cloudscraper.create_scraper()
    return _scraper


def _get_proxies() -> Optional[Dict[str, str]]:
    """
    获取代理配置。
    注意：如果使用了 CF Worker，通常不需要本地代理去连接 CF Worker（除非本地无法直连 CF）。
    这里保持逻辑：如果配置了 proxy，就让 requests/scraper 走这个 proxy。
    """
    if PROXY_CONFIG:
        return {
            "http": PROXY_CONFIG,
            "https": PROXY_CONFIG,
        }
    return None


def _limit_content_length(content: str) -> Tuple[str, bool]:
    estimated_tokens = len(content) // 4
    if estimated_tokens > MAX_TOKEN_LIMIT:
        chars_to_keep = MAX_TOKEN_LIMIT * 4
        truncated_content = content[:chars_to_keep]
        return truncated_content, True
    return content, False


# ============================================================================
# 全局浏览器管理器
# ============================================================================
class BrowserManager:
    def __init__(self):
        self._playwright = None
        self._browser = None
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def get_browser(self):
        async with self._get_lock():
            if self._browser is None or not self._browser.is_connected():
                await self._start_browser()
            return self._browser

    async def _start_browser(self):
        try:
            await self._cleanup()
            self._playwright = await async_playwright().start()

            launch_kwargs = {"headless": HEADLESS}
            # 如果配置了本地代理，Playwright 依然使用它
            # 注意：如果使用了 CF Worker，浏览器访问的是 Worker 地址
            if PROXY_CONFIG:
                launch_kwargs["proxy"] = {"server": PROXY_CONFIG}
                logger.info(f"浏览器使用本地代理: {PROXY_CONFIG}")

            self._browser = await self._playwright.chromium.launch(**launch_kwargs)
            logger.info("持久化浏览器已启动")

        except Exception as e:
            logger.error(f"启动浏览器失败: {e}")
            raise

    async def new_page(self):
        browser = await self.get_browser()
        context = await browser.new_context(user_agent=USER_AGENT, java_script_enabled=False)
        page = await context.new_page()
        page.set_default_timeout(TIMEOUT_MS)
        return page, context

    async def _cleanup(self):
        if self._browser:
            try:
                await self._browser.close()
            except Exception:
                pass
            self._browser = None

        if self._playwright:
            try:
                await self._playwright.stop()
            except Exception:
                pass
            self._playwright = None

    async def close(self):
        async with self._get_lock():
            await self._cleanup()
            logger.info("持久化浏览器已关闭")


_browser_manager = BrowserManager()
mcp = FastMCP("web-search")


# ============================================================================
# Brave Search 核心
# ============================================================================
async def _search_brave_core(
    query: str,
    max_results: int = 20,
) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    context = None

    try:
        page, context = await _browser_manager.new_page()

        # 构造原始 Brave 搜索 URL
        target_url = f"https://search.brave.com/search?q={quote_plus(query)}"
        
        # 转换为实际访问的 URL (直连 或 经 CF Worker)
        visit_url = _get_target_url(target_url)

        logger.info(f"正在搜索: {query}")
        if CF_WORKER_URL:
            logger.info(f"Via Cloudflare Worker: {visit_url}")

        # 【修改点1】使用 'commit' 策略
        # 只要服务器发回了 HTML 响应头就算成功，不再等待 DOM 加载完成
        # 配合禁用 JS，速度极快
        await page.goto(visit_url, wait_until="commit", timeout=60000)

        # 【修改点2】移除 wait_for_selector，改用简单的缓冲
        # 因为没有 JS，DOM 事件可能不完整，直接等 1 秒让 HTML 接收完毕即可
        await asyncio.sleep(1.5)

        html = await page.content()
        soup = BeautifulSoup(html, "html.parser")
        
        # 【修改点3】Brave 静态 HTML 的选择器优化
        # 你的 curl 结果显示 content 都在 #results 里面
        items = soup.select('[data-type="web"]') # 保持原样，通常有效
        
        # 如果标准选择器没找到，尝试备用方案 (针对无 JS 版页面结构)
        if not items:
            items = soup.select('.snippet') 

        if max_results and max_results > 0:
            items = items[:max_results]

        for item in items:
            link = item.select_one("a[href]")
            if not link:
                continue
            href = link.get("href", "")
            
            # 过滤
            if not href.startswith("http"):
                continue
            if CF_WORKER_URL and CF_WORKER_URL in href:
                continue

            # 提取标题和描述
            title_elem = item.select_one(".snippet-title, .title")
            desc_elem = item.select_one(".snippet-description, .snippet-content, .description")

            title = (title_elem.get_text(strip=True) if title_elem else "No Title")
            body = (desc_elem.get_text(strip=True) if desc_elem else "")

            results.append(
                {
                    "title": title,
                    "url": href,
                    "description": body,
                }
            )

        logger.info(f"搜索完成，找到 {len(results)} 个结果")

    except PlaywrightError as e:
        logger.error(f"Playwright 错误: {e}")
        # 返回部分结果而不是直接报错
        if results:
            return results
        return [{"title": "搜索失败", "url": "", "description": f"Playwright错误: {str(e)}"}]
    except Exception as e:
        logger.error(f"搜索过程发生错误: {e}")
        return [{"title": "搜索失败", "url": "", "description": f"错误: {str(e)}"}]
    finally:
        if context:
            try:
                await context.close()
            except Exception:
                pass

    return results

# ============================================================================
# Fetch 工具 (支持 CF Worker)
# ============================================================================
@mcp.tool()
async def fetch_html(url: str, *, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    try:
        def _fetch():
            scraper = _get_scraper()
            proxies = _get_proxies()
            
            # 使用 Worker 包装 URL
            target_url = _get_target_url(url)
            
            # 如果用了 Worker，headers 中的 Host 可能需要调整，通常 cloudscraper 会自动处理
            # 但我们要确保传给 Worker 的请求头是合理的。
            # 这里简单起见，直接请求 Worker，Headers 由 cloudscraper 默认处理
            
            response = scraper.get(
                target_url, headers=headers or {}, proxies=proxies, timeout=15
            )
            response.raise_for_status()

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
        result = await loop.run_in_executor(None, _fetch)
        return result
    except Exception as e:
        logger.error(f"抓取失败 {url}: {e}")
        return {"success": False, "url": url, "error": str(e)}


@mcp.tool()
async def fetch_text(url: str, *, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    try:
        def _fetch():
            scraper = _get_scraper()
            proxies = _get_proxies()
            target_url = _get_target_url(url)

            response = scraper.get(
                target_url, headers=headers or {}, proxies=proxies, timeout=15
            )
            response.raise_for_status()

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
        result = await loop.run_in_executor(None, _fetch)
        return result
    except Exception as e:
        logger.error(f"抓取文本失败 {url}: {e}")
        return {"success": False, "url": url, "error": str(e)}


@mcp.tool()
async def fetch_metadata(url: str, *, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    try:
        def _fetch():
            scraper = _get_scraper()
            proxies = _get_proxies()
            target_url = _get_target_url(url)

            response = scraper.get(
                target_url, headers=headers or {}, proxies=proxies, timeout=15
            )
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "lxml")
            title = ""
            if soup.find("title"):
                title = soup.find("title").get_text(strip=True)

            description = ""
            desc_tag = soup.find("meta", attrs={"name": "description"})
            if desc_tag:
                description = desc_tag.get("content", "")

            links = []
            for a in soup.find_all("a", href=True, limit=50):
                # 注意：经过 Worker 后，HTML 中的相对路径链接可能会失效
                # 解析出来的 href 仍然是原始 HTML 中的样子
                links.append({"text": a.get_text(strip=True), "href": a["href"]})

            links_str = str(links)
            _, was_truncated = _limit_content_length(links_str)
            if was_truncated:
                avg_length = len(links_str) / len(links) if links else 0
                keep_count = max(1, int(MAX_TOKEN_LIMIT * 4 / avg_length) if avg_length > 0 else 0)
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
        result = await loop.run_in_executor(None, _fetch)
        return result
    except Exception as e:
        logger.error(f"提取元数据失败 {url}: {e}")
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
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
