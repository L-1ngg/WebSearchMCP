#!/usr/bin/env python3
# /// script
# dependencies = ['mcp', 'beautifulsoup4', 'playwright', 'cloudscraper']
# ///

"""
Web Search MCP Server for CherryStudio (Async版)

变更：
- 使用 playwright.async_api，避免在事件循环内调用同步API的报错
- 添加了第一代WebSearchMCP的fetch工具
- 代理配置统一，通过命令行 --proxy 传入，适用于所有工具
"""

import argparse
import asyncio
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus

# 设置 Playwright 浏览器路径为项目本地目录（必须在导入 playwright 之前）
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["PLAYWRIGHT_BROWSERS_PATH"] = os.path.join(_SCRIPT_DIR, ".playwright-browsers")

import cloudscraper
from bs4 import BeautifulSoup
from mcp.server.fastmcp import FastMCP
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright

# ============================================================================
# Logging（STDIO 模式：只能写 stderr）
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)

# ============================================================================
# 全局配置（proxy 通过命令行传入）
# ============================================================================
parser = argparse.ArgumentParser(description="Web Search MCP Server")
parser.add_argument(
    "--proxy",
    type=str,
    default=None,
    help="例如: http://127.0.0.1:7890（不传则不使用代理）",
)
# 兼容 MCP 可能传入的其它未知参数
CLI_ARGS, _ = parser.parse_known_args()
PROXY_CONFIG = CLI_ARGS.proxy

# 全局 scraper 实例（用于fetch工具）
_scraper: Optional[cloudscraper.CloudScraper] = None

# Playwright配置
HEADLESS = True
TIMEOUT_MS = 20_000
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)

# 最大token限制 (10k)
MAX_TOKEN_LIMIT = 10000


# ============================================================================
# 全局浏览器管理器（持久化浏览器实例）
# ============================================================================
class BrowserManager:
    """管理持久化的浏览器实例"""

    def __init__(self):
        self._playwright = None
        self._browser = None
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        """获取锁，确保在正确的事件循环中创建"""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def get_browser(self):
        """获取浏览器实例，如果不存在则创建"""
        async with self._get_lock():
            if self._browser is None or not self._browser.is_connected():
                await self._start_browser()
            return self._browser

    async def _start_browser(self):
        """启动浏览器"""
        try:
            # 如果有旧的实例，先清理
            await self._cleanup()

            self._playwright = await async_playwright().start()

            launch_kwargs = {"headless": HEADLESS}
            if PROXY_CONFIG:
                launch_kwargs["proxy"] = {"server": PROXY_CONFIG}
                logger.info(f"浏览器使用代理: {PROXY_CONFIG}")

            self._browser = await self._playwright.chromium.launch(**launch_kwargs)
            logger.info("持久化浏览器已启动")

        except Exception as e:
            logger.error(f"启动浏览器失败: {e}")
            raise

    async def new_page(self):
        """创建新页面"""
        browser = await self.get_browser()
        context = await browser.new_context(user_agent=USER_AGENT)
        page = await context.new_page()
        page.set_default_timeout(TIMEOUT_MS)
        return page, context

    async def _cleanup(self):
        """清理浏览器资源"""
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
        """关闭浏览器"""
        async with self._get_lock():
            await self._cleanup()
            logger.info("持久化浏览器已关闭")


# 全局浏览器管理器实例
_browser_manager = BrowserManager()

# ============================================================================
# 初始化 FastMCP Server
# ============================================================================
mcp = FastMCP("web-search")


# ============================================================================
# 辅助函数
# ============================================================================
def _get_scraper() -> cloudscraper.CloudScraper:
    """获取全局 scraper 实例"""
    global _scraper
    if _scraper is None:
        _scraper = cloudscraper.create_scraper()
    return _scraper


def _get_proxies() -> Optional[Dict[str, str]]:
    """获取代理配置（用于cloudscraper）"""
    if PROXY_CONFIG:
        return {
            "http": PROXY_CONFIG,
            "https": PROXY_CONFIG,
        }
    return None


def _limit_content_length(content: str) -> Tuple[str, bool]:
    """
    限制内容长度，防止返回过大的响应

    参数：
      - content: 原始内容

    返回：
      - 截断后的内容
      - 是否进行了截断
    """
    # 粗略估算 token 数量 (大约每4个字符1个token)
    estimated_tokens = len(content) // 4

    # 如果估计的token数超过限制，截断内容
    if estimated_tokens > MAX_TOKEN_LIMIT:
        # 计算需要保留的字符数
        chars_to_keep = MAX_TOKEN_LIMIT * 4
        truncated_content = content[:chars_to_keep]
        return truncated_content, True

    return content, False


# ============================================================================
# Brave Search 核心（Async）- 使用持久化浏览器
# ============================================================================
async def _search_brave_core(
    query: str,
    max_results: int = 20,
) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    context = None

    try:
        # 使用持久化浏览器管理器
        page, context = await _browser_manager.new_page()

        url = f"https://search.brave.com/search?q={quote_plus(query)}"
        logger.info(f"正在搜索: {query}")
        await page.goto(url, wait_until="domcontentloaded")

        # 等待结果区域（尽量不死等）
        try:
            await page.wait_for_selector('[data-type="web"]', timeout=TIMEOUT_MS)
        except PlaywrightTimeoutError:
            logger.warning("等待选择器超时，尝试解析已有内容")

        html = await page.content()

        # 解析搜索结果
        soup = BeautifulSoup(html, "html.parser")
        items = soup.select('[data-type="web"]')

        if max_results and max_results > 0:
            items = items[:max_results]

        for item in items:
            link = item.select_one("a[href]")
            if not link:
                continue
            href = link.get("href", "")
            if not href.startswith("http"):
                continue

            # 标题
            title_elem = item.select_one(".snippet-title") or item.select_one(
                "h2, h3, .title, .result-title"
            )
            # 描述
            desc_elem = item.select_one(
                ".snippet-description, .result-snippet, .snippet__description"
            )

            title = (title_elem.text or "").strip() if title_elem else "No Title"
            body = (desc_elem.text or "").strip() if desc_elem else ""

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
        return [
            {
                "title": "搜索失败",
                "url": "",
                "description": f"Playwright错误: {str(e)}",
            }
        ]
    except Exception as e:
        logger.error(f"搜索过程发生错误: {e}")
        return [
            {
                "title": "搜索失败",
                "url": "",
                "description": f"错误: {str(e)}",
            }
        ]
    finally:
        # 只关闭context，不关闭browser
        if context:
            try:
                await context.close()
            except Exception:
                pass

    return results


# ============================================================================
# 第一代 fetch 工具 (异步改造)
# ============================================================================
@mcp.tool()
async def fetch_html(
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    抓取网页HTML内容

    参数：
      - url: 目标网址
      - headers: 可选的请求头

    返回：包含HTML内容或错误信息
    """
    try:
        # 在异步上下文中运行同步代码
        def _fetch():
            scraper = _get_scraper()
            proxies = _get_proxies()

            response = scraper.get(
                url, headers=headers or {}, proxies=proxies, timeout=10
            )
            response.raise_for_status()

            # 限制响应内容长度
            html_content, was_truncated = _limit_content_length(response.text)

            return {
                "success": True,
                "url": url,
                "status_code": response.status_code,
                "html": html_content,
                "length": len(html_content),
                "truncated": was_truncated,
            }

        # 在线程池中执行同步代码，避免阻塞事件循环
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _fetch)
        return result
    except Exception as e:
        logger.error(f"抓取失败 {url}: {e}")
        return {
            "success": False,
            "url": url,
            "error": str(e),
        }


@mcp.tool()
async def fetch_text(
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    抓取网页并提取纯文本（去除HTML标签）

    参数：
      - url: 目标网址
      - headers: 可选的请求头

    返回：包含纯文本内容
    """
    try:
        # 在异步上下文中运行同步代码
        def _fetch():
            scraper = _get_scraper()
            proxies = _get_proxies()

            response = scraper.get(
                url, headers=headers or {}, proxies=proxies, timeout=10
            )
            response.raise_for_status()

            # 解析HTML并提取文本
            soup = BeautifulSoup(response.text, "lxml")

            # 移除script和style标签
            for tag in soup(["script", "style", "header", "footer", "nav"]):
                tag.decompose()

            text = soup.get_text(separator="\n", strip=True)

            # 限制文本内容长度
            limited_text, was_truncated = _limit_content_length(text)

            return {
                "success": True,
                "url": url,
                "text": limited_text,
                "length": len(limited_text),
                "truncated": was_truncated,
            }

        # 在线程池中执行同步代码，避免阻塞事件循环
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _fetch)
        return result
    except Exception as e:
        logger.error(f"抓取文本失败 {url}: {e}")
        return {
            "success": False,
            "url": url,
            "error": str(e),
        }


@mcp.tool()
async def fetch_metadata(
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    抓取网页并提取元数据（标题、描述、链接等）

    参数：
      - url: 目标网址
      - headers: 可选的请求头

    返回：包含标题、描述、链接等元数据
    """
    try:
        # 在异步上下文中运行同步代码
        def _fetch():
            scraper = _get_scraper()
            proxies = _get_proxies()

            response = scraper.get(
                url, headers=headers or {}, proxies=proxies, timeout=10
            )
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "lxml")

            # 提取标题
            title = ""
            title_tag = soup.find("title")
            if title_tag:
                title = title_tag.get_text(strip=True)

            # 提取描述
            description = ""
            desc_tag = soup.find("meta", attrs={"name": "description"})
            if desc_tag:
                description = desc_tag.get("content", "")

            # 提取所有链接
            links = []
            for a in soup.find_all("a", href=True, limit=50):
                links.append({"text": a.get_text(strip=True), "href": a["href"]})

            # 检查链接数据大小是否超过限制
            links_str = str(links)
            _, was_truncated = _limit_content_length(links_str)

            # 如果链接数据超过限制，截断链接列表
            if was_truncated:
                # 估算每个链接的平均长度
                avg_length = len(links_str) / len(links) if links else 0
                # 计算可以保留的链接数量
                keep_count = (
                    int(MAX_TOKEN_LIMIT * 4 / avg_length) if avg_length > 0 else 0
                )
                # 确保至少保留一个链接
                keep_count = max(1, keep_count)
                # 截断链接列表
                links = links[:keep_count]
                logger.info(f"元数据链接超过token限制，截断为{keep_count}个链接")

            return {
                "success": True,
                "url": url,
                "title": title,
                "description": description,
                "links": links,
                "link_count": len(links),
                "truncated": was_truncated,
            }

        # 在线程池中执行同步代码，避免阻塞事件循环
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _fetch)
        return result
    except Exception as e:
        logger.error(f"提取元数据失败 {url}: {e}")
        return {
            "success": False,
            "url": url,
            "error": str(e),
        }


# ============================================================================
# MCP 工具（Brave搜索）
# ============================================================================
@mcp.tool()
async def web_search(
    query: str,
    max_results: int = 10,
) -> List[Dict[str, str]]:
    """
    使用 Brave 搜索引擎进行网页搜索

    Args:
        query: 搜索关键词（必填）
        max_results: 返回的最大结果数，默认 10

    Returns:
        搜索结果列表，每项包含：
        - title: 标题
        - url: 链接
        - description: 描述
    """
    logger.info(f"收到搜索请求: query='{query}', max_results={max_results}")
    return await _search_brave_core(query=query, max_results=max_results)


# ============================================================================
# 启动
# ============================================================================
def main():
    logger.info("Web Search MCP Server 启动中...")
    if PROXY_CONFIG:
        logger.info(f"使用代理: {PROXY_CONFIG}")
    else:
        logger.info("未配置代理，直连模式")

    logger.info("浏览器将在首次搜索时启动（懒加载）")
    logger.info("等待 MCP 客户端连接...")

    # 启动 MCP Server（STDIO 模式）
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
