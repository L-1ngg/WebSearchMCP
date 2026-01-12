# Web Search MCP Server

一个基于 MCP (Model Context Protocol) 的网页搜索服务器，支持 Brave 搜索和网页抓取功能。

> **推荐**: 本 MCP 服务器非常适合与 [CherryStudio](https://github.com/kangfenmao/cherry-studio) 配合使用，提供稳定的网页搜索和抓取能力。

## 功能

- **web_search**: 使用 Brave 搜索引擎进行网页搜索
- **fetch_html**: 抓取网页 HTML 内容
- **fetch_text**: 抓取网页并提取纯文本
- **fetch_metadata**: 抓取网页元数据（标题、描述、链接）

## 安装

### 1. 克隆项目

```bash
git clone https://github.com/yourusername/WebSearchMCP.git
cd WebSearchMCP
```

### 2. 创建虚拟环境（推荐）

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### 3. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 4. 安装 Playwright 浏览器（项目本地）

浏览器会安装到项目目录下的 `.playwright-browsers`，不会污染系统目录。

```bash
# 设置浏览器安装路径为项目本地
export PLAYWRIGHT_BROWSERS_PATH="$(pwd)/.playwright-browsers"

# 安装 Chromium
python -m playwright install chromium
```

Windows PowerShell：

```powershell
$env:PLAYWRIGHT_BROWSERS_PATH = "$PWD\.playwright-browsers"
python -m playwright install chromium
```

### 5. 验证安装

```bash
python WebSearchMCP.py
```

如果看到 `Web Search MCP Server 启动中...` 说明安装成功。

## 使用

### 直接运行

```bash
python WebSearchMCP.py
```

### 使用代理

```bash
python WebSearchMCP.py --proxy http://127.0.0.1:7890
```

### 使用 Cloudflare Worker 代理

通过 Cloudflare Worker 转发请求，目标网站看到的 IP 为 Cloudflare 节点 IP：

```bash
python WebSearchMCP.py --cf-worker https://your-worker.workers.dev
```

可以同时使用本地代理和 CF Worker（本地代理用于连接 CF Worker）：

```bash
python WebSearchMCP.py --proxy http://127.0.0.1:7890 --cf-worker https://your-worker.workers.dev
```

### 配置 CherryStudio

在 CherryStudio 的 MCP 服务器设置中添加：

```json
{
  "mcpServers": {
    "web-search": {
      "command": "python",
      "args": ["/完整路径/WebSearchMCP.py"]
    }
  }
}
```

如需使用代理：

```json
{
  "mcpServers": {
    "web-search": {
      "command": "python",
      "args": ["/完整路径/WebSearchMCP.py", "--proxy", "http://127.0.0.1:7890"]
    }
  }
}
```

如需使用 Cloudflare Worker 代理：

```json
{
  "mcpServers": {
    "web-search": {
      "command": "python",
      "args": ["/完整路径/WebSearchMCP.py", "--cf-worker", "https://your-worker.workers.dev"]
    }
  }
}
```

同时使用本地代理和 CF Worker：

```json
{
  "mcpServers": {
    "web-search": {
      "command": "python",
      "args": [
        "/完整路径/WebSearchMCP.py",
        "--proxy", "http://127.0.0.1:7890",
        "--cf-worker", "https://your-worker.workers.dev"
      ]
    }
  }
}
```

## 命令行参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--proxy` | 本地代理地址 | `--proxy http://127.0.0.1:7890` |
| `--cf-worker` | Cloudflare Worker 地址 | `--cf-worker https://xxx.workers.dev` |

> **注意**: 本工具兼容 CherryStudio 等 MCP 客户端可能将 `--arg value` 合并为单个字符串的情况。

## 工具说明

### web_search

使用 Brave 搜索引擎搜索网页。

参数：
- `query`: 搜索关键词（必填）
- `max_results`: 返回的最大结果数，默认 10

### fetch_html

抓取网页的原始 HTML 内容。

参数：
- `url`: 目标网址（必填）
- `headers`: 可选的请求头

### fetch_text

抓取网页并提取纯文本内容（去除 HTML 标签）。

参数：
- `url`: 目标网址（必填）
- `headers`: 可选的请求头

### fetch_metadata

抓取网页的元数据信息。

参数：
- `url`: 目标网址（必填）
- `headers`: 可选的请求头

返回：标题、描述、页面链接列表

## 特性

- 持久化浏览器实例，提升搜索性能
- 支持本地代理配置 (`--proxy`)
- 支持 Cloudflare Worker 代理 (`--cf-worker`)，隐藏真实 IP
- 自动内容截断，防止响应过大
- 使用 cloudscraper 绕过基础反爬
- 兼容 CherryStudio 等 MCP 客户端的参数传递方式

## 常见问题

### Playwright 浏览器未安装

错误信息：`Playwright 未正确安装或启动失败`

解决方法：
```bash
playwright install chromium
```

### 代理连接失败

确保代理服务正在运行，并且端口正确。

### 搜索超时

可能是网络问题或 Brave 搜索页面结构变化，检查网络连接或更新选择器。

## 依赖

- Python 3.8+
- fastmcp
- beautifulsoup4
- playwright
- cloudscraper
- lxml
