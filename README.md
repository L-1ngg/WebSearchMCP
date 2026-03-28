![这是图片](./images/title.png)

<div align="center">

<!-- # Grok Search MCP -->

[English](./docs/README_EN.md) | 简体中文

**Grok-with-Tavily MCP，为 Claude Code 提供更完善的网络访问能力**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/) [![FastMCP](https://img.shields.io/badge/FastMCP-2.0.0+-green.svg)](https://github.com/jlowin/fastmcp)

</div>

---

## 一、概述

Grok Search MCP 是一个基于 [FastMCP](https://github.com/jlowin/fastmcp) 构建的 MCP 服务器，采用**双引擎架构**：**Grok** 负责 AI 驱动的智能搜索，**Tavily** 负责高保真网页抓取与站点映射，各取所长为 Claude Code / Cherry Studio 等LLM Client提供完整的实时网络访问能力。

## 项目来源

本仓库基于 [GuDaStudio/GrokSearch](https://github.com/GuDaStudio/GrokSearch) 进行修改与扩展，保留原项目的 MIT License 及版权声明。

当前仓库包含针对本地 `.env` 配置、多 Tavily API Key 轮询等功能的二次开发；新增功能与后续维护由当前仓库维护者负责，与原项目仓库的发布节奏和维护计划相互独立。

### 相对原仓库的新增功能

在保留原项目核心能力的基础上，当前仓库主要围绕配置读取与 Tavily 接入方式做了以下补充：

- **本地配置读取增强**：支持从项目根目录 `.env`、`~/.config/web-search/.env` 以及 `GROK_SEARCH_ENV_FILE` 指定的 env 文件读取配置，补充原有环境变量方式。
- **多 Tavily Key 支持**：支持通过 `TAVILY_API_KEYS` 配置多个 Tavily API Key，并在单个 Key 失败后按冷却时间自动轮换。
- **Tavily 调用统一封装**：将 Tavily 的 `search`、`extract`、`map` 调用统一收敛到客户端中，复用同一套 Key 选择、失败冷却与错误处理逻辑。
- **多 Key 场景兼容修正**：额外信源补充、网页抓取与站点映射等 Tavily 相关能力，改为基于多 Key 配置判断可用性，使 `TAVILY_API_KEYS` 场景下能够正常工作。
- **配置诊断信息补充**：`get_config_info` 会额外展示已加载的 env 文件列表以及 Tavily Key 数量，便于排查配置来源与多 Key 状态；默认不会主动联网，只有显式启用时才测试 Grok `/models` 连通性。

```
Claude ──MCP──► Grok Search Server
                  ├─ web_search  ───► Grok API（AI 搜索）
                  ├─ web_fetch   ───► Tavily Extract → Firecrawl Scrape（内容抓取，自动降级）
                  └─ web_map     ───► Tavily Map（站点映射）
```

### 功能特性

- **双引擎**：Grok 搜索 + Tavily 抓取/映射，互补协作
- **Firecrawl 托底**：Tavily 提取失败时自动降级到 Firecrawl Scrape，支持空内容自动重试
- **OpenAI 兼容接口**，支持任意 Grok 镜像站
- **自动时间注入**（检测时间相关查询，注入本地时间上下文）
- 一键禁用 Claude Code 官方 WebSearch/WebFetch，强制路由到本工具
- 智能重试（支持 Retry-After 头解析 + 指数退避）
- 父进程监控（Windows 下自动检测父进程退出，防止僵尸进程）

### 效果展示

我们以在`cherry studio`中配置本MCP为例，展示了`claude-opus-4.6`模型如何通过本项目实现外部知识搜集，降低幻觉率。
![](./images/wogrok.png)
如上图，**为公平实验，我们打开了claude模型内置的搜索工具**，然而opus 4.6仍然相信自己的内部常识，不查询FastAPI的官方文档，以获取最新示例。
![](./images/wgrok.png)
如上图，当打开`web-search MCP`时，在相同的实验条件下，opus 4.6主动调用多次搜索，以**获取官方文档，回答更可靠。**

### cherrystudio配置

参数设置：
```
--from
git+https://github.com/L-1ngg/WebSearchMCP
web-search
```
环境变量：根据需要配置,这里展示我的配置项
```
GROK_API_URL=
GROK_API_KEY=
GROK_MODEL=grok-4.20-beta
TAVILY_API_URL=https://api.tavily.com
TAVILY_API_KEYS=["tvly-key-1","tvly-key-2","tvly-key-3"]
TAVILY_ENABLED=true
GROK_DEBUG=false
GROK_LOG_LEVEL=INFO
```
这里没有配置 `TAVILY_API_KEYS` 的原因是我在.env文件里配置的，其余的配置项也可以写在.env文件里

## 二、安装

### 前置条件

- Python 3.10+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)（推荐的 Python 包管理器）
- Claude Code

<details>
<summary><b>安装 uv</b></summary>

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows PowerShell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

> Windows 用户**强烈推荐**在 WSL 中运行本项目。

</details>

### 一键安装

若之前安装过本项目，使用以下命令卸载旧版MCP。

```
claude mcp remove web-search
```

将以下命令中的环境变量替换为你自己的值后执行。Grok 接口需为 OpenAI 兼容格式；Tavily 为可选配置，未配置时工具 `web_fetch` 和 `web_map` 不可用。

<details> <summary>如果遇到 SSL / 证书验证错误</summary>

在部分企业网络或代理环境中，可能会出现类似错误：

certificate verify failed
self signed certificate in certificate chain

可以在 uvx 参数中添加 --native-tls，使其使用系统证书库：

claude mcp add-json web-search --scope user '{
  "type": "stdio",
  "command": "uvx",
  "args": [
    "--native-tls",
    "--from",
    "git+https://github.com/GuDaStudio/GrokSearch@grok-with-tavily",
    "web-search"
  ],
  "env": {
    "GROK_API_URL": "https://your-api-endpoint.com/v1",
    "GROK_API_KEY": "your-grok-api-key",
    "TAVILY_API_KEYS": ["tvly-your-tavily-key1", "tvly-your-tavily-key2"],
    "TAVILY_API_URL": "https://api.tavily.com"
  }
}'

</details> ```

也支持在本地 `.env` 中配置 Tavily。服务会按以下顺序读取配置：

1. MCP Client 显式传入的环境变量（如 Cherry Studio / Claude Code 的 `env`）
2. 项目根目录 `.env`
3. `~/.config/web-search/.env`

示例：

```env
TAVILY_API_URL=https://api.tavily.com
TAVILY_API_KEYS=["tvly-key-1","tvly-key-2","tvly-key-3"]
```

如果只配置单个 Key，也仍然兼容旧写法：

```env
TAVILY_API_URL=https://api.tavily.com
TAVILY_API_KEY=tvly-your-tavily-key
```

除此之外，你还可以在`env`字段中配置更多环境变量

| 变量 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `GROK_API_URL` | ✅ | - | Grok API 地址（OpenAI 兼容格式） |
| `GROK_API_KEY` | ✅ | - | Grok API 密钥 |
| `GROK_MODEL` | ❌ | `grok-4-fast` | 默认模型（设置后优先于 `~/.config/web-search/config.json`） |
| `TAVILY_API_KEY` | ❌ | - | 单个 Tavily API 密钥（兼容旧写法，用于 web_fetch / web_map） |
| `TAVILY_API_KEYS` | ❌ | - | 多个 Tavily API 密钥，使用 JSON 数组格式配置，按轮询顺序使用 |
| `TAVILY_API_URL` | ❌ | `https://api.tavily.com` | Tavily API 地址 |
| `TAVILY_ENABLED` | ❌ | `true` | 是否启用 Tavily |
| `TAVILY_KEY_COOLDOWN_SECONDS` | ❌ | `60` | 单个 Tavily Key 失败后的冷却秒数 |
| `FIRECRAWL_API_KEY` | ❌ | - | Firecrawl API 密钥（Tavily 失败时托底） |
| `FIRECRAWL_API_URL` | ❌ | `https://api.firecrawl.dev/v2` | Firecrawl API 地址 |
| `GROK_DEBUG` | ❌ | `false` | 调试模式 |
| `GROK_LOG_LEVEL` | ❌ | `INFO` | 日志级别 |
| `GROK_LOG_DIR` | ❌ | `logs` | 日志目录 |
| `GROK_RETRY_MAX_ATTEMPTS` | ❌ | `3` | 最大重试次数 |
| `GROK_RETRY_MULTIPLIER` | ❌ | `1` | 重试退避乘数 |
| `GROK_RETRY_MAX_WAIT` | ❌ | `10` | 重试最大等待秒数 |

### 验证安装

```bash
claude mcp list
```

🍟 显示连接成功后，我们**十分推荐**在 Claude 对话中输入

```
调用 web-search toggle_builtin_tools，关闭Claude Code's built-in WebSearch and WebFetch tools
```

工具将自动修改**项目级** `.claude/settings.json` 的 `permissions.deny`，一键禁用 Claude Code 官方的 WebSearch 和 WebFetch，从而迫使claude code调用本项目实现搜索！

## 三、MCP 工具介绍

<details>
<summary>本项目提供八个 MCP 工具（展开查看）</summary>

### `web_search` — AI 网络搜索

默认可直接调用 `web_search`。服务端会基于 query 自动选择内建的 bounded search strategy：简单问题偏 direct，复杂问题会在受控范围内做 breadth-first 式广度探索，再对关键分支做 depth-first 式深入搜索，最后返回可直接用于回答用户的正文，以及 `session_id` 供后续获取信源。

对于复杂搜索或上层 agent 需要预先规划的场景，也可以额外传入 `planning_session_id`。若提供了规划会话，服务端会尝试将其作为参考上下文使用；是否强制校验由 `planning_mode` 控制。

`web_search` 输出不展开信源，仅返回 `sources_count`；信源会按 `session_id` 缓存在服务端，可用 `get_sources` 拉取。

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `query` | string | ✅ | - | 搜索查询语句 |
| `planning_session_id` | string | ❌ | `""` | 可选的规划会话 ID；若提供，服务端会按 `planning_mode` 决定是否应用 |
| `planning_mode` | string | ❌ | `"auto"` | `auto` 尝试应用合法 planning，非法 planning 会忽略并继续搜索；`require` 强制 planning 合法；`ignore` 完全忽略 planning |
| `platform` | string | ❌ | `""` | 聚焦平台（如 `"Twitter"`, `"GitHub, Reddit"`） |
| `model` | string | ❌ | `null` | 按次指定 Grok 模型 ID |
| `search_prompt` | string | ❌ | `""` | 调用方自定义的搜索策略 Prompt，可控制搜索深度、信源偏好与回答风格；服务端仍保留内建安全护栏与固定格式子任务 Prompt |
| `source_preference` | string | ❌ | `"auto"` | 结构化信源偏好：`auto` / `official` / `community` / `news` / `academic` |
| `answer_style` | string | ❌ | `"auto"` | 结构化回答风格：`auto` / `concise` / `detailed` / `bullet_summary` |
| `search_depth` | string | ❌ | `"auto"` | 结构化搜索深度：`auto` / `direct` / `balanced` / `deep` |
| `extra_sources` | int | ❌ | `0` | 额外补充信源数量（Tavily/Firecrawl，可为 0 关闭） |

若上层调用方希望自己编排搜索主 Prompt，可传入 `search_prompt`；该参数只覆盖主搜索策略，不影响 `web_fetch` / `describe_url` / `rank_sources` 等底层固定任务 Prompt。若未传入，服务端会回退到默认的 bounded search strategy。

若不想自己写整段 Prompt，也可以只传结构化参数：

- `source_preference=official`：优先官方文档、厂商说明、第一方公告
- `answer_style=bullet_summary`：倾向输出简短要点列表
- `search_depth=deep`：倾向先广后深地做更充分搜索

自动检测查询中的时间相关关键词（如"最新""今天""recent"等），注入本地时间上下文以提升时效性搜索的准确度。

返回值（结构化字典）：

- `session_id`: 本次查询的会话 ID
- `content`: Grok 回答正文（已自动剥离信源）
- `sources_count`: 已缓存的信源数量
- `status`: `ok` / `error`
- `answer_ready`: 当前 `content` 是否可直接用于回答用户
- `used_custom_search_prompt`: 是否使用了调用方自定义 `search_prompt`
- `planning_applied`: 本次搜索是否实际应用了 planning 上下文
- `planning_status`: planning 的处理结果，例如 `not_provided` / `applied` / `ignored_*`
- `sources_preview`: 最多 3 条轻量信源预览
- `warnings`: 可选警告列表，例如在 `planning_mode=auto` 下忽略了无效 planning
- `error`: 仅在 `status=error` 时出现，包含错误码与是否建议原样重试

当 `status=error` 时，应将其视为该查询的终止结果，不要对同一查询原样重复调用，应改为向用户说明限制或先重写查询。

若 `planning_mode=auto` 且提供的 planning 无法通过校验，服务端会忽略该 planning 并继续执行默认搜索，同时在 `planning_status` / `warnings` 中说明原因；只有 `planning_mode=require` 时才会将 planning 错误视为终止条件。

### Advanced Planning Workflow

默认调用 `web_search` 时无需先做 planning。以下流程仅适用于复杂搜索、上层 agent 需要事先规划搜索路径，或调用方希望在 `planning_mode=require` 下显式约束搜索行为的场景。

推荐调用顺序如下：

1. 调用 `plan_intent`
   必须传入 `original_query`（原始用户问题）以及蒸馏后的 `core_question`
2. 调用 `plan_complexity`
   先确定复杂度等级，服务端据此决定后续必须完成哪些 phase
3. 按复杂度补齐剩余 phase
   - Level 1: 至少完成 `plan_sub_query`
   - Level 2: 还需完成 `plan_search_term`、`plan_tool_mapping`
   - Level 3: 还需完成 `plan_execution`
4. 调用 `web_search`
   传入原始 `query`、上一步得到的 `planning_session_id`，并根据需要设置 `planning_mode`

调用约束：
- `query` 必须与 `plan_intent.original_query` 严格绑定，旧 planning 不能复用到新 query
- `planning_mode=auto` 时，未通过校验的 planning 会被忽略并继续使用默认搜索策略
- `planning_mode=require` 时，规划不完整、query 不匹配或缺少绑定信息会直接报错

### `get_sources` — 获取信源

通过 `session_id` 获取对应 `web_search` 的全部缓存信源，用于校验或补充引用。

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `session_id` | string | ✅ | `web_search` 返回的 `session_id` |

返回值（结构化字典）：

- `session_id`
- `sources_count`
- `sources`: 信源列表（每项包含 `url`，可能包含 `title`/`description`/`provider`）

### `web_fetch` — 网页内容抓取

通过 Tavily Extract API 获取完整网页内容，返回 Markdown 格式。Tavily 失败时自动降级到 Firecrawl Scrape 进行托底抓取。

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `url` | string | ✅ | 目标网页 URL |

### `web_map` — 站点结构映射

通过 Tavily Map API 遍历网站结构，发现 URL 并生成站点地图。

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `url` | string | ✅ | - | 起始 URL |
| `instructions` | string | ❌ | `""` | 自然语言过滤指令 |
| `max_depth` | int | ❌ | `1` | 最大遍历深度（1-5） |
| `max_breadth` | int | ❌ | `20` | 每页最大跟踪链接数（1-500） |
| `limit` | int | ❌ | `50` | 总链接处理数上限（1-500） |
| `timeout` | int | ❌ | `150` | 超时秒数（10-150） |

### `get_config_info` — 配置诊断

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `include_connection_test` | bool | ❌ | `false` | 是否显式探测 Grok `/models` 端点；默认关闭，避免把“查看配置”变成依赖网络的操作 |

默认零参数调用 `get_config_info()` 仍然有效，会返回结构化对象并保留原有顶层诊断字段（如 `GROK_API_URL`、`GROK_MODEL`、`config_status`、`connection_test`），同时新增：

- `status`：整体结果，`ok` / `error`
- `config`：配置快照的嵌套对象副本，方便外部调用方稳定读取
- `error`：仅在配置快照采集失败时出现，说明失败原因

默认情况下 `connection_test.status` 为 `skipped`，不会主动请求网络。只有传入 `include_connection_test=true` 时，工具才会探测 Grok `/models` 端点，并返回响应时间与 `available_models`。

### `switch_model` — 模型切换

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model` | string | ✅ | 模型 ID（如 `"grok-4-fast"`, `"grok-2-latest"`） |

切换后配置持久化到 `~/.config/web-search/config.json`，跨会话保持。

### `toggle_builtin_tools` — 工具路由控制

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `action` | string | ❌ | `"status"` | `"on"` 禁用官方工具 / `"off"` 启用官方工具 / `"status"` 查看状态 |

修改项目级 `.claude/settings.json` 的 `permissions.deny`，一键禁用 Claude Code 官方的 WebSearch 和 WebFetch。

### `search_planning` — 搜索规划

结构化搜索规划脚手架（分阶段、多轮），用于在执行复杂搜索前先生成可执行的搜索计划。

</details>

## 四、常见问题

<details>
<summary>
Q: 必须同时配置 Grok 和 Tavily 吗？
</summary>
A: Grok（`GROK_API_URL` + `GROK_API_KEY`）为必填，提供核心搜索能力。Tavily 和 Firecrawl 均为可选：配置 Tavily 后 `web_fetch` 优先使用 Tavily Extract，失败时降级到 Firecrawl Scrape；两者均未配置时 `web_fetch` 将返回配置错误提示。`web_map` 依赖 Tavily。
</details>

<details>
<summary>
Q: Grok API 地址需要什么格式？
</summary>
A: 需要 OpenAI 兼容格式的 API 地址（支持 `/chat/completions` 和 `/models` 端点）。如使用官方 Grok，需通过兼容 OpenAI 格式的镜像站访问。
</details>

<details>
<summary>
Q: 如何验证配置？
</summary>
A: 在 Claude 对话中说"显示 web-search 配置信息"，默认会返回本地配置诊断而不主动联网。若需要显式验证 Grok API 连通性，请调用 `get_config_info(include_connection_test=true)`。
</details>

## 许可证

[MIT License](LICENSE)

---

**如果这个项目对您有帮助，请给个 Star！**
