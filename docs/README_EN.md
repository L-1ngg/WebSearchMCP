![Image](../images/title.png)
<div align="center">

<!-- # Grok Search MCP -->

English | [简体中文](../README.md)

**Grok-with-Tavily MCP, providing enhanced web access for Claude Code**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/) [![FastMCP](https://img.shields.io/badge/FastMCP-2.0.0+-green.svg)](https://github.com/jlowin/fastmcp)

</div>

---

## 1. Overview

Grok Search MCP is an MCP server built on [FastMCP](https://github.com/jlowin/fastmcp), featuring a **dual-engine architecture**: **Grok** handles AI-driven intelligent search, while **Tavily** handles high-fidelity web content extraction and site mapping. Together they provide complete real-time web access for LLM clients such as Claude Code and Cherry Studio.

## Project Origin

This repository is based on [GuDaStudio/GrokSearch](https://github.com/GuDaStudio/GrokSearch), while preserving the original MIT License and copyright notice.

This fork includes secondary development around local `.env` configuration and multiple Tavily API key rotation. New features and ongoing maintenance are handled by the current repository maintainer, independent from the original repository's release cadence and maintenance plan.

### Additional Features Compared with the Upstream Repository

While keeping the original core capabilities, this fork mainly adds the following enhancements around configuration loading and Tavily integration:

- **Enhanced local config loading**: supports reading settings from the project root `.env`, `~/.config/web-search/.env`, and an env file specified via `GROK_SEARCH_ENV_FILE`, in addition to regular environment variables.
- **Multiple Tavily key support**: supports configuring multiple Tavily API keys through `TAVILY_API_KEYS`, with automatic rotation after a key enters cooldown on failure.
- **Unified Tavily client wrapper**: consolidates Tavily `search`, `extract`, and `map` calls behind one client so the same key selection, cooldown, and error-handling logic is reused.
- **Multi-key compatibility fixes**: Tavily-dependent features such as extra source retrieval, page fetch, and site map now determine availability from the multi-key configuration, so `TAVILY_API_KEYS` setups work correctly.
- **Expanded config diagnostics**: `get_config_info` also reports loaded env files and the Tavily key count, making configuration troubleshooting easier; it stays local by default and only probes Grok `/models` when explicitly requested.

```
Claude --MCP--> Grok Search Server
                  ├─ web_search  ---> Grok API (AI Search)
                  ├─ web_fetch   ---> Tavily Extract (Content Extraction)
                  └─ web_map     ---> Tavily Map (Site Mapping)
```

### Features

- **Dual Engine**: Grok search + Tavily extraction/mapping, complementary collaboration
- **OpenAI-compatible interface**, supports any Grok mirror endpoint
- **Automatic time injection** (detects time-related queries, injects local time context)
- One-click disable Claude Code's built-in WebSearch/WebFetch, force routing to this tool
- Smart retry (Retry-After header parsing + exponential backoff)
- Parent process monitoring (auto-detects parent process exit on Windows, prevents zombie processes)

### Demo

Using `cherry studio` with this MCP configured, here's how `claude-opus-4.6` leverages this project for external knowledge retrieval, reducing hallucination rates.

![](../images/wogrok.png)
As shown above, **for a fair experiment, we enabled Claude's built-in search tools**, yet Opus 4.6 still relied on its internal knowledge without consulting FastAPI's official documentation for the latest examples.

![](../images/wgrok.png)
As shown above, with `web-search MCP` enabled under the same experimental conditions, Opus 4.6 proactively made multiple search calls to **retrieve official documentation, producing more reliable answers.**


## 2. Installation

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (recommended Python package manager)
- Claude Code

<details>
<summary><b>Install uv</b></summary>

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows PowerShell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

> Windows users are **strongly recommended** to run this project in WSL.

</details>

### One-Click Install

If you have previously installed this project, remove the old MCP first:
```
claude mcp remove web-search
```

Replace the environment variables in the following command with your own values. The Grok endpoint must be OpenAI-compatible; Tavily is optional — `web_fetch` and `web_map` will be unavailable without it.

#### GuDa Users (Recommended)

GuDa users only need to set `GUDA_API_KEY` to access all services — API URLs are automatically derived:


#### Custom Configuration

To use your own API endpoints, configure each service separately:

```bash
claude mcp add-json web-search --scope user '{
  "type": "stdio",
  "command": "uvx",
  "args": [
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
```

You can also configure Tavily locally via `.env`. The server loads settings in this order:

1. Environment variables explicitly injected by the MCP client
2. Project root `.env`
3. `~/.config/web-search/.env`

Example:

```env
TAVILY_API_URL=https://api.tavily.com
TAVILY_API_KEYS=["tvly-key-1","tvly-key-2","tvly-key-3"]
```

The legacy single-key form still works:

```env
TAVILY_API_URL=https://api.tavily.com
TAVILY_API_KEY=tvly-your-tavily-key
```

You can also configure additional environment variables in the `env` field:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GUDA_API_KEY` | No | - | GuDa API key (auto-derives all service URLs and keys when set) |
| `GUDA_BASE_URL` | No | `https://code.guda.studio` | GuDa service base URL |
| `GROK_API_URL` | No | `{GUDA_BASE_URL}/grok/v1` | Grok API endpoint (OpenAI-compatible), overrides GuDa-derived value |
| `GROK_API_KEY` | No | `{GUDA_API_KEY}` | Grok API key, overrides GuDa-derived value |
| `GROK_MODEL` | No | `grok-4.20-beta` | Default model (takes precedence over `~/.config/web-search/config.json` when set) |
| `TAVILY_API_KEY` | No | `{GUDA_API_KEY}` | Tavily API key (for web_fetch / web_map) |
| `TAVILY_API_KEYS` | No | - | Multiple Tavily API keys in JSON array format, used in rotation |
| `TAVILY_API_URL` | No | `{GUDA_BASE_URL}/tavily` | Tavily API endpoint |
| `TAVILY_ENABLED` | No | `true` | Enable Tavily |
| `TAVILY_KEY_COOLDOWN_SECONDS` | No | `60` | Cooldown after a Tavily key fails |
| `FIRECRAWL_API_KEY` | No | `{GUDA_API_KEY}` | Firecrawl API key (fallback when Tavily fails) |
| `FIRECRAWL_API_URL` | No | `{GUDA_BASE_URL}/firecrawl` | Firecrawl API endpoint |
| `GROK_DEBUG` | No | `false` | Debug mode |
| `GROK_LOG_LEVEL` | No | `INFO` | Log level |
| `GROK_LOG_DIR` | No | `logs` | Log directory |
| `GROK_RETRY_MAX_ATTEMPTS` | No | `3` | Max retry attempts |
| `GROK_RETRY_MULTIPLIER` | No | `1` | Retry backoff multiplier |
| `GROK_RETRY_MAX_WAIT` | No | `10` | Max retry wait in seconds |

> **Note**: When `GUDA_API_KEY` is set, all `GROK_API_URL`/`GROK_API_KEY`/`TAVILY_*`/`FIRECRAWL_*` variables become optional as they are auto-derived from `GUDA_BASE_URL`. Explicitly set variables take higher priority.


### Verify Installation

```bash
claude mcp list
```

After confirming a successful connection, we **highly recommend** typing the following in a Claude conversation:
```
Call web-search toggle_builtin_tools to disable Claude Code's built-in WebSearch and WebFetch tools
```
This will automatically modify the **project-level** `.claude/settings.json` `permissions.deny`, disabling Claude Code's built-in WebSearch and WebFetch, forcing Claude Code to use this project for searches!



## 3. MCP Tools

<details>
<summary>This project provides eight MCP tools (click to expand)</summary>

### `web_search` — AI Web Search

By default, you can call `web_search` directly. The server chooses a bounded internal search strategy based on the query itself: simple queries stay direct, while more complex ones may use breadth-first exploration for recall and depth-first follow-up on the most important branches before returning answer text suitable for the user, plus a `session_id` for retrieving sources later.

For complex searches or higher-level agents that want explicit search planning, you can additionally provide `planning_session_id`. When a planning session is present, the server will try to use it as reference context; whether validation is enforced depends on `planning_mode`.

`web_search` does not expand sources in the response; it only returns `sources_count`. Sources are cached server-side by `session_id` and can be fetched with `get_sources`.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | Yes | - | Search query |
| `planning_session_id` | string | No | `""` | Optional planning session ID; when provided, the server decides whether to apply it based on `planning_mode` |
| `planning_mode` | string | No | `"auto"` | `auto` applies valid planning and ignores invalid planning with warnings; `require` enforces valid planning; `ignore` skips planning entirely |
| `platform` | string | No | `""` | Focus platform (e.g., `"Twitter"`, `"GitHub, Reddit"`) |
| `model` | string | No | `null` | Per-request Grok model ID |
| `search_prompt` | string | No | `""` | Caller-authored search strategy prompt for search depth, source preference, and answer style. Server-side guardrails and fixed-format helper prompts remain enforced |
| `source_preference` | string | No | `"auto"` | Structured source preference: `auto` / `official` / `community` / `news` / `academic` |
| `answer_style` | string | No | `"auto"` | Structured answer style: `auto` / `concise` / `detailed` / `bullet_summary` |
| `search_depth` | string | No | `"auto"` | Structured search depth: `auto` / `direct` / `balanced` / `deep` |
| `extra_sources` | int | No | `0` | Extra sources via Tavily/Firecrawl (0 disables) |

If the calling agent wants to author its own main search prompt, pass `search_prompt`. This only overrides the main search strategy and does not affect fixed lower-level prompts such as `web_fetch`, `describe_url`, or `rank_sources`. If omitted, the server falls back to its default bounded search strategy.

If you do not want to write a full prompt, you can steer behavior with structured controls instead:

- `source_preference=official`: prefer first-party docs, vendor references, and official announcements
- `answer_style=bullet_summary`: bias toward short bullet-led answers
- `search_depth=deep`: bias toward broader exploration before targeted drill-down

Automatically detects time-related keywords in queries (e.g., "latest", "today", "recent"), injecting local time context to improve accuracy for time-sensitive searches.

Return value (structured dict):
- `session_id`: search session ID
- `content`: answer only (sources removed)
- `sources_count`: cached sources count
- `status`: `ok` / `error`
- `answer_ready`: whether `content` is suitable for directly answering the user
- `used_custom_search_prompt`: whether a caller-authored `search_prompt` was used
- `planning_applied`: whether planning context was actually applied
- `planning_status`: planning handling status, such as `not_provided` / `applied` / `ignored_*`
- `sources_preview`: up to 3 lightweight cached source previews
- `warnings`: optional warning list, for example when invalid planning was ignored in `planning_mode=auto`
- `error`: present only when `status=error`, including an error code and whether retrying the exact same query is advised

When `status=error`, treat it as a terminal outcome for that exact query. Do not repeat the same query verbatim; either explain the limitation or refine the query first.

If `planning_mode=auto` and the supplied planning fails validation, the server ignores the planning and continues with its default search strategy while reporting the reason in `planning_status` / `warnings`. Only `planning_mode=require` turns planning validation failures into terminal errors.

### Advanced Planning Workflow

You do not need planning for normal `web_search` calls. The workflow below is only for complex searches, pre-planned agent flows, or callers that want strict planning enforcement with `planning_mode=require`.

The recommended call sequence is:

1. Call `plan_intent`
   You must provide `original_query` (the raw user request) and the distilled `core_question`
2. Call `plan_complexity`
   This determines the complexity level, which in turn decides which later phases are required
3. Complete the remaining required phases based on complexity
   - Level 1: at least `plan_sub_query`
   - Level 2: also requires `plan_search_term` and `plan_tool_mapping`
   - Level 3: also requires `plan_execution`
4. Call `web_search`
   Pass the original `query`, the resulting `planning_session_id`, and set `planning_mode` as needed

Execution constraints:
- `query` must remain strictly bound to `plan_intent.original_query`; an old plan cannot be reused for a different query
- In `planning_mode=auto`, invalid planning is ignored and the default search path continues
- In `planning_mode=require`, incomplete plans, mismatched queries, or missing binding data fail fast

### `get_sources` — Retrieve Sources

Retrieves the full cached source list for a previous `web_search` call, typically for verification or citations.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `session_id` | string | Yes | `session_id` returned by `web_search` |

Return value (structured dict):
- `session_id`
- `sources_count`
- `sources`: source list (each item includes `url`, may include `title`/`description`/`provider`)

### `web_fetch` — Web Content Extraction

Extracts complete web content via Tavily Extract API, returning Markdown format.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `url` | string | Yes | Target webpage URL |

### `web_map` — Site Structure Mapping

Traverses website structure via Tavily Map API, discovering URLs and generating a site map.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `url` | string | Yes | - | Starting URL |
| `instructions` | string | No | `""` | Natural language filtering instructions |
| `max_depth` | int | No | `1` | Max traversal depth (1-5) |
| `max_breadth` | int | No | `20` | Max links to follow per page (1-500) |
| `limit` | int | No | `50` | Total link processing limit (1-500) |
| `timeout` | int | No | `150` | Timeout in seconds (10-150) |

### `get_config_info` — Configuration Diagnostics

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `include_connection_test` | bool | No | `false` | Explicitly probe the Grok `/models` endpoint. Disabled by default so config inspection does not depend on network availability |

Legacy no-argument `get_config_info()` calls remain valid. The tool now returns a structured object while preserving the previous top-level diagnostic fields such as `GROK_API_URL`, `GROK_MODEL`, `config_status`, and `connection_test`. It also adds:

- `status`: overall result, `ok` / `error`
- `config`: nested copy of the configuration snapshot for stable machine consumption
- `error`: present only when configuration snapshot gathering fails

By default, `connection_test.status` is `skipped` and no network call is made. Pass `include_connection_test=true` to run the Grok `/models` probe and receive response timing plus `available_models`.

### `switch_model` — Model Switching

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | Yes | Model ID (e.g., `"grok-4-fast"`, `"grok-2-latest"`) |

Settings persist to `~/.config/web-search/config.json` across sessions.

### `toggle_builtin_tools` — Tool Routing Control

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string | No | `"status"` | `"on"` disable built-in tools / `"off"` enable built-in tools / `"status"` check status |

Modifies project-level `.claude/settings.json` `permissions.deny` to disable Claude Code's built-in WebSearch and WebFetch.

### `search_planning` — Search Planning

A structured multi-phase planning scaffold to generate an executable search plan before running complex searches.
</details>

## 4. FAQ

<details>
<summary>
Q: Must I configure both Grok and Tavily?
</summary>
A: Set `GUDA_API_KEY` to get full Grok + Tavily + Firecrawl service. Without GuDa, Grok (`GROK_API_URL` + `GROK_API_KEY`) is required and provides the core search capability. Tavily is optional — without it, `web_fetch` and `web_map` will return configuration error messages.
</details>

<details>
<summary>
Q: What format does the Grok API URL need?
</summary>
A: An OpenAI-compatible API endpoint (supporting `/chat/completions` and `/models` endpoints). If using official Grok, access it through an OpenAI-compatible mirror.
</details>

<details>
<summary>
Q: How to verify configuration?
</summary>
A: Say "Show web-search configuration info" in a Claude conversation to inspect local diagnostics without making a network call. If you want to explicitly validate Grok API connectivity, call `get_config_info(include_connection_test=true)`.
</details>

## License

[MIT License](LICENSE)

---

<div align="center">

**If this project helps you, please give it a Star!**

[![Star History Chart](https://api.star-history.com/svg?repos=GuDaStudio/GrokSearch&type=date&legend=top-left)](https://www.star-history.com/#GuDaStudio/GrokSearch&type=date&legend=top-left)
</div>
