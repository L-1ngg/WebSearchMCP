# Phase 1 Compatibility and Wave 1 Refactor Status

This document supersedes:

- `docs/superpowers/plans/2026-04-21-phase-1-compatibility-and-diagnostics.md`
- `docs/superpowers/plans/2026-04-22-server-wave1-refactor.md`

It records the current audited status of the Phase 1 compatibility work and the
Server Wave 1 refactor work as of 2026-04-29.

## Summary

- Phase 1 compatibility and diagnostics work is functionally complete.
- Server Wave 1 refactor work is complete.
- The test suite is green at the time of this audit: `uv run pytest -q` reported
  `61 passed in 1.35s`.
- The old implementation-plan checkboxes were not maintained during execution;
  this merged status document is the source of truth for completion tracking.

## Validation Evidence

Last verified command:

```bash
uv run pytest -q
```

Observed result:

```text
61 passed in 1.35s
```

Repository state at audit time:

```bash
git status --short
```

Observed result: no output before this documentation update.

## Phase 1 Compatibility and Diagnostics

Goal: add stable `search`, `fetch`, `map`, and `doctor` tools; add optional
cached-source pagination; preserve existing `web_*` and `get_config_info`
compatibility.

### Task Status

- [x] Task 1: Add `diagnostics.py` and the `doctor` MCP tool.
- [x] Task 2: Add optional cached-source pagination without breaking old callers.
- [x] Task 3: Add structured `fetch` and `map` aliases with bounded output.
- [x] Task 4: Add a stable `search` alias and document the Phase 1 migration path.
- [x] Task 5: Run the focused regression matrix and prepare release notes.

### Confirmed Artifacts

- `src/web_search/diagnostics.py`
  - Provides compact doctor diagnostics.
  - Hides `available_models` from the `doctor` response.
  - Provides `recommended_next_step`.
- `src/web_search/server.py`
  - Registers `doctor`.
  - Registers stable `search`, `fetch`, and `map` aliases.
  - Keeps legacy `web_search`, `web_fetch`, `web_map`, and `get_config_info`.
- `src/web_search/sources.py`
  - Adds `SourcesCache.page(...)`.
  - Adds `build_get_sources_response(...)`.
- `tests/test_doctor_tool.py`
  - Covers doctor response shape and model-list hiding.
- `tests/test_core_tool_aliases.py`
  - Covers `search`, `fetch`, and `map` alias behavior.
- `tests/test_sources_pagination.py`
  - Covers paginated and legacy full-list `get_sources` behavior.
- `README.md`
  - Documents Stable Core Tools and Phase 1 release notes.
- `docs/README_EN.md`
  - Documents the English Stable Core Tools migration table.

### Notes

- The implemented `doctor` payload uses boolean `has_*` configuration fields
  derived from `config.get_config_info()`, rather than the exact field names from
  the original implementation plan. Tests cover the implemented contract.
- The implemented `fetch` alias uses `_FETCH_STATUS` to distinguish extraction
  failures from page text that happens to begin with an error-looking prefix.

## Server Wave 1 Refactor

Goal: extract the first low-risk group of helper code from
`src/web_search/server.py` while preserving the MCP tool surface, return schemas,
and server-level monkeypatch targets used by tests.

### Task Status

- [x] Task 1: Extract planning adapter helpers.
- [x] Task 2: Extract retrieval helpers and retrieval-owned state.
- [x] Task 3: Move `build_get_sources_response()` into `sources.py`.
- [x] Task 4: Run Wave 1 regression and stop.

### Task 2 Status Detail

Task 2 is complete against the original acceptance criteria.

Confirmed complete:

- `src/web_search/retrieval_service.py` exists.
- Retrieval-owned state is in `retrieval_service.py`:
  - `_TAVILY_CLIENT`
  - `_TAVILY_CLIENT_FINGERPRINT`
  - `_FETCH_STATUS`
- Retrieval helpers are present in `retrieval_service.py`:
  - `_get_tavily_client`
  - `_call_tavily_extract`
  - `_call_tavily_search`
  - `_call_firecrawl_search`
  - `_call_firecrawl_scrape`
  - `_truncate_content`
  - `_build_tavily_map_payload`
  - `_call_tavily_map_structured`
  - `_call_tavily_map`
- `tests/test_retrieval_service.py` covers the extracted module.
- `src/web_search/server.py` imports `_call_tavily_map` from
  `retrieval_service.py` in both import branches.
- `src/web_search/server.py` no longer defines a local `_call_tavily_map(...)`
  wrapper.
- `tests/test_retrieval_service.py` includes a structural regression check that
  `server._call_tavily_map is retrieval_service._call_tavily_map`.
- `tests/test_core_tool_aliases.py` confirms existing server-level monkeypatch
  points still work for compatibility alias tests.

### Confirmed Artifacts

- `src/web_search/planning_adapter.py`
  - Owns planning helper functions and `_PHASE_TO_TOOL_NAME`.
  - `server.py` imports the helpers back under the same names.
- `src/web_search/retrieval_service.py`
  - Owns retrieval state and retrieval helpers.
- `src/web_search/sources.py`
  - Owns `build_get_sources_response(...)`.
- `tests/test_planning_adapter.py`
  - Covers query fingerprinting and missing planning phases.
- `tests/test_retrieval_service.py`
  - Covers retrieval status state, truncation metadata, and map payload shaping.
- `tests/test_sources_pagination.py`
  - Covers the moved `build_get_sources_response(...)`.

## Current Completion Conclusion

- Phase 1 compatibility and diagnostics: complete.
- Server Wave 1 refactor: complete.

## Historical Implementation Commits

Relevant commits already present in history:

- `0463ab0 feat(compat): add stable phase-1 core tools`
- `2e4b0ba refactor(server): extract planning adapter helpers`
- `f44acca refactor(server): extract retrieval helpers`
- `3655655 refactor(sources): move get_sources response builder`
