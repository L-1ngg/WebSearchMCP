from __future__ import annotations

import re
import unicodedata
from hashlib import sha256
from typing import Literal

from .planning import PHASE_NAMES, PlanningSession, engine as planning_engine

_PHASE_TO_TOOL_NAME = {
    "intent_analysis": "plan_intent",
    "complexity_assessment": "plan_complexity",
    "query_decomposition": "plan_sub_query",
    "search_strategy": "plan_search_term",
    "tool_selection": "plan_tool_mapping",
    "execution_order": "plan_execution",
}


def _missing_required_phases(session: PlanningSession) -> list[str]:
    return [phase for phase in PHASE_NAMES if phase in session.required_phases() and phase not in session.phases]


def _sanitize_context_token(value: str, max_len: int = 64) -> str:
    cleaned = re.sub(r"[\x00-\x1f\x7f`]+", " ", value or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len].rstrip()
    return cleaned


def _sanitize_context_id(value: str) -> str:
    cleaned = _sanitize_context_token(value, max_len=32)
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "", cleaned)
    return cleaned


def _sanitize_search_term(value: str) -> str:
    cleaned = _sanitize_context_token(value, max_len=80)
    cleaned = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff\s\+\#\.\-\/:\?]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _build_planning_context_data(session: PlanningSession) -> dict:
    plan = session.build_executable_plan()
    data: dict = {"planning_session_id": session.session_id}
    if session.complexity_level is not None:
        data["complexity_level"] = session.complexity_level

    intent = plan.get("intent_analysis")
    if isinstance(intent, dict):
        intent_data: dict = {}
        query_type = (intent.get("query_type") or "").strip()
        time_sensitivity = (intent.get("time_sensitivity") or "").strip()
        if query_type:
            intent_data["query_type"] = query_type
        if time_sensitivity:
            intent_data["time_sensitivity"] = time_sensitivity
        if intent_data:
            data["intent"] = intent_data

    sub_queries = plan.get("query_decomposition")
    if isinstance(sub_queries, list) and sub_queries:
        prioritized_sub_queries: list[dict] = []
        for item in sub_queries[:5]:
            if not isinstance(item, dict):
                continue
            sub_id = _sanitize_context_id(item.get("id") or "")
            if sub_id:
                prioritized_sub_queries.append({"id": sub_id})
        if prioritized_sub_queries:
            data["sub_queries"] = prioritized_sub_queries

    strategy = plan.get("search_strategy")
    if isinstance(strategy, dict):
        strategy_data: dict = {}
        approach = (strategy.get("approach") or "").strip()
        if approach:
            strategy_data["approach"] = approach
        search_terms = strategy.get("search_terms")
        if isinstance(search_terms, list) and search_terms:
            planned_terms: list[dict] = []
            for item in search_terms[:6]:
                if not isinstance(item, dict):
                    continue
                term = _sanitize_search_term(item.get("term") or "")
                purpose = _sanitize_context_id(item.get("purpose") or "")
                round_no = item.get("round")
                if term and purpose:
                    planned_terms.append({"term": term, "purpose": purpose, "round": round_no})
            if planned_terms:
                strategy_data["search_terms"] = planned_terms
        if strategy_data:
            data["search_strategy"] = strategy_data

    tool_selection = plan.get("tool_selection")
    if isinstance(tool_selection, list) and tool_selection:
        tool_mapping: list[dict] = []
        for item in tool_selection[:5]:
            if not isinstance(item, dict):
                continue
            sub_query_id = _sanitize_context_id(item.get("sub_query_id") or "")
            tool = (item.get("tool") or "").strip()
            if sub_query_id and tool:
                tool_mapping.append({"sub_query_id": sub_query_id, "tool": tool})
        if tool_mapping:
            data["tool_selection"] = tool_mapping

    return data


def _normalize_query_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "").casefold()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    # Preserve semantically meaningful technical separators while tolerating
    # incidental whitespace around them, e.g. "gpt - 4.1" == "gpt-4.1".
    normalized = re.sub(r"(?<=\S)\s*([+#./:_-])\s*(?=\S)", r"\1", normalized)
    return normalized


def _query_fingerprint(text: str) -> str:
    normalized = _normalize_query_text(text)
    return sha256(normalized.encode("utf-8")).hexdigest() if normalized else ""


def _planning_matches_query(session: PlanningSession, query: str) -> bool:
    plan = session.build_executable_plan()
    intent = plan.get("intent_analysis")
    if not isinstance(intent, dict):
        return False

    query_fingerprint = _query_fingerprint(query)
    if not query_fingerprint:
        return False

    planning_fingerprint = (intent.get("query_fingerprint") or "").strip()
    if not planning_fingerprint:
        return False

    return query_fingerprint == planning_fingerprint


def _resolve_planning_context(
    planning_session_id: str,
    query: str,
    planning_mode: Literal["auto", "require", "ignore"],
) -> tuple[dict | None, str, list[str], str, str]:
    normalized = (planning_session_id or "").strip()
    warnings: list[str] = []

    if planning_mode == "ignore":
        if normalized:
            return None, "ignored_by_mode", warnings, "", ""
        return None, "not_provided", warnings, "", ""

    if not normalized:
        if planning_mode == "require":
            return None, "not_provided", warnings, "planning_required", "调用 web_search 前必须先调用 plan_intent，并传入 planning_session_id。"
        return None, "not_provided", warnings, "", ""

    session = planning_engine.get_session(normalized)
    if session is None:
        message = f"planning_session_id 不存在或已过期: {normalized}"
        if planning_mode == "require":
            return None, "ignored_invalid", warnings, "planning_session_not_found", message
        warnings.append(message)
        return None, "ignored_invalid", warnings, "", ""

    if "intent_analysis" not in session.phases:
        message = "规划会话缺少 intent_analysis，必须先完成 plan_intent。"
        if planning_mode == "require":
            return None, "ignored_incomplete", warnings, "planning_incomplete", message
        warnings.append(message)
        return None, "ignored_incomplete", warnings, "", ""

    if session.complexity_level is None or "complexity_assessment" not in session.phases:
        message = "规划会话缺少 complexity_assessment，必须先完成 plan_complexity。"
        if planning_mode == "require":
            return None, "ignored_incomplete", warnings, "planning_incomplete", message
        warnings.append(message)
        return None, "ignored_incomplete", warnings, "", ""

    missing = _missing_required_phases(session)
    if missing:
        missing_tools = [_PHASE_TO_TOOL_NAME.get(phase, phase) for phase in missing]
        message = "规划阶段未完成，缺少: " + ", ".join(missing_tools)
        if planning_mode == "require":
            return None, "ignored_incomplete", warnings, "planning_incomplete", message
        warnings.append(message)
        return None, "ignored_incomplete", warnings, "", ""

    plan = session.build_executable_plan()
    intent = plan.get("intent_analysis")
    if not isinstance(intent, dict) or not (intent.get("query_fingerprint") or "").strip():
        message = "规划会话缺少 query 绑定信息，请重新调用 plan_intent 并传入 original_query。"
        if planning_mode == "require":
            return None, "ignored_invalid", warnings, "planning_unbound", message
        warnings.append(message)
        return None, "ignored_invalid", warnings, "", ""

    if not _planning_matches_query(session, query):
        message = "planning_session_id 与当前 query 不匹配，请重新调用 plan_intent 生成新的规划会话。"
        if planning_mode == "require":
            return None, "ignored_mismatch", warnings, "planning_query_mismatch", message
        warnings.append(message)
        return None, "ignored_mismatch", warnings, "", ""

    return _build_planning_context_data(session), "applied", warnings, "", ""
