from __future__ import annotations

import time

import httpx

from .config import config


def _display_env_files(env_files: list[str] | None = None) -> str:
    files = env_files if env_files is not None else config.get_config_info().get("ENV_FILES_LOADED", [])
    normalized = [str(path).strip() for path in files if str(path).strip()]
    return ", ".join(normalized) if normalized else "none"


async def fetch_available_models(api_url: str, api_key: str) -> list[str]:
    models_url = f"{api_url.rstrip('/')}/models"
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(
            models_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        response.raise_for_status()
        data = response.json()

    models: list[str] = []
    for item in (data or {}).get("data", []) or []:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            models.append(item["id"])
    return models


async def test_grok_connection() -> dict:
    try:
        api_url = config.grok_api_url
        api_key = config.grok_api_key

        start_time = time.perf_counter()
        available_models = await fetch_available_models(api_url, api_key)
        response_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return {
            "status": "ok",
            "message": "Connection test succeeded.",
            "response_time_ms": response_time_ms,
            "available_models": available_models,
        }
    except httpx.TimeoutException:
        return {
            "status": "error",
            "message": "Connection test timed out after 10 seconds.",
            "response_time_ms": None,
            "available_models": [],
        }
    except httpx.RequestError as exc:
        return {
            "status": "error",
            "message": f"Connection test failed: {str(exc)}",
            "response_time_ms": None,
            "available_models": [],
        }
    except ValueError as exc:
        return {
            "status": "error",
            "message": f"Connection test could not start: {str(exc)}",
            "response_time_ms": None,
            "available_models": [],
        }
    except Exception as exc:
        return {
            "status": "error",
            "message": f"Connection test failed unexpectedly: {str(exc)}",
            "response_time_ms": None,
            "available_models": [],
        }


def _has_value(value: object) -> bool:
    if isinstance(value, str):
        return bool(value.strip()) and value.strip() != "未配置"
    if isinstance(value, list):
        return bool(value)
    return bool(value)


def _recommended_next_step(configuration: dict, connection_test: dict, env_files_display: str) -> str:
    if not configuration["has_grok_api_url"] or not configuration["has_grok_api_key"]:
        if configuration["has_env_files"]:
            return f"Update GROK_API_URL and GROK_API_KEY in your loaded env files ({env_files_display}), then rerun doctor."
        return "Set GROK_API_URL and GROK_API_KEY, then rerun doctor."

    if connection_test.get("status") != "ok":
        return "Verify GROK_API_URL, GROK_API_KEY, and network reachability, then rerun doctor."

    return "Run get_config_info(include_connection_test=true) if you need the available model list."


async def get_doctor_info() -> dict:
    try:
        config_snapshot = config.get_config_info()
    except Exception:
        config_snapshot = {}

    env_files = config_snapshot.get("ENV_FILES_LOADED", [])
    env_files_display = _display_env_files(env_files if isinstance(env_files, list) else None)

    configuration = {
        "has_grok_api_url": _has_value(config_snapshot.get("GROK_API_URL")),
        "has_grok_api_key": _has_value(config_snapshot.get("GROK_API_KEY")),
        "has_grok_model": _has_value(config_snapshot.get("GROK_MODEL")),
        "has_tavily_api_key": _has_value(config_snapshot.get("TAVILY_API_KEY")),
        "has_firecrawl_api_key": _has_value(config_snapshot.get("FIRECRAWL_API_KEY")),
        "has_env_files": env_files_display != "none",
    }

    raw_connection_test = await test_grok_connection()
    connection_test = {
        "status": raw_connection_test.get("status", "error"),
        "message": raw_connection_test.get("message", "Connection test failed unexpectedly."),
        "response_time_ms": raw_connection_test.get("response_time_ms"),
    }

    return {
        "configuration": configuration,
        "connection_test": connection_test,
        "recommended_next_step": _recommended_next_step(configuration, connection_test, env_files_display),
    }
