import pytest

from web_search import diagnostics, server


@pytest.mark.asyncio
async def test_get_doctor_info_reports_configuration_booleans_and_hides_available_models(monkeypatch):
    monkeypatch.setattr(
        diagnostics.config,
        "get_config_info",
        lambda: {
            "GROK_API_URL": "https://example.com/v1",
            "GROK_API_KEY": "test****key",
            "GROK_MODEL": "grok-4-fast",
            "TAVILY_API_KEY": "未配置",
            "FIRECRAWL_API_KEY": "fire****key",
            "ENV_FILES_LOADED": ["/tmp/.env"],
        },
    )

    async def fake_test_grok_connection() -> dict:
        return {
            "status": "ok",
            "message": "Connection test succeeded.",
            "response_time_ms": 12.5,
            "available_models": ["grok-4-fast", "grok-2-latest"],
        }

    monkeypatch.setattr(diagnostics, "test_grok_connection", fake_test_grok_connection)

    result = await diagnostics.get_doctor_info()

    assert result["configuration"] == {
        "has_grok_api_url": True,
        "has_grok_api_key": True,
        "has_grok_model": True,
        "has_tavily_api_key": False,
        "has_firecrawl_api_key": True,
        "has_env_files": True,
    }
    assert result["connection_test"] == {
        "status": "ok",
        "message": "Connection test succeeded.",
        "response_time_ms": 12.5,
    }
    assert "available_models" not in result["connection_test"]


@pytest.mark.asyncio
async def test_doctor_returns_recommended_next_step(monkeypatch):
    async def fake_get_doctor_info() -> dict:
        return {
            "configuration": {"has_grok_api_url": True},
            "connection_test": {"status": "ok", "message": "Connection test succeeded.", "response_time_ms": 9.1},
            "recommended_next_step": "Run get_config_info(include_connection_test=true) if you need the available model list.",
        }

    monkeypatch.setattr(server.diagnostics, "get_doctor_info", fake_get_doctor_info)

    result = await server.doctor()

    assert result["recommended_next_step"] == "Run get_config_info(include_connection_test=true) if you need the available model list."
    assert set(result) == {"configuration", "connection_test", "recommended_next_step"}
