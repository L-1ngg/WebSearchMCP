import inspect

import pytest

from web_search import server


def test_get_config_info_signature_defaults_connection_test_to_false():
    parameter = inspect.signature(server.get_config_info).parameters["include_connection_test"]

    assert parameter.default is False


def test_get_config_info_signature_accepts_optional_reason_for_compatibility():
    parameter = inspect.signature(server.get_config_info).parameters["reason"]

    assert parameter.default == ""


@pytest.mark.asyncio
async def test_get_config_info_returns_structured_error_when_config_snapshot_fails(monkeypatch):
    def raise_config_snapshot_error():
        raise RuntimeError("snapshot boom")

    monkeypatch.setattr(server.config, "get_config_info", raise_config_snapshot_error)

    result = await server.get_config_info()

    assert result["status"] == "error"
    assert result["config"] == {}
    assert result["config_status"].startswith("❌")
    assert result["connection_test"]["status"] == "skipped"
    assert result["error"]["code"] == "config_snapshot_failed"
    assert "snapshot boom" in result["error"]["message"]


@pytest.mark.asyncio
async def test_get_config_info_skips_connection_test_by_default(monkeypatch):
    monkeypatch.setattr(server.config, "get_config_info", lambda: {"config_status": "ok", "GROK_MODEL": "grok-4-fast"})

    async def should_not_run(api_url: str, api_key: str) -> list[str]:
        raise AssertionError("connection test should be skipped by default")

    monkeypatch.setattr(server, "_fetch_available_models", should_not_run)

    result = await server.get_config_info()

    assert result["status"] == "ok"
    assert result["config_status"] == "ok"
    assert result["GROK_MODEL"] == "grok-4-fast"
    assert result["config"]["config_status"] == "ok"
    assert result["connection_test"]["status"] == "skipped"
    assert "not run" in result["connection_test"]["message"].lower()


@pytest.mark.asyncio
async def test_get_config_info_ignores_optional_reason_argument(monkeypatch):
    monkeypatch.setattr(server.config, "get_config_info", lambda: {"config_status": "ok", "GROK_MODEL": "grok-4-fast"})

    result = await server.get_config_info(reason="Test connection and check why search is returning empty results.")

    assert result["status"] == "ok"
    assert result["GROK_MODEL"] == "grok-4-fast"
    assert result["connection_test"]["status"] == "skipped"


@pytest.mark.asyncio
async def test_get_config_info_runs_connection_test_when_explicitly_requested(monkeypatch):
    monkeypatch.setattr(server.config, "get_config_info", lambda: {"config_status": "ok", "GROK_MODEL": "grok-4-fast"})
    monkeypatch.setenv("GROK_API_URL", "https://example.com/v1")
    monkeypatch.setenv("GROK_API_KEY", "test-key")

    async def fake_fetch_available_models(api_url: str, api_key: str) -> list[str]:
        assert api_url == "https://example.com/v1"
        assert api_key == "test-key"
        return ["grok-4-fast", "grok-2-latest"]

    monkeypatch.setattr(server, "_fetch_available_models", fake_fetch_available_models)

    result = await server.get_config_info(include_connection_test=True)

    assert result["status"] == "ok"
    assert result["GROK_MODEL"] == "grok-4-fast"
    assert result["connection_test"]["status"] == "ok"
    assert result["connection_test"]["available_models"] == ["grok-4-fast", "grok-2-latest"]
    assert result["connection_test"]["message"] == "Connection test succeeded."


@pytest.mark.asyncio
async def test_get_config_info_reports_connection_test_failure_without_throwing(monkeypatch):
    monkeypatch.setattr(server.config, "get_config_info", lambda: {"config_status": "ok", "GROK_MODEL": "grok-4-fast"})
    monkeypatch.setenv("GROK_API_URL", "https://example.com/v1")
    monkeypatch.setenv("GROK_API_KEY", "test-key")

    async def fail_fetch_available_models(api_url: str, api_key: str) -> list[str]:
        raise RuntimeError("upstream unavailable")

    monkeypatch.setattr(server, "_fetch_available_models", fail_fetch_available_models)

    result = await server.get_config_info(include_connection_test=True)

    assert result["status"] == "ok"
    assert result["GROK_MODEL"] == "grok-4-fast"
    assert result["connection_test"]["status"] == "error"
    assert result["connection_test"]["available_models"] == []
    assert "upstream unavailable" in result["connection_test"]["message"]
