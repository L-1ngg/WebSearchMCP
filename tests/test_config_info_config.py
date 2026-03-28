import json
from unittest.mock import mock_open
from pathlib import Path

from web_search import config as config_module


def _new_config(monkeypatch):
    home = Path("D:/virtual-home")
    cwd = Path("D:/virtual-cwd")

    monkeypatch.setattr(config_module.Path, "home", lambda: home)
    monkeypatch.setattr(config_module.Path, "cwd", lambda: cwd)
    config_module.Config._instance = None

    return config_module.Config(), home, cwd


def test_get_config_info_returns_snapshot_when_config_dirs_cannot_be_created(
    monkeypatch,
):
    config, home, cwd = _new_config(monkeypatch)
    tmp_dir = Path("D:/virtual-tmp")

    monkeypatch.setenv("GROK_API_URL", "https://example.com/v1")
    monkeypatch.setenv("GROK_API_KEY", "test-key-12345678")
    monkeypatch.setattr(config_module.tempfile, "gettempdir", lambda: str(tmp_dir))

    original_mkdir = Path.mkdir

    def blocked_mkdir(self, mode=0o777, parents=False, exist_ok=False):
        if self.is_relative_to(home) or self.is_relative_to(cwd) or self.is_relative_to(
            tmp_dir
        ):
            raise OSError("read-only filesystem")
        return original_mkdir(self, mode=mode, parents=parents, exist_ok=exist_ok)

    monkeypatch.setattr(config_module.Path, "mkdir", blocked_mkdir)

    info = config.get_config_info()

    assert info["GROK_API_URL"] == "https://example.com/v1"
    assert info["GROK_API_KEY"] == config_module.Config._mask_api_key(
        "test-key-12345678"
    )
    assert info["GROK_MODEL"] == config_module.Config._DEFAULT_MODEL
    assert info["config_status"] == "✅ 配置完整"


def test_get_config_info_returns_snapshot_when_config_paths_cannot_be_accessed(
    monkeypatch,
):
    config, home, cwd = _new_config(monkeypatch)
    config._config_file = home / ".config" / "web-search" / "config.json"

    monkeypatch.setenv("GROK_API_URL", "https://example.com/v1")
    monkeypatch.setenv("GROK_API_KEY", "test-key-12345678")

    original_exists = Path.exists

    def broken_exists(self):
        if self in {
            config._config_file,
            config._config_file.parent / ".env",
            cwd / ".env",
        }:
            raise OSError("cannot access path")
        return original_exists(self)

    monkeypatch.setattr(config_module.Path, "exists", broken_exists)

    info = config.get_config_info()

    assert info["GROK_API_URL"] == "https://example.com/v1"
    assert info["GROK_MODEL"] == config_module.Config._DEFAULT_MODEL
    assert info["ENV_FILES_LOADED"] == []


def test_set_model_keeps_directory_creation_for_writes(monkeypatch):
    config, home, cwd = _new_config(monkeypatch)
    mkdir_calls = []
    open_mock = mock_open()

    def home_fails_once(self, mode=0o777, parents=False, exist_ok=False):
        mkdir_calls.append(self)
        if self == home / ".config" / "web-search":
            raise OSError("home config unavailable")
        return None

    monkeypatch.setattr(config_module.Path, "mkdir", home_fails_once)
    monkeypatch.setattr(config_module.Path, "exists", lambda self: False)
    monkeypatch.setattr("builtins.open", open_mock)

    config.set_model("grok-2-latest")

    assert mkdir_calls == [home / ".config" / "web-search", cwd / ".web-search"]
    open_mock.assert_called_once_with(
        cwd / ".web-search" / "config.json", "w", encoding="utf-8"
    )
    written = "".join(call.args[0] for call in open_mock().write.call_args_list)
    assert json.loads(written) == {"model": "grok-2-latest"}
