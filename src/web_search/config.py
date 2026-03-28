from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


class Config:
    _instance = None
    _SETUP_COMMAND = (
        "claude mcp add-json web-search --scope user "
        '\'{"type":"stdio","command":"uvx","args":["--from",'
        '"git+https://github.com/GuDaStudio/GrokSearch","grok-search"],'
        '"env":{"GROK_API_URL":"your-api-url","GROK_API_KEY":"your-api-key"}}\''
    )
    _DEFAULT_MODEL = "grok-4-fast"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._config_file = None
            cls._instance._cached_model = None
            cls._instance._env_file_cache = None
        return cls._instance

    def _config_dirs(self) -> tuple[Path, Path]:
        return Path.home() / ".config" / "web-search", Path.cwd() / ".web-search"

    @staticmethod
    def _path_exists(path: Path) -> bool:
        try:
            return path.exists()
        except OSError:
            return False

    @staticmethod
    def _path_is_file(path: Path) -> bool:
        try:
            return path.is_file()
        except OSError:
            return False

    def _resolve_config_file(self, *, create_dirs: bool) -> Path:
        home_dir, cwd_dir = self._config_dirs()
        home_config_file = home_dir / "config.json"
        cwd_config_file = cwd_dir / "config.json"

        if create_dirs:
            if self._config_file is None:
                try:
                    home_dir.mkdir(parents=True, exist_ok=True)
                    self._config_file = home_config_file
                except OSError:
                    cwd_dir.mkdir(parents=True, exist_ok=True)
                    self._config_file = cwd_config_file
            return self._config_file

        if self._config_file is not None:
            return self._config_file
        if self._path_exists(home_config_file):
            return home_config_file
        if self._path_exists(cwd_config_file):
            return cwd_config_file
        return home_config_file

    @property
    def config_file(self) -> Path:
        return self._resolve_config_file(create_dirs=True)

    def _load_config_file(self) -> dict:
        config_file = self._resolve_config_file(create_dirs=False)
        if not self._path_exists(config_file):
            return {}
        try:
            with open(config_file, "r", encoding="utf-8") as file:
                return json.load(file)
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_config_file(self, config_data: dict) -> None:
        try:
            with open(self.config_file, "w", encoding="utf-8") as file:
                json.dump(config_data, file, ensure_ascii=False, indent=2)
        except IOError as exc:
            raise ValueError(f"无法保存配置文件: {str(exc)}")

    def _iter_env_files(self) -> list[Path]:
        candidates: list[Path] = [
            self._resolve_config_file(create_dirs=False).parent / ".env",
            Path.cwd() / ".env",
        ]
        # 支持新旧两种环境变量名，保持向后兼容
        explicit = os.getenv("WEB_SEARCH_ENV_FILE") or os.getenv(
            "GROK_SEARCH_ENV_FILE", ""
        )
        explicit = explicit.strip()
        if explicit:
            candidates.append(Path(explicit).expanduser())

        deduplicated: list[Path] = []
        seen: set[Path] = set()
        for candidate in candidates:
            resolved = candidate.resolve(strict=False)
            if resolved in seen:
                continue
            seen.add(resolved)
            deduplicated.append(candidate)
        return deduplicated

    @staticmethod
    def _parse_env_file(path: Path) -> dict[str, str]:
        if not Config._path_exists(path) or not Config._path_is_file(path):
            return {}

        env_data: dict[str, str] = {}
        try:
            with open(path, "r", encoding="utf-8") as file:
                for raw_line in file:
                    line = raw_line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith("export "):
                        line = line[7:].strip()
                    if "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    if not key:
                        continue
                    if (
                        len(value) >= 2
                        and value[0] == value[-1]
                        and value[0] in ('"', "'")
                    ):
                        value = value[1:-1]
                    env_data[key] = value
        except OSError:
            return {}

        return env_data

    def _load_env_file_values(self) -> dict[str, str]:
        if self._env_file_cache is not None:
            return self._env_file_cache

        merged: dict[str, str] = {}
        for path in self._iter_env_files():
            merged.update(self._parse_env_file(path))
        self._env_file_cache = merged
        return merged

    def _get_setting(self, name: str, default: str | None = None) -> str | None:
        value = os.getenv(name)
        if value not in (None, ""):
            return value
        return self._load_env_file_values().get(name, default)

    @staticmethod
    def _parse_json_array_value(raw_value: str | None) -> list[str]:
        if not raw_value:
            return []

        raw_value = raw_value.strip()
        if not raw_value:
            return []

        try:
            parsed = json.loads(raw_value)
        except json.JSONDecodeError:
            return []

        if not isinstance(parsed, list):
            return []

        return [str(item).strip() for item in parsed if str(item).strip()]

    @property
    def debug_enabled(self) -> bool:
        return (self._get_setting("GROK_DEBUG", "false") or "false").lower() in (
            "true",
            "1",
            "yes",
        )

    @staticmethod
    def _safe_int(value: str | None, default: int) -> int:
        try:
            return int(value) if value else default
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _safe_float(value: str | None, default: float) -> float:
        try:
            return float(value) if value else default
        except (ValueError, TypeError):
            return default

    @property
    def retry_max_attempts(self) -> int:
        return self._safe_int(self._get_setting("GROK_RETRY_MAX_ATTEMPTS", "3"), 3)

    @property
    def retry_multiplier(self) -> float:
        return self._safe_float(self._get_setting("GROK_RETRY_MULTIPLIER", "1"), 1.0)

    @property
    def retry_max_wait(self) -> int:
        return self._safe_int(self._get_setting("GROK_RETRY_MAX_WAIT", "10"), 10)

    @property
    def tavily_key_cooldown_seconds(self) -> int:
        return self._safe_int(
            self._get_setting("TAVILY_KEY_COOLDOWN_SECONDS", "60"), 60
        )

    @property
    def grok_api_url(self) -> str:
        url = self._get_setting("GROK_API_URL")
        if not url:
            raise ValueError(
                f"Grok API URL 未配置！\n"
                f"请使用以下命令配置 MCP 服务：\n{self._SETUP_COMMAND}"
            )
        return url

    @property
    def grok_api_key(self) -> str:
        key = self._get_setting("GROK_API_KEY")
        if not key:
            raise ValueError(
                f"Grok API Key 未配置！\n"
                f"请使用以下命令配置 MCP 服务：\n{self._SETUP_COMMAND}"
            )
        return key

    @property
    def tavily_enabled(self) -> bool:
        return (self._get_setting("TAVILY_ENABLED", "true") or "true").lower() in (
            "true",
            "1",
            "yes",
        )

    @property
    def tavily_api_url(self) -> str:
        return (
            self._get_setting("TAVILY_API_URL", "https://api.tavily.com")
            or "https://api.tavily.com"
        )

    @property
    def tavily_api_keys(self) -> list[str]:
        raw_multi_keys = self._get_setting("TAVILY_API_KEYS")
        if raw_multi_keys not in (None, ""):
            return self._parse_json_array_value(raw_multi_keys)

        single_key = (self._get_setting("TAVILY_API_KEY") or "").strip()
        return [single_key] if single_key else []

    @property
    def tavily_api_key(self) -> str | None:
        keys = self.tavily_api_keys
        return keys[0] if keys else None

    @property
    def firecrawl_api_url(self) -> str:
        return (
            self._get_setting("FIRECRAWL_API_URL", "https://api.firecrawl.dev/v2")
            or "https://api.firecrawl.dev/v2"
        )

    @property
    def firecrawl_api_key(self) -> str | None:
        return self._get_setting("FIRECRAWL_API_KEY")

    @property
    def log_level(self) -> str:
        return (self._get_setting("GROK_LOG_LEVEL", "INFO") or "INFO").upper()

    @property
    def log_dir(self) -> Path:
        return self._resolve_log_dir(create_dirs=True)

    def _resolve_log_dir(self, *, create_dirs: bool) -> Path:
        log_dir_str = self._get_setting("GROK_LOG_DIR", "logs") or "logs"
        log_dir = Path(log_dir_str)
        if log_dir.is_absolute():
            return log_dir

        home_log_dir = Path.home() / ".config" / "web-search" / log_dir_str
        if not create_dirs and self._path_exists(home_log_dir):
            return home_log_dir
        if create_dirs:
            try:
                home_log_dir.mkdir(parents=True, exist_ok=True)
                return home_log_dir
            except OSError:
                pass

        cwd_log_dir = Path.cwd() / log_dir_str
        if not create_dirs and self._path_exists(cwd_log_dir):
            return cwd_log_dir
        if create_dirs:
            try:
                cwd_log_dir.mkdir(parents=True, exist_ok=True)
                return cwd_log_dir
            except OSError:
                pass

        tmp_log_dir = Path(tempfile.gettempdir()) / "web-search" / log_dir_str
        if create_dirs:
            tmp_log_dir.mkdir(parents=True, exist_ok=True)
        return tmp_log_dir

    def _apply_model_suffix(self, model: str) -> str:
        try:
            url = self.grok_api_url
        except ValueError:
            return model
        if "openrouter" in url and ":online" not in model:
            return f"{model}:online"
        return model

    @property
    def grok_model(self) -> str:
        if self._cached_model is not None:
            return self._cached_model

        model = (
            self._get_setting("GROK_MODEL")
            or self._load_config_file().get("model")
            or self._DEFAULT_MODEL
        )
        self._cached_model = self._apply_model_suffix(model)
        return self._cached_model

    def set_model(self, model: str) -> None:
        config_data = self._load_config_file()
        config_data["model"] = model
        self._save_config_file(config_data)
        self._cached_model = self._apply_model_suffix(model)

    @staticmethod
    def _mask_api_key(key: str) -> str:
        if not key or len(key) <= 8:
            return "***"
        return f"{key[:4]}{'*' * (len(key) - 8)}{key[-4:]}"

    def get_config_info(self) -> dict:
        try:
            api_url = self.grok_api_url
            api_key_raw = self.grok_api_key
            api_key_masked = self._mask_api_key(api_key_raw)
            config_status = "✅ 配置完整"
        except ValueError as exc:
            api_url = "未配置"
            api_key_masked = "未配置"
            config_status = f"❌ 配置错误: {str(exc)}"

        tavily_keys = self.tavily_api_keys
        env_files = [str(path) for path in self._iter_env_files() if self._path_exists(path)]

        return {
            "GROK_API_URL": api_url,
            "GROK_API_KEY": api_key_masked,
            "GROK_MODEL": self.grok_model,
            "GROK_DEBUG": self.debug_enabled,
            "GROK_LOG_LEVEL": self.log_level,
            "GROK_LOG_DIR": str(self._resolve_log_dir(create_dirs=False)),
            "TAVILY_API_URL": self.tavily_api_url,
            "TAVILY_ENABLED": self.tavily_enabled,
            "TAVILY_API_KEY": self._mask_api_key(tavily_keys[0])
            if tavily_keys
            else "未配置",
            "TAVILY_API_KEYS_COUNT": len(tavily_keys),
            "FIRECRAWL_API_URL": self.firecrawl_api_url,
            "FIRECRAWL_API_KEY": self._mask_api_key(self.firecrawl_api_key)
            if self.firecrawl_api_key
            else "未配置",
            "ENV_FILES_LOADED": env_files,
            "config_status": config_status,
        }


config = Config()
