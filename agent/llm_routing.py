"""从同一份环境配置创建五类模块客户端，并提供进程内主备熔断。"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Mapping
from urllib.parse import urlsplit

from agent.requirements import LLMClient
from agent.runtime_logging import LOGGER


MODULE_NAMES = ("REQUIREMENTS", "EVALUATION", "AUDIT", "PORTFOLIO", "ADVICE")
_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


class LLMConfigurationError(RuntimeError):
    """模块线路配置缺失或格式无效。"""


@dataclass(frozen=True)
class LLMSettings:
    """一条 OpenAI 兼容线路的已校验配置。"""

    name: str
    api_key: str
    base_url: str
    model: str
    timeout_seconds: float
    stream: bool
    trust_env: bool

    def create_client(self) -> LLMClient:
        return LLMClient(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            trust_env=self.trust_env,
            timeout_seconds=self.timeout_seconds,
            stream=self.stream,
        )


class FailoverLLMClient:
    """主线路首次失败后，本进程内为当前模块固定使用全局备用线路。"""

    def __init__(
        self,
        module_name: str,
        *,
        primary: LLMClient | None,
        backup: LLMClient,
    ) -> None:
        self.module_name = module_name.upper()
        self._primary = primary
        self._backup = backup
        self._using_backup = primary is None
        if self._using_backup:
            LOGGER.warning(
                "llm.route_disabled module=%s route=%s fallback=BACKUP",
                self.module_name,
                self.module_name,
            )

    @property
    def using_backup(self) -> bool:
        return self._using_backup

    @property
    def active_route(self) -> str:
        return "BACKUP" if self._using_backup else self.module_name

    @property
    def _active_client(self) -> LLMClient:
        return self._backup if self._using_backup else self._primary  # type: ignore[return-value]

    @property
    def api_key(self) -> str:
        return self._active_client.api_key

    @property
    def base_url(self) -> str:
        return self._active_client.base_url

    @property
    def model(self) -> str:
        return self._active_client.model

    @property
    def timeout_seconds(self) -> float:
        return self._active_client.timeout_seconds

    @property
    def stream(self) -> bool:
        return self._active_client.stream

    @property
    def trust_env(self) -> bool:
        return self._active_client.trust_env

    def complete_json(self, system_prompt: str, payload: dict) -> dict:
        if self._using_backup:
            return self._backup.complete_json(system_prompt, payload)
        try:
            return self._primary.complete_json(system_prompt, payload)  # type: ignore[union-attr]
        except Exception as exc:
            self._activate_backup(exc)
            return self._backup.complete_json(system_prompt, payload)

    def report_validation_failure(self, error: Exception) -> bool:
        """业务模块确认模型格式无效时熔断；返回本次是否发生切换。"""
        if self._using_backup:
            return False
        self._activate_backup(error)
        return True

    def _activate_backup(self, error: Exception) -> None:
        self._using_backup = True
        LOGGER.warning(
            "llm.failover module=%s from=%s to=BACKUP reason=%s",
            self.module_name,
            self.module_name,
            type(error).__name__,
        )


@dataclass(frozen=True)
class ModuleLLMClients:
    """五类模块客户端；每类共享配置，但拥有独立熔断状态。"""

    requirements: FailoverLLMClient
    evaluation: FailoverLLMClient
    audit: FailoverLLMClient
    portfolio: FailoverLLMClient
    advice: FailoverLLMClient
    backup_settings: LLMSettings

    @classmethod
    def from_env(cls, *, load_dotenv_file: bool = True) -> "ModuleLLMClients":
        if load_dotenv_file:
            try:
                from dotenv import load_dotenv

                load_dotenv()
            except ImportError:
                pass
        environment = dict(os.environ)
        trust_env = _parse_global_trust_env(environment)
        backup, backup_errors = _read_settings("BACKUP", environment, trust_env)
        if backup is None:
            raise LLMConfigurationError("备用线路配置无效：" + "；".join(backup_errors))

        clients = {}
        shared_backup = backup.create_client()
        for name in MODULE_NAMES:
            settings, errors = _read_settings(name, environment, trust_env)
            if errors:
                LOGGER.warning(
                    "llm.config_invalid module=%s fields=%s fallback=BACKUP",
                    name,
                    ",".join(errors),
                )
            clients[name.lower()] = FailoverLLMClient(
                name,
                primary=settings.create_client() if settings is not None else None,
                backup=shared_backup,
            )
        return cls(backup_settings=backup, **clients)

    def module_clients(self) -> tuple[FailoverLLMClient, ...]:
        return (
            self.requirements,
            self.evaluation,
            self.audit,
            self.portfolio,
            self.advice,
        )


def _read_settings(
    name: str,
    environment: Mapping[str, str],
    trust_env: bool,
) -> tuple[LLMSettings | None, list[str]]:
    prefix = f"LLM_{name}"
    errors: list[str] = []

    def required(field: str) -> str:
        variable = f"{prefix}_{field}"
        value = environment.get(variable, "").strip()
        if not value:
            errors.append(variable)
        return value

    api_key = required("API_KEY")
    base_url = required("BASE_URL")
    model = required("MODEL")
    if base_url:
        try:
            parsed = urlsplit(base_url)
            valid_url = parsed.scheme in {"http", "https"} and bool(parsed.netloc)
        except ValueError:
            valid_url = False
        if not valid_url:
            errors.append(f"{prefix}_BASE_URL")

    timeout_name = f"{prefix}_TIMEOUT_SECONDS"
    timeout_raw = environment.get(timeout_name, "").strip()
    try:
        timeout = float(timeout_raw)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError
    except ValueError:
        timeout = 0.0
        errors.append(timeout_name)

    stream_name = f"{prefix}_STREAM"
    stream_raw = environment.get(stream_name, "").strip().lower()
    if stream_raw in _TRUE_VALUES:
        stream = True
    elif stream_raw in _FALSE_VALUES:
        stream = False
    else:
        stream = False
        errors.append(stream_name)

    errors = list(dict.fromkeys(errors))
    if errors:
        return None, errors
    return LLMSettings(
        name=name,
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout_seconds=timeout,
        stream=stream,
        trust_env=trust_env,
    ), []


def _parse_global_trust_env(environment: Mapping[str, str]) -> bool:
    variable = "LLM_TRUST_ENV"
    raw = environment.get(variable, "false").strip().lower()
    if raw in _TRUE_VALUES:
        return True
    if raw in _FALSE_VALUES:
        return False
    raise LLMConfigurationError(f"{variable} 必须是 true 或 false")
