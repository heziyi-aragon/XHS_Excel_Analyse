from __future__ import annotations

from collections import Counter
from unittest.mock import patch

import pytest

from agent.evaluation import EvaluationAgent
from agent.graph import build_graph
from agent.llm_routing import (
    FailoverLLMClient,
    LLMConfigurationError,
    ModuleLLMClients,
)
from agent.requirements import LLMClient
from agent.portfolio_planning import prepare_portfolio_config
from tests.test_portfolio_planning import _input as portfolio_input


MODULES = ("REQUIREMENTS", "EVALUATION", "AUDIT", "PORTFOLIO", "ADVICE", "BACKUP")


def _complete_environment() -> dict[str, str]:
    values = {"LLM_TRUST_ENV": "false"}
    for index, name in enumerate(MODULES, start=1):
        prefix = f"LLM_{name}"
        values.update({
            f"{prefix}_API_KEY": f"test-key-{name.lower()}",
            f"{prefix}_BASE_URL": f"https://{name.lower()}.example.com/v1",
            f"{prefix}_MODEL": f"model-{name.lower()}",
            f"{prefix}_TIMEOUT_SECONDS": str(index * 10),
            f"{prefix}_STREAM": "true" if index % 2 else "false",
        })
    return values


def test_module_clients_read_six_independent_routes() -> None:
    with patch.dict("os.environ", _complete_environment(), clear=True):
        clients = ModuleLLMClients.from_env(load_dotenv_file=False)

    assert clients.requirements.model == "model-requirements"
    assert clients.requirements.timeout_seconds == 10
    assert clients.requirements.stream is True
    assert clients.evaluation.model == "model-evaluation"
    assert clients.evaluation.timeout_seconds == 20
    assert clients.evaluation.stream is False
    assert clients.audit.base_url == "https://audit.example.com/v1"
    assert clients.portfolio.model == "model-portfolio"
    assert clients.advice.model == "model-advice"
    assert clients.backup_settings.model == "model-backup"
    assert all(not client.using_backup for client in clients.module_clients())


def test_incomplete_primary_starts_that_module_on_the_shared_backup() -> None:
    environment = _complete_environment()
    environment.pop("LLM_EVALUATION_MODEL")
    with patch.dict("os.environ", environment, clear=True):
        clients = ModuleLLMClients.from_env(load_dotenv_file=False)

    assert clients.evaluation.using_backup is True
    assert clients.evaluation.active_route == "BACKUP"
    assert clients.evaluation.model == "model-backup"
    assert clients.audit.using_backup is False


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("LLM_BACKUP_API_KEY", ""),
        ("LLM_BACKUP_BASE_URL", "http://["),
        ("LLM_BACKUP_TIMEOUT_SECONDS", "0"),
        ("LLM_BACKUP_STREAM", "sometimes"),
    ],
)
def test_invalid_backup_configuration_stops_startup(variable: str, value: str) -> None:
    environment = _complete_environment()
    environment[variable] = value
    with patch.dict("os.environ", environment, clear=True):
        with pytest.raises(LLMConfigurationError, match=variable):
            ModuleLLMClients.from_env(load_dotenv_file=False)


def test_service_failure_switches_once_and_stays_on_backup() -> None:
    calls = Counter()

    def primary_transport(_system, _payload):
        calls["primary"] += 1
        raise RuntimeError("primary unavailable")

    def backup_transport(_system, payload):
        calls["backup"] += 1
        return {"value": payload["value"]}

    client = FailoverLLMClient(
        module_name="EVALUATION",
        primary=LLMClient(model="primary-model", transport=primary_transport),
        backup=LLMClient(model="backup-model", transport=backup_transport),
    )

    assert client.complete_json("prompt", {"value": 1}) == {"value": 1}
    assert client.complete_json("prompt", {"value": 2}) == {"value": 2}
    assert calls == Counter(primary=1, backup=2)
    assert client.using_backup is True
    assert client.active_route == "BACKUP"


def test_invalid_primary_url_starts_only_that_module_on_backup() -> None:
    environment = _complete_environment()
    environment["LLM_AUDIT_BASE_URL"] = "not-a-valid-url"
    with patch.dict("os.environ", environment, clear=True):
        clients = ModuleLLMClients.from_env(load_dotenv_file=False)

    assert clients.audit.using_backup is True
    assert clients.evaluation.using_backup is False


def test_validation_failure_switches_only_the_affected_module() -> None:
    primary = LLMClient(model="primary-model", transport=lambda *_: {"invalid": True})
    backup = LLMClient(model="backup-model", transport=lambda *_: {"valid": True})
    evaluation = FailoverLLMClient("EVALUATION", primary=primary, backup=backup)
    audit = FailoverLLMClient("AUDIT", primary=primary, backup=backup)

    assert evaluation.complete_json("prompt", {}) == {"invalid": True}
    assert evaluation.report_validation_failure(ValueError("schema mismatch")) is True
    assert evaluation.complete_json("prompt", {}) == {"valid": True}
    assert evaluation.using_backup is True
    assert audit.using_backup is False


def test_evaluation_schema_failure_retries_on_backup_and_keeps_it_active() -> None:
    calls = Counter()

    def primary_transport(_system, _payload):
        calls["primary"] += 1
        return {"stance": "recommend", "tag": "invalid-tag", "sentence": "格式错误。"}

    def backup_transport(_system, _payload):
        calls["backup"] += 1
        return {
            "stance": "recommend",
            "tag": "进店好",
            "sentence": "进店中位数较高，但进店成本仍有优化空间。",
        }

    client = FailoverLLMClient(
        "EVALUATION",
        primary=LLMClient(model="primary-model", transport=primary_transport),
        backup=LLMClient(model="backup-model", transport=backup_transport),
    )
    payload = {
        "xhs_id": "routing-test",
        "nickname": "路由测试达人",
        "band": "mid",
        "score": 0.55,
        "missing_fields": [],
        "attempt": 1,
        "talent": {
            "xhs_id": "routing-test",
            "nickname": "路由测试达人",
            "visit_store_med": 800,
            "cpuv": 10,
            "main_format": "图文为主",
        },
        "score_parts": [
            {
                "field": "visit_store_med",
                "operator": "maximize",
                "label": "进店中位数",
                "raw": 800,
                "normalized": 0.8,
                "weight": 0.6,
                "weight_effective": 0.6,
                "contribution": 0.48,
            },
            {
                "field": "cpuv",
                "operator": "minimize",
                "label": "进店成本",
                "raw": 10,
                "normalized": 0.2,
                "weight": 0.4,
                "weight_effective": 0.4,
                "contribution": 0.08,
            },
        ],
    }

    result = EvaluationAgent(client).write_reason(payload)

    assert result["tag"] == "进店好"
    assert result["error"] is None
    assert calls == Counter(primary=1, backup=1)
    assert client.using_backup is True
def test_graph_wires_each_dedicated_client_to_its_module() -> None:
    clients = {
        name: LLMClient(model=name, transport=lambda *_: {})
        for name in ("requirements", "evaluation", "audit", "portfolio", "advice")
    }
    with (
        patch("agent.graph.RequirementNodes") as requirement_nodes,
        patch("agent.graph.EvaluationNodes") as evaluation_nodes,
        patch("agent.graph.AuditNodes") as audit_nodes,
        patch("agent.graph.make_portfolio_handoff_node", return_value=lambda _state: {} ) as portfolio_node,
        patch("agent.graph.make_advice_node", return_value=lambda _state: {}) as advice_node,
    ):
        build_graph(
            clients["requirements"],
            evaluation_llm_client=clients["evaluation"],
            audit_llm_client=clients["audit"],
            portfolio_llm_client=clients["portfolio"],
            advice_llm_client=clients["advice"],
        )

    requirement_nodes.assert_called_once_with(clients["requirements"])
    evaluation_nodes.assert_called_once_with(clients["evaluation"])
    audit_nodes.assert_called_once_with(clients["audit"])
    portfolio_node.assert_called_once_with(clients["portfolio"])
    advice_node.assert_called_once_with(clients["advice"])


def test_portfolio_schema_failure_retries_on_backup() -> None:
    calls = Counter()

    def primary_transport(_system, _payload):
        calls["primary"] += 1
        return {"unexpected": "invalid planning response"}

    def backup_transport(_system, _payload):
        calls["backup"] += 1
        return {"proposals": [{
            "path": "max_posts_per_creator",
            "value": 2,
            "source_quote": "每人2篇",
            "source_kind": "feedback",
        }]}

    client = FailoverLLMClient(
        "PORTFOLIO",
        primary=LLMClient(model="primary-model", transport=primary_transport),
        backup=LLMClient(model="backup-model", transport=backup_transport),
    )

    config, metadata = prepare_portfolio_config(
        portfolio_input(), {}, client, {"message": "每人2篇"}
    )

    assert config["max_posts_per_creator"] == 2
    assert metadata["model_used"] is True
    assert calls == Counter(primary=1, backup=1)
    assert client.using_backup is True

# All routing tests use offline transports and never call external model APIs.
