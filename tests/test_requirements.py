"""需求分析与筛选策略测试；通过假 LLM 响应运行，不访问外部接口。"""
from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any
from unittest.mock import patch

from langgraph.types import Command

from agent.domain import PolicyValidationError, RequirementProfile
from agent.graph import build_graph
from agent.nodes.requirements import RequirementNodes, route_after_requirement_analysis
from agent.requirements import LLMClient, LLMServiceError, ScreeningPolicyAgent
import main as main_module
from main import describe_criterion, format_value, humanize_text


DATA = Path(__file__).parent.parent / "static"


def reason_draft() -> dict[str, Any]:
    return {
        "stance": "recommend",
        "tag": "进店好",
        "sentence": "进店中位数较高，但进店成本只是中等，先留在中间档。",
    }


class ResponseSequence:
    def __init__(
        self,
        responses: list[dict[str, Any]],
        default: dict[str, Any] | None = None,
    ) -> None:
        self.responses = responses
        self.default = default
        self.calls = 0
        self.payloads: list[dict[str, Any]] = []

    def __call__(self, _system: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.payloads.append(payload)
        if self.calls < len(self.responses):
            response = self.responses[self.calls]
        elif self.default is not None:
            response = self.default
        else:
            raise IndexError("scripted LLM responses exhausted")
        self.calls += 1
        return response


def complete_profile() -> dict[str, Any]:
    return {
        "summary": "需求单将18-34岁女性列为主要受众，并把进店成本和进店量列为考核指标。",
        "objective": "种草并提升进店",
        "target_audience": "18-34 岁女性",
        "kpi": "进店成本和进店量",
        "campaign_constraints": [
            "需求单明确总预算为35万元，后续达人组合不得超过该金额",
            "需求单禁止医疗功效表述，内容发布前需要据此审核",
        ],
        "hard_requirements": [],
        "soft_preferences": [
            "需求单考核进店量，因此比较达人时优先参考进店中位数",
            "需求单要求护肤内容，因此优先选择美妆护肤达人",
        ],
        "exclusions": [
            "需求单明确排除机构营销号",
            "需求单明确排除纯 vlog 生活号",
        ],
        "consultant_assumptions": [
            "需求单没有规定近三个月的截止日；筛选有明确执行日，因此按执行日向前计算"
        ],
        "downstream_dependencies": [
            "投放复盘前需要确认品牌词搜索指数基线，否则无法计算实际增长幅度"
        ],
        "ambiguities": [],
        "clarification_questions": [],
        "is_complete": True,
    }


def valid_policy() -> dict[str, Any]:
    return {
        "name": "屿光精华达人筛选策略",
        "criteria": [
            {
                "field": "main_format",
                "operator": "not_contains",
                "value": "vlog",
                "hard": True,
                "weight": 0,
                "reason": "需求单明确排除纯 vlog 生活号",
            },
            {
                "field": "visit_store_med",
                "operator": "maximize",
                "value": None,
                "hard": False,
                "weight": 0.6,
                "reason": "需求单把进店量列为考核指标",
            },
            {
                "field": "cpuv",
                "operator": "minimize",
                "value": None,
                "hard": False,
                "weight": 0.4,
                "reason": "需求单把进店成本列为考核指标",
            },
        ],
        "manual_checks": ["确认达人是否已有两次屿光品牌合作记录"],
        "rationale": "需求单同时考核进店量和进店成本，因此剩余达人按这两个指标比较。",
    }


class RequirementNodeTests(unittest.TestCase):
    def test_cli_shows_short_message_for_model_connection_failure(self) -> None:
        output = io.StringIO()
        error = LLMServiceError("连接模型服务超时，请确认 Clash Verge 的 TUN 模式。")
        with (
            patch.object(main_module, "main", side_effect=error),
            redirect_stdout(output),
            self.assertRaises(SystemExit) as exit_context,
        ):
            main_module.run_cli()

        self.assertEqual(exit_context.exception.code, 1)
        self.assertIn("这次没有连上模型服务", output.getvalue())
        self.assertNotIn("Traceback", output.getvalue())

    def test_llm_client_ignores_proxy_environment_by_default(self) -> None:
        env_values = {
            "API_KEY": "test-key",
            "BASE_URL": "https://example.com",
            "LLM_MODEL": "test-model",
            "MODEL": "fallback-model",
            "LLM_TRUST_ENV": "false",
            "LLM_TIMEOUT_SECONDS": "120",
        }
        with patch("agent.requirements.os.getenv", side_effect=env_values.get):
            client = LLMClient.from_env()

        self.assertFalse(client.trust_env)
        self.assertEqual(client.timeout_seconds, 120)

    def test_llm_client_can_explicitly_use_proxy_environment(self) -> None:
        env_values = {
            "API_KEY": "test-key",
            "BASE_URL": "https://example.com",
            "LLM_MODEL": "test-model",
            "MODEL": "fallback-model",
            "LLM_TRUST_ENV": "true",
            "LLM_TIMEOUT_SECONDS": "45",
        }
        with patch("agent.requirements.os.getenv", side_effect=env_values.get):
            client = LLMClient.from_env()

        self.assertTrue(client.trust_env)
        self.assertEqual(client.timeout_seconds, 45)

    def test_complete_requirements_continue_to_policy(self) -> None:
        sequence = ResponseSequence([complete_profile(), valid_policy()])
        nodes = RequirementNodes(LLMClient(transport=sequence))

        analysis = nodes.analyze_requirements({"brief": {}, "brief_raw": {}})
        self.assertEqual(analysis["workflow_status"], "requirements_ready")
        self.assertEqual(route_after_requirement_analysis(analysis), "build_policy")

        policy_result = nodes.build_screening_policy(analysis)
        self.assertEqual(policy_result["workflow_status"], "awaiting_policy_review")
        self.assertEqual(len(policy_result["screening_policy"]["criteria"]), 3)

    def test_incomplete_requirements_stop_for_confirmation(self) -> None:
        response = complete_profile()
        response["clarification_questions"] = ["请确认单个达人最高报价。"]
        response["is_complete"] = False
        nodes = RequirementNodes(LLMClient(transport=ResponseSequence([response])))

        analysis = nodes.analyze_requirements({"brief": {}, "brief_raw": {}})
        self.assertEqual(analysis["workflow_status"], "needs_client_confirmation")
        self.assertEqual(
            route_after_requirement_analysis(analysis), "client_confirmation"
        )

    def test_non_blocking_ambiguity_does_not_stop_consultant_work(self) -> None:
        response = complete_profile()
        response["ambiguities"] = ["单个达人最高报价没有明确"]
        nodes = RequirementNodes(LLMClient(transport=ResponseSequence([response])))

        analysis = nodes.analyze_requirements({"brief": {}, "brief_raw": {}})
        self.assertEqual(analysis["workflow_status"], "requirements_ready")
        self.assertEqual(analysis["clarification_questions"], [])

    def test_questions_are_deduplicated_and_limited_to_three(self) -> None:
        response = complete_profile()
        response["clarification_questions"] = [
            "请确认合作统计周期。",
            "合作统计周期。",
            "请确认 KPI 验收数据来源。",
            "请确认品牌词搜索基线。",
            "请确认最终合规红线。",
        ]
        nodes = RequirementNodes(LLMClient(transport=ResponseSequence([response])))

        analysis = nodes.analyze_requirements({"brief": {}, "brief_raw": {}})
        self.assertEqual(analysis["workflow_status"], "needs_client_confirmation")
        self.assertEqual(len(analysis["clarification_questions"]), 3)
        self.assertEqual(
            sum("合作统计周期" in q for q in analysis["clarification_questions"]),
            1,
        )

    def test_long_sections_are_compacted(self) -> None:
        response = complete_profile()
        response["soft_preferences"] = [f"偏好 {index}" for index in range(8)]
        response["consultant_assumptions"] = [
            f"顾问假设 {index}" for index in range(6)
        ]
        response["downstream_dependencies"] = [
            f"后续事项 {index}" for index in range(7)
        ]
        nodes = RequirementNodes(LLMClient(transport=ResponseSequence([response])))

        analysis = nodes.analyze_requirements({"brief": {}, "brief_raw": {}})
        profile = analysis["requirement_profile"]
        self.assertEqual(len(profile["soft_preferences"]), 5)
        self.assertEqual(len(profile["consultant_assumptions"]), 3)
        self.assertEqual(len(profile["downstream_dependencies"]), 4)

    def test_console_text_hides_machine_notation(self) -> None:
        self.assertEqual(format_value(["腰部", "初级"]), "腰部、初级")
        description = describe_criterion(
            {
                "field": "tier",
                "operator": "in",
                "value": ["腰部", "初级"],
                "hard": False,
                "weight": 0.2,
                "reason": "需求单要求优先选择这两个达人层级",
            },
            1,
        )
        self.assertIn("腰部、初级", description)
        self.assertNotIn("['", description)
        self.assertNotIn("[加权评分]", description)

        text = humanize_text("criteria 使用 maximize/minimize，其他放入 manual_checks")
        for machine_word in ("criteria", "maximize", "minimize", "manual_checks"):
            self.assertNotIn(machine_word, text)

    def test_graph_skips_client_confirmation_when_requirements_are_complete(self) -> None:
        sequence = ResponseSequence(
            [complete_profile(), valid_policy()],
            default=reason_draft(),
        )
        graph = build_graph(LLMClient(transport=sequence))
        config = {"configurable": {"thread_id": "complete-review"}}
        result = graph.invoke(
            {
                "brief_path": str(DATA / "甲方投放需求单_屿光精华.xlsx"),
                "talent_path": str(DATA / "蒲公英达人数据_屿光精华.xlsx"),
            },
            config=config,
        )

        payload = result["__interrupt__"][0].value
        self.assertEqual(payload["type"], "policy_review")
        self.assertEqual(result["workflow_status"], "awaiting_policy_review")

        completed = graph.invoke(
            Command(resume={"action": "approve"}),
            config=config,
        )
        self.assertEqual(completed["workflow_status"], "reasons_ready")
        self.assertEqual(completed["policy_review_status"], "approved")
        self.assertFalse(completed["next_step"]["implemented"])
        self.assertEqual(completed["next_step"]["name"], "audit_reasons")

        evaluated = completed["evaluated_talents"]
        by_name = {item["nickname"]: item for item in evaluated}
        self.assertNotIn("走走停停Vlog", by_name)
        self.assertNotIn("慢生活收藏家", by_name)
        self.assertNotIn("Mira的护肤手账", by_name)
        self.assertEqual(by_name["麦浪MaiLang"]["band"], "clear_yes")
        self.assertEqual(by_name["麦浪MaiLang"]["source"], "rules")
        self.assertEqual(by_name["麦浪MaiLang"]["tag"], "进店好")
        self.assertEqual(by_name["美丽研习社官方号"]["band"], "clear_no")
        self.assertEqual(by_name["美丽研习社官方号"]["source"], "rules")
        self.assertEqual(by_name["早八小满"]["band"], "mid")
        self.assertEqual(by_name["早八小满"]["tag"], "进店好")
        self.assertEqual(by_name["早八小满"]["source"], "model")
        self.assertEqual(
            by_name["早八小满"]["evidence"]["visit_store_med"],
            by_name["早八小满"]["talent"]["visit_store_med"],
        )

        reason_payloads = [item for item in sequence.payloads if "candidate" in item]
        mid_names = [item["nickname"] for item in evaluated if item["band"] == "mid"]
        self.assertEqual(len(reason_payloads), len(mid_names))
        everyone = [item["nickname"] for item in completed["cleaned_talents"]]
        for payload in reason_payloads:
            text = json.dumps(payload, ensure_ascii=False)
            nickname = payload["candidate"]["nickname"]
            self.assertIn(nickname, mid_names)
            for name in everyone:
                if name != nickname:
                    self.assertNotIn(name, text)

    def test_graph_resumes_after_client_answers(self) -> None:
        incomplete = complete_profile()
        incomplete["clarification_questions"] = ["请确认合作记录的统计范围。"]
        sequence = ResponseSequence([incomplete, complete_profile(), valid_policy()])
        graph = build_graph(LLMClient(transport=sequence))
        config = {"configurable": {"thread_id": "client-answer"}}
        result = graph.invoke(
            {
                "brief_path": str(DATA / "甲方投放需求单_屿光精华.xlsx"),
                "talent_path": str(DATA / "蒲公英达人数据_屿光精华.xlsx"),
            },
            config=config,
        )
        self.assertEqual(result["__interrupt__"][0].value["type"], "client_confirmation")

        resumed = graph.invoke(
            Command(resume={"answers": "仅统计屿光品牌过去一年合作记录。"}),
            config=config,
        )
        self.assertEqual(resumed["__interrupt__"][0].value["type"], "policy_review")
        self.assertEqual(resumed["client_answers"], "仅统计屿光品牌过去一年合作记录。")

    def test_revision_feedback_rebuilds_policy_before_approval(self) -> None:
        revised = valid_policy()
        revised["name"] = "按顾问意见修订的策略"
        sequence = ResponseSequence([complete_profile(), valid_policy(), revised])
        graph = build_graph(LLMClient(transport=sequence))
        config = {"configurable": {"thread_id": "policy-revision"}}
        result = graph.invoke(
            {
                "brief_path": str(DATA / "甲方投放需求单_屿光精华.xlsx"),
                "talent_path": str(DATA / "蒲公英达人数据_屿光精华.xlsx"),
            },
            config=config,
        )
        self.assertEqual(result["__interrupt__"][0].value["type"], "policy_review")

        revised_result = graph.invoke(
            Command(
                resume={"action": "revise", "feedback": "提高进店表现的权重。"}
            ),
            config=config,
        )
        self.assertEqual(
            revised_result["screening_policy"]["name"], "按顾问意见修订的策略"
        )
        self.assertEqual(revised_result["policy_revision_count"], 1)
        self.assertEqual(
            revised_result["__interrupt__"][0].value["type"], "policy_review"
        )

    def test_cli_approval_reaches_next_step_without_client_prompt(self) -> None:
        graph = build_graph(
            LLMClient(
                transport=ResponseSequence(
                    [complete_profile(), valid_policy()],
                    default=reason_draft(),
                )
            )
        )
        output = io.StringIO()
        with (
            patch.object(main_module, "build_graph", return_value=graph),
            patch.object(
                main_module,
                "save_result",
                return_value=Path("output/step3_requirements.json"),
            ),
            patch("builtins.input", return_value="同意"),
            redirect_stdout(output),
        ):
            main_module.main()

        rendered = output.getvalue()
        self.assertIn("筛选方案审核通过", rendered)
        self.assertIn("推荐名单", rendered)
        self.assertIn("中间档", rendered)
        self.assertNotIn("表格预处理", rendered)
        self.assertNotIn("接口已经预留", rendered)
        self.assertNotIn("请把客户的补充答复", rendered)

    def test_cli_distinguishes_waiting_for_client(self) -> None:
        incomplete = complete_profile()
        incomplete["clarification_questions"] = [
            "需求单没有说明合作记录的统计范围；不同范围会改变达人是否被排除，需要客户确认。"
        ]
        graph = build_graph(LLMClient(transport=ResponseSequence([incomplete])))
        output = io.StringIO()
        with (
            patch.object(main_module, "build_graph", return_value=graph),
            patch.object(main_module, "save_result", return_value=Path("result.json")),
            patch("builtins.input", return_value=""),
            redirect_stdout(output),
        ):
            main_module.main()

        rendered = output.getvalue()
        self.assertIn("当前状态：等待客户答复，筛选方案尚未生成", rendered)
        self.assertIn("尚未收到客户答复，筛选没有继续", rendered)
        self.assertNotIn("筛选方案审核通过", rendered)

    def test_cli_marks_revision_before_regenerating_policy(self) -> None:
        revised = valid_policy()
        revised["name"] = "修改后的方案"
        graph = build_graph(
            LLMClient(
                transport=ResponseSequence([complete_profile(), valid_policy(), revised])
            )
        )
        output = io.StringIO()
        with (
            patch.object(main_module, "build_graph", return_value=graph),
            patch.object(main_module, "save_result", return_value=Path("result.json")),
            patch("builtins.input", side_effect=["提高进店量的权重", ""]),
            redirect_stdout(output),
        ):
            main_module.main()

        rendered = output.getvalue()
        self.assertIn("筛选方案需要修改", rendered)
        self.assertIn("正在重新生成筛选方案", rendered)
        self.assertIn("方案仍在等待审核", rendered)
        self.assertNotIn("筛选方案审核通过", rendered)

    def test_render_shows_all_scoring_metrics_with_direction_and_weight(self) -> None:
        metrics = [
            ("ratio_female", "maximize", 0.20, "需求单将女性列为主要受众"),
            ("ratio_age_18_24", "maximize", 0.15, "需求单将年轻人列为主要受众"),
            ("cpe", "minimize", 0.15, "需求单考核单次互动成本"),
            ("interact_med", "maximize", 0.15, "需求单考核互动总量"),
            ("exposure_med", "maximize", 0.15, "需求单考核总曝光"),
            ("ctr", "maximize", 0.10, "需求单要求内容带动进一步了解"),
            ("ratio_search", "maximize", 0.10, "需求单考核品牌搜索增长"),
        ]
        policy = {
            "name": "完整方案",
            "criteria": [valid_policy()["criteria"][0]]
            + [
                {
                    "field": field,
                    "operator": operator,
                    "value": None,
                    "hard": False,
                    "weight": weight,
                    "reason": reason,
                }
                for field, operator, weight, reason in metrics
            ],
            "manual_checks": [],
            "rationale": "需求单明确了受众和考核指标，因此按对应数据比较。",
        }
        result = {
            "brief": {},
            "talents": [],
            "issues": [],
            "requirement_profile": complete_profile(),
            "screening_policy": policy,
            "workflow_status": "awaiting_policy_review",
        }

        output = io.StringIO()
        with (
            patch.object(main_module, "save_result", return_value=Path("result.json")),
            redirect_stdout(output),
        ):
            main_module.render_result(result)

        rendered = output.getvalue()
        for field, _operator, weight, reason in metrics:
            self.assertIn(main_module.FIELD_LABELS[field], rendered)
            self.assertIn(f"权重 {weight * 100:g}%", rendered)
            self.assertIn(reason, rendered)
        self.assertIn("以上评分权重合计：100%", rendered)
        self.assertNotIn("另有", rendered)
        for phrase in ("硬性筛选", "加权评分", "顾问暂定口径", "后续要补齐"):
            self.assertNotIn(phrase, rendered)

    def test_policy_without_reason_never_reaches_review(self) -> None:
        invalid = valid_policy()
        invalid["criteria"][1]["reason"] = ""
        builder = ScreeningPolicyAgent(
            LLMClient(transport=ResponseSequence([invalid, invalid]))
        )

        with self.assertRaises(PolicyValidationError) as context:
            builder.build(RequirementProfile.from_dict(complete_profile()))
        self.assertIn("事实依据", str(context.exception))

    def test_policy_repairs_invalid_weight_once(self) -> None:
        invalid = valid_policy()
        invalid["criteria"][1]["weight"] = 0.2
        sequence = ResponseSequence([invalid, valid_policy()])
        builder = ScreeningPolicyAgent(LLMClient(transport=sequence))

        policy = builder.build(RequirementProfile.from_dict(complete_profile()))
        self.assertEqual(sequence.calls, 2)
        self.assertAlmostEqual(
            sum(c.weight for c in policy.criteria if not c.hard), 1.0
        )

    def test_policy_rejects_unknown_field_after_retry(self) -> None:
        invalid = valid_policy()
        invalid["criteria"][0]["field"] = "imaginary_metric"
        sequence = ResponseSequence([invalid, invalid])
        builder = ScreeningPolicyAgent(LLMClient(transport=sequence))

        with self.assertRaises(PolicyValidationError):
            builder.build(RequirementProfile.from_dict(complete_profile()))
        self.assertEqual(sequence.calls, 2)


if __name__ == "__main__":
    unittest.main()
