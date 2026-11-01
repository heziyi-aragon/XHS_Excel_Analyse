"""仅验证验收工具计数和失败分类，不把假模型结果当成真实准确率。"""
from copy import deepcopy
import json

import pytest

from agent.requirements import LLMClient, LLMServiceError
from scripts.audit_acceptance import acceptance_cases, judge_case, main, numeric_variant_cases, run_acceptance, validate_cases


def test_acceptance_runner_separates_code_rule_from_model_cases_and_preserves_inputs():
    cases = acceptance_cases()
    original = deepcopy(cases)
    expected = iter(case["expected"] for case in cases if case["category"] == "model_semantics")

    def transport(*_args):
        answer = next(expected)
        return {"verdict": answer["verdict"], "issues": [
            {"code": code, "message": "假模型返回预先构造的意见，用于校验报告计数。", "fields": answer["fields"].get(code, [])}
            for code in answer["issue_codes"]
        ]}

    report = run_acceptance(LLMClient(transport=transport), cases)
    assert report["success"] and report["matched_cases"] == report["total_cases"] == len(cases)
    assert report["semantic_matched"] == report["semantic_cases"] == len(report["calls"]) == len(cases) - 2
    assert all(case["input_unchanged"] for case in report["cases"])
    assert cases == original


def test_a_wrong_business_verdict_is_a_mismatch_rather_than_an_execution_error():
    case = next(case for case in acceptance_cases() if case["name"] == "valid_tradeoff")
    client = LLMClient(transport=lambda *_: {"verdict": "reject", "issues": [{
        "code": "data_conflict", "message": "假模型误报。", "fields": ["visit_store_med"],
    }]})
    report = run_acceptance(client, [case])
    assert not report["success"]
    assert report["completed_cases"] == 1 and report["matched_cases"] == 0
    assert report["execution_errors"] == report["not_run_cases"] == 0


def test_invalid_model_output_is_reported_as_an_execution_error_and_next_case_still_runs():
    cases = [case for case in acceptance_cases() if case["name"] in ("valid_tradeoff", "unrelated_missing_price")]
    responses = iter([{"verdict": "reject", "issues": []}] * 2 + [{"verdict": "pass", "issues": []}])
    report = run_acceptance(LLMClient(transport=lambda *_: next(responses)), cases)
    assert not report["success"] and report["execution_errors"] == 1
    assert report["completed_cases"] == report["matched_cases"] == 1
    assert report["not_run_cases"] == 0 and len(report["calls"]) == 3
    assert "actual" not in report["cases"][0]


def test_service_failure_stops_further_requests_without_fabricating_verdicts_or_logging_secrets():
    cases = [case for case in acceptance_cases() if case["category"] == "model_semantics"]

    def transport(*_args):
        raise LLMServiceError("server echoed secret-for-test")

    report = run_acceptance(LLMClient(transport=transport), cases)
    assert report["execution_errors"] == len(report["calls"]) == 1
    assert report["not_run_cases"] == len(cases) - 1
    assert report["completed_cases"] == report["matched_cases"] == 0
    assert "actual" not in report["cases"][0]
    assert "secret-for-test" not in json.dumps(report)


def test_expected_issue_type_cannot_hide_an_extra_false_accusation():
    case = next(case for case in acceptance_cases() if case["name"] == "wrong_visit_number")
    assert not judge_case(case, {"verdict": "reject", "issues": [
        {"code": "data_conflict", "fields": ["visit_store_med"]},
        {"code": "client_exclusion", "fields": ["main_format"]},
    ]})


def test_repeated_runs_measure_verdict_instability_without_rewriting_expected_answers():
    case = next(case for case in acceptance_cases() if case["name"] == "valid_tradeoff")
    original = deepcopy(case)
    answers = iter([{"verdict": "pass", "issues": []}, {"verdict": "reject", "issues": [
        {"code": "data_conflict", "message": "模拟误报", "fields": []},
    ]}])
    report = run_acceptance(LLMClient(transport=lambda *_: next(answers)), [case], repeat=2)
    assert report["unstable_cases"] == [case["name"]]
    assert report["false_rejects"] == 1 and report["false_accepts"] == 0
    assert report["matched_cases"] == 1 and not report["success"]
    assert [row["repeat_index"] for row in report["cases"]] == [1, 2]
    assert case == original


@pytest.mark.parametrize("cases,repeat", [([], 1), (None, 0), (None, -1)])
def test_empty_or_invalid_runs_cannot_report_success(cases, repeat):
    with pytest.raises(ValueError):
        run_acceptance(LLMClient(transport=lambda *_: pytest.fail("不能调用模型")), cases, repeat=repeat)


def test_listing_and_unknown_case_do_not_read_api_configuration(monkeypatch, capsys):
    monkeypatch.setattr(LLMClient, "from_env", lambda: pytest.fail("不能读取API配置"))
    monkeypatch.setattr("sys.argv", ["audit_acceptance", "--list", "--case", "mcn_does_not_prove_marketing"])
    assert main() == 0
    assert "mcn_does_not_prove_marketing" in capsys.readouterr().out
    monkeypatch.setattr("sys.argv", ["audit_acceptance", "--case", "typo"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2


def test_exported_case_bank_can_be_extended_and_selected_without_loading_api(tmp_path, monkeypatch):
    monkeypatch.setattr(LLMClient, "from_env", lambda: pytest.fail("导出不应读取API配置"))
    path = tmp_path / "my_cases.json"
    monkeypatch.setattr("sys.argv", ["audit_acceptance", "--numeric-variants", "--export-cases", str(path)])
    assert main() == 0
    cases = validate_cases(json.loads(path.read_text(encoding="utf-8")))
    assert len(cases) == len(acceptance_cases()) + len(numeric_variant_cases())
    monkeypatch.setattr("sys.argv", ["audit_acceptance", "--cases-file", str(path), "--list", "--case", "ctr_variant_1_correct"])
    assert main() == 0


@pytest.mark.parametrize("problem", ["duplicate", "missing_talent", "unknown_field", "contradictory_expected", "bad_context", "bad_code_type", "bad_field_type"])
def test_invalid_custom_cases_fail_before_any_api_call(problem):
    cases = [deepcopy(acceptance_cases()[1])]
    if problem == "duplicate":
        cases += deepcopy(cases)
    elif problem == "missing_talent":
        cases[0]["record"].pop("talent")
    elif problem == "unknown_field":
        cases[0]["expected"]["fields"]["data_conflict"] = ["unknown_metric"]
    elif problem == "bad_context":
        cases[0]["context"]["brief"] = "错误输入"
    elif problem == "bad_code_type":
        cases[0]["expected"]["issue_codes"] = [{}]
    elif problem == "bad_field_type":
        cases[0]["expected"]["fields"]["data_conflict"] = [{}]
    else:
        cases[0]["expected"]["verdict"] = "pass"
    with pytest.raises(ValueError):
        run_acceptance(LLMClient(transport=lambda *_: pytest.fail("非法题库不能发请求")), cases)


def test_numeric_variants_include_correct_and_wrong_pairs_with_consistent_raw_metrics():
    cases = numeric_variant_cases()
    assert len(cases) == 6
    for correct, wrong in zip(cases[::2], cases[1::2]):
        assert correct["record"]["talent"] == wrong["record"]["talent"]
        assert correct["expected"]["verdict"] == "pass" and wrong["expected"]["verdict"] == "reject"
        metrics = correct["record"]["talent"]
        assert metrics["read_med"] / metrics["exposure_med"] == metrics["ctr"]
        assert metrics["cpv"] == metrics["price_image"] / metrics["read_med"]
