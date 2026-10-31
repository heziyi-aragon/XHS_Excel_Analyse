"""清洗、筛选、去重、打分。不启动整张图，也不调用模型。"""
from __future__ import annotations

from pathlib import Path

from agent.domain import LoadReport
from agent.loaders import TalentLoader
from agent.nodes.screening import (
    preprocess_table,
    relax_screening,
    route_after_screen,
    screen_talents,
)
from agent.reasons import MAX_SENTENCE_LENGTH
from agent.screening import (
    clean_talent,
    dedupe_by_agency_fans,
    score_candidates,
)

DATA = Path(__file__).parent.parent / "static"


def _ratios() -> dict[str, float]:
    return {
        "ratio_female": 0.9,
        "ratio_age_18_24": 0.4,
        "ratio_age_over_44": 0.05,
    }


def _person(nickname: str, **extra: object) -> dict:
    record = {
        "nickname": nickname,
        "xhs_id": None,
        "mcn": None,
        "main_format": "图文为主",
        "missing_fields": [],
        "fans_wan": 10,
        **_ratios(),
    }
    record.update(extra)
    return record


def test_clean_talent_normalizes_commas_and_marks_missing_prices() -> None:
    cleaned = clean_talent(
        {
            "nickname": "  测试达人  ",
            "main_format": "图文，为主",
            "price_image": None,
            "price_video": None,
            "ctr": 0.1,
        }
    )
    assert cleaned["nickname"] == "测试达人"
    assert cleaned["main_format"] == "图文,为主"
    assert "price_image" in cleaned["missing_fields"]
    assert "price_video" in cleaned["missing_fields"]
    assert "mcn" not in cleaned["missing_fields"]


def test_same_agency_and_similar_fans_keep_the_larger_account() -> None:
    smaller = _person("小号", mcn="声量文化", fans_wan=10)
    larger = _person("大号", mcn="声量文化", fans_wan=30)
    different = _person(
        "年龄不同",
        mcn="声量文化",
        fans_wan=40,
        ratio_age_18_24=0.7,
    )
    independent = _person("独立", fans_wan=5)
    dropped, kept = dedupe_by_agency_fans([smaller, larger, different, independent])
    assert {item["nickname"] for item in kept} == {"大号", "年龄不同", "独立"}
    assert dropped[0]["nickname"] == "小号"
    assert dropped[0]["kept"] == "大号"


def test_client_text_exclusion_is_not_relaxed() -> None:
    talents = [
        _person("正常", fans_wan=10),
        *[_person(f"生活{index}", fans_wan=80, main_format="vlog为主") for index in range(5)],
    ]
    policy = {
        "criteria": [
            {
                "field": "main_format",
                "operator": "not_contains",
                "value": "vlog",
                "hard": True,
                "weight": 0,
                "reason": "需求单明确排除 vlog",
            }
        ]
    }
    state = {
        "approved_policy": policy,
        "cleaned_talents": talents,
        "issues": [],
        "screen_relax_count": 0,
    }
    state.update(screen_talents(state))
    assert route_after_screen(state) == "score"
    assert state.get("screen_relax_count", 0) == 0
    assert [item["nickname"] for item in state["screened_talents"]] == ["正常"]


def test_numeric_hard_rule_relaxes_at_most_three_times() -> None:
    talents = [
        _person(f"达人{fans}", fans_wan=fans)
        for fans in (10, 20, 30, 40, 50, 60)
    ]
    talents.append(_person("生活号", fans_wan=100, main_format="vlog为主"))
    policy = {
        "criteria": [
            {
                "field": "fans_wan",
                "operator": "gte",
                "value": 45,
                "hard": True,
                "weight": 0,
                "reason": "先看粉丝量",
            },
            {
                "field": "main_format",
                "operator": "not_contains",
                "value": "vlog",
                "hard": True,
                "weight": 0,
                "reason": "排除 vlog",
            },
            {
                "field": "fans_wan",
                "operator": "maximize",
                "value": None,
                "hard": False,
                "weight": 1,
                "reason": "粉丝更多优先",
            },
        ]
    }
    state = {
        "approved_policy": policy,
        "cleaned_talents": talents,
        "issues": [],
        "screen_relax_count": 0,
    }
    for _ in range(6):
        state.update(screen_talents(state))
        if route_after_screen(state) == "score":
            break
        state.update(relax_screening(state))
    else:
        raise AssertionError("筛选放宽没有在 3 次内停止")

    assert state["screen_relax_count"] == 3
    assert {item["nickname"] for item in state["screened_talents"]} == {
        "达人30",
        "达人40",
        "达人50",
        "达人60",
    }
    assert any("放宽" in line for line in state["screen_log"])
    assert all("vlog" not in line for line in state["screen_log"] if "放宽为" in line)


def test_clear_bands_are_tagged_from_the_dominant_metric() -> None:
    talents = [
        _person("低", ctr=0.01),
        _person("中", ctr=0.05),
        _person("高", ctr=0.10),
    ]
    policy = {
        "criteria": [
            {
                "field": "ctr",
                "operator": "maximize",
                "value": None,
                "hard": False,
                "weight": 1,
                "reason": "点击率越高越好",
            }
        ]
    }
    scored = {item["nickname"]: item for item in score_candidates(talents, policy)}
    assert scored["高"]["band"] == "clear_yes"
    assert scored["高"]["tag"] == "数据好"
    assert scored["高"]["source"] == "rules"
    assert scored["中"]["band"] == "mid"
    assert scored["中"]["tag"] is None
    assert scored["低"]["band"] == "clear_no"
    assert scored["低"]["tag"] == "数据差"
    assert len(scored["高"]["sentence"]) <= MAX_SENTENCE_LENGTH
    assert "点击率" in scored["高"]["sentence"]


def test_bilingual_category_and_flag_matching_in_scoring() -> None:
    talents = [
        _person("美妆新号", content_type="Beauty", is_new_kol="Y", ctr=0.12),
        _person("穿搭老号", content_type="Fashion", is_new_kol="N", ctr=0.04),
    ]
    policy = {
        "criteria": [
            {
                "field": "content_type",
                "operator": "contains",
                "value": "美妆",
                "hard": False,
                "weight": 0.5,
                "reason": "美妆护肤垂类优先",
            },
            {
                "field": "is_new_kol",
                "operator": "eq",
                "value": "是",
                "hard": False,
                "weight": 0.5,
                "reason": "新达人优先",
            },
        ]
    }
    scored = {item["nickname"]: item for item in score_candidates(talents, policy)}
    assert scored["美妆新号"]["score"] == 1.0
    assert scored["美妆新号"]["band"] == "clear_yes"
    assert scored["穿搭老号"]["score"] == 0.0
    assert scored["穿搭老号"]["band"] == "clear_no"


def test_fixture_splits_three_bands_and_dedupes_agency_pair() -> None:
    report = LoadReport()
    loaded = [
        talent.to_dict()
        for talent in TalentLoader(DATA / "蒲公英达人数据_屿光精华.xlsx").load(report)
    ]
    state = preprocess_table({"talents": loaded, "issues": report.issues})
    assert any(item["nickname"] == "Luna的深夜书桌" for item in state["cleaned_talents"])
    luna = next(
        item for item in state["cleaned_talents"] if item["nickname"] == "Luna的深夜书桌"
    )
    assert "price_image" in luna["missing_fields"]
    assert "price_video" in luna["missing_fields"]

    policy = {
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
    }
    screened = screen_talents(
        {
            "approved_policy": policy,
            "cleaned_talents": state["cleaned_talents"],
            "issues": state["issues"],
        }
    )
    assert route_after_screen(screened) == "score"
    nicknames = {item["nickname"] for item in screened["screened_talents"]}
    assert "走走停停Vlog" not in nicknames
    assert "慢生活收藏家" not in nicknames
    assert "Mira的护肤手账" not in nicknames
    assert "Kiki的化妆台" in nicknames
    assert screened["deduped_talents"][0]["kept"] == "Kiki的化妆台"

    scored = {item["nickname"]: item for item in score_candidates(screened["screened_talents"], policy)}
    assert scored["麦浪MaiLang"]["band"] == "clear_yes"
    assert scored["麦浪MaiLang"]["tag"] == "进店好"
    assert scored["美丽研习社官方号"]["band"] == "clear_no"
    assert scored["美丽研习社官方号"]["tag"] == "数据差"
    assert scored["早八小满"]["band"] == "mid"
    assert scored["凌晨四点的厨房"]["score"] is None
    assert scored["凌晨四点的厨房"]["band"] == "mid"
    assert {item["band"] for item in scored.values()} == {"clear_yes", "clear_no", "mid"}
