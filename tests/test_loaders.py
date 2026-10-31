"""loaders 的独立测试：不启动图、不连 LLM，直接测领域层。

运行：python -m pytest tests/ -q   （或直接 python tests/test_loaders.py）
"""
from __future__ import annotations

from pathlib import Path

import pytest

from agent.domain import LoadReport, parse_number, parse_ratio
from agent.loaders import BriefLoader, TalentLoader

DATA = Path(__file__).parent.parent / "static"


def test_parse_number() -> None:
    assert parse_number(9500) == 9500.0
    assert parse_number("¥350,000（含达人合作费用，不含投流）") == 350000.0
    assert parse_number("-") is None
    assert parse_number(None) is None


def test_parse_ratio() -> None:
    assert parse_ratio(0.942) == 0.942
    assert parse_ratio("94.2%") == pytest.approx(0.942)


def test_load_brief_and_talents() -> None:
    report = LoadReport()
    brief = BriefLoader(DATA / "甲方投放需求单_屿光精华.xlsx").load(report)
    assert brief.brand.startswith("屿光")
    assert brief.total_budget == 350000.0

    talents = TalentLoader(DATA / "蒲公英达人数据_屿光精华.xlsx").load(report)
    assert len(talents) >= 15
    by_name = {t.nickname: t for t in talents}
    assert "桃桃吃梨" in by_name
    # CPV（元）不应被后面的「前选CPV优于行业top50%」列覆盖
    assert by_name["桃桃吃梨"].cpv == pytest.approx(0.99)
    assert by_name["美丽研习社官方号"].cpv == pytest.approx(2.87)
    # 模拟数据里故意埋了整行缺价的达人，loader 应该把它记进问题报告
    assert any("参考价均缺失" in i for i in report.issues)


if __name__ == "__main__":
    test_parse_number()
    test_parse_ratio()
    test_load_brief_and_talents()
    print("3 tests passed")
