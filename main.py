from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from langgraph.types import Command

from agent.graph import build_graph
from agent.domain import FIELD_LABELS, PolicyValidationError
from agent.reasons import BAND_LABELS
from agent.requirements import LLMServiceError

BASE = Path(__file__).parent
STATIC = BASE / "static"

OPERATOR_LABELS = {
    "eq": "等于",
    "ne": "不等于",
    "gt": "高于",
    "gte": "不低于",
    "lt": "低于",
    "lte": "不高于",
    "contains": "包含",
    "not_contains": "不包含",
    "in": "属于",
    "not_in": "不属于",
    "is_missing": "数据缺失",
    "not_missing": "数据完整",
    "maximize": "越高越优先",
    "minimize": "越低越优先",
}


def print_section(title: str) -> None:
    """输出清晰的章节标题。"""
    print(f"\n{'─' * 12} {title} {'─' * 12}")


def format_value(value: object, default: str = "暂未提供") -> str:
    """将缺失或空白的信息转换为易读的提示。"""
    if value is None or value == "":
        return default
    if isinstance(value, (list, tuple, set)):
        return "、".join(str(item) for item in value)
    return str(value)


def format_money(value: object, default: str = "未知") -> str:
    if not isinstance(value, (int, float)):
        return default
    return f"¥{value:,.0f}"


def format_ratio(value: object, default: str = "未知") -> str:
    if not isinstance(value, (int, float)):
        return default
    return f"{value * 100:.1f}%"


def humanize_text(text: object) -> str:
    """隐藏模型内部使用的字段词，避免把机器语言展示给顾问。"""
    result = str(text or "")
    replacements = {
        "manual_checks": "人工核对项",
        "criteria": "评分条件",
        "maximize/minimize": "按指标高低排序",
        "maximize": "越高越优先",
        "minimize": "越低越优先",
        "顾问暂定口径": "依据现有信息采用的处理方式",
        "后续要补齐": "投放前需要处理",
        "硬性筛选": "排除条件",
        "加权评分": "综合比较",
    }
    for source, target in replacements.items():
        result = result.replace(source, target)
    return result


def describe_criterion(criterion: dict, index: int) -> str:
    """把机器条件翻译成顾问能直接阅读的自然语言。"""
    field = FIELD_LABELS.get(
        str(criterion.get("field")), str(criterion.get("field") or "未指定指标")
    )
    operator_key = str(criterion.get("operator") or "")
    value = criterion.get("value")
    reason = humanize_text(criterion.get("reason") or "")

    if criterion.get("hard"):
        inverse_operators = {
            "eq": "不等于",
            "ne": "等于",
            "gt": "不高于",
            "gte": "低于",
            "lt": "不低于",
            "lte": "高于",
            "contains": "不包含",
            "not_contains": "包含",
            "in": "不属于",
            "not_in": "属于",
            "is_missing": "数据完整",
            "not_missing": "数据缺失",
        }
        inverse = inverse_operators.get(operator_key, "不符合要求：")
        if operator_key in {"is_missing", "not_missing"}:
            condition = f"{field}{inverse}"
        else:
            condition = f"{field}{inverse}{format_value(value)}"
        sentence = f"{index}. 排除{condition}的达人"
    else:
        operator = OPERATOR_LABELS.get(operator_key, operator_key or "未指定条件")
        if operator_key in {"maximize", "minimize"}:
            comparison = f"{field}{operator}"
        elif operator_key in {"is_missing", "not_missing"}:
            comparison = f"优先考虑{field}{operator}的达人"
        else:
            comparison = (
                f"符合“{field}{operator}{format_value(value)}”的达人优先"
            )
        weight = criterion.get("weight")
        weight_text = (
            f"{weight * 100:g}%" if isinstance(weight, (int, float)) else "未提供"
        )
        sentence = f"{index}. {comparison}，权重 {weight_text}"

    if reason:
        sentence += f"。依据：{reason}"
    else:
        sentence += "。依据：尚未说明，需要补充采用这一条件的原因"
    return sentence


def save_result(result: dict) -> Path:
    """保存完整状态，供恢复流程和后续节点读取。"""
    out = BASE / "output" / "step7_recommendations.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = (
        "brief",
        "talents",
        "issues",
        "requirement_profile",
        "clarification_questions",
        "client_answers",
        "screening_policy",
        "policy_review_status",
        "policy_review_feedback",
        "policy_revision_count",
        "approved_policy",
        "active_policy",
        "screen_relax_count",
        "screen_log",
        "excluded_talents",
        "deduped_talents",
        "evaluated_talents",
        "next_step",
        "workflow_status",
    )
    out.write_text(
        json.dumps(
            {key: result.get(key) for key in keys},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return out


def render_result(result: dict) -> None:
    """把当前暂停点所需的信息展示给顾问。"""
    brief = result["brief"]
    talents = result["talents"]
    issues = result["issues"]
    profile = result.get("requirement_profile", {})

    # 一、客户投放需求
    print_section("本次投放需求")

    print(f"品牌：{format_value(brief.get('brand'))}")
    print(f"推广产品：{format_value(brief.get('product'))}")
    print(f"投放预算：{format_money(brief.get('total_budget'), '暂未提供')}")
    print(f"投放周期：{format_value(brief.get('campaign_period'))}")
    print(f"核心目标：{format_value(brief.get('kpi'))}")

    # 二、达人资源概况
    print_section("达人资源概况")
    print(f"本次共读取 {len(talents)} 位达人，先看前 5 位：")

    if talents:
        for talent in talents[:5]:
            print(
                f"  · {talent.get('nickname', '未命名')}："
                f"图文 {format_money(talent.get('price_image'))}，"
                f"视频 {format_money(talent.get('price_video'))}，"
                f"点击率 {format_ratio(talent.get('ctr'))}，"
                f"粉丝 {format_value(talent.get('fans_wan'), '未知')} 万"
            )
    else:
        print("暂未读取到达人数据，请检查达人数据文件。")

    # 三、数据检查
    print_section("数据检查")

    if issues:
        print(f"发现 {len(issues)} 项需要留意的信息：")
        for issue in issues:
            print(f"  · {issue}")
    else:
        print("当前未记录需要特别说明的数据问题。")

    # 四、需求分析
    print_section("投放需求解读")

    hard_requirements = profile.get("hard_requirements", [])
    soft_preferences = profile.get("soft_preferences", [])
    exclusions = profile.get("exclusions", [])
    campaign_constraints = profile.get("campaign_constraints", [])
    consultant_assumptions = profile.get("consultant_assumptions", [])
    downstream_dependencies = profile.get("downstream_dependencies", [])

    def print_items(title: str, items: list) -> None:
        if not items:
            return
        print(f"\n{title}")
        for item in items:
            print(f"  · {item}")

    summary = profile.get("summary")
    if summary:
        print(humanize_text(summary))

    questions = result.get("clarification_questions", [])
    if result.get("workflow_status") == "needs_client_confirmation":
        print_section("需要客户确认")
        print(
            "下面的信息会改变达人是否入选、内容是否合规或最终如何验收，"
            "收到客户答复后才能继续生成筛选方案："
        )
        if questions:
            for question in questions:
                print(f"  · {humanize_text(question)}")
        else:
            print("  · 需求分析没有给出具体问题，暂时不能向客户发起确认。")

    print_section("目前可以继续处理的事项")
    ready_items = (
        campaign_constraints
        + hard_requirements
        + soft_preferences
        + exclusions
    )
    print_items("需求单已经明确", ready_items)
    print_items("依据现有信息可以这样处理", consultant_assumptions)
    if not ready_items and not consultant_assumptions:
        print("目前没有足够的已知信息可用于筛选，需要先检查需求单内容。")

    if downstream_dependencies:
        print_section("投放执行前需要处理的事项")
        for item in downstream_dependencies:
            print(f"  · {humanize_text(item)}")

    # 五、筛选标准
    if result.get("workflow_status") != "needs_client_confirmation":
        policy = result.get("screening_policy", {})
        criteria = policy.get("criteria", [])
        exclusion_criteria = [item for item in criteria if item.get("hard")]
        scoring_criteria = [item for item in criteria if not item.get("hard")]

        print_section("先排除哪些达人")
        if exclusion_criteria:
            for index, criterion in enumerate(exclusion_criteria, start=1):
                print(f"  {describe_criterion(criterion, index)}")
        else:
            print("需求单和现有字段没有形成可自动执行的排除条件。")

        print_section("如何比较剩余达人")
        if scoring_criteria:
            for index, criterion in enumerate(scoring_criteria, start=1):
                print(f"  {describe_criterion(criterion, index)}")
            total_weight = sum(
                item.get("weight", 0)
                for item in scoring_criteria
                if isinstance(item.get("weight"), (int, float))
            )
            print(f"\n以上评分权重合计：{total_weight * 100:g}%")
        else:
            print("目前没有合法的评分指标，方案不能提交审核。")

        manual_checks = policy.get("manual_checks", [])
        if manual_checks:
            print("\n现有表格无法直接判断，还需要逐一核对：")
            for item in manual_checks:
                print(f"  · {humanize_text(item)}")

        rationale = policy.get("rationale")
        if rationale:
            print(f"\n为什么这样比较：{humanize_text(rationale)}")

        print("\n当前状态：筛选方案等待你的审核。")
    else:
        print("\n当前状态：等待客户答复，筛选方案尚未生成。")

    out = save_result(result)
    print(f"\n完整分析已保存至：{out}")


def render_shortlist(result: dict) -> None:
    """打印清洗、筛选、分档和已经写好的推荐理由。"""
    cleaned = result.get("cleaned_talents") or []
    missing_count = sum(1 for item in cleaned if item.get("missing_fields"))
    print_section("清洗结果")
    print(f"可计算达人 {len(cleaned)} 人，其中 {missing_count} 人有缺失字段。")

    print_section("筛选与去重")
    relax_count = int(result.get("screen_relax_count") or 0)
    if relax_count:
        print(f"数值硬条件放宽了 {relax_count} 次。")
    logs = result.get("screen_log") or []
    if logs:
        for line in logs:
            print(f"  · {line}")
    else:
        print("没有筛选记录。")

    evaluated = result.get("evaluated_talents") or []
    counts = {}
    for item in evaluated:
        band = str(item.get("band") or "")
        counts[band] = counts.get(band, 0) + 1
    print_section("分档结果")
    print(
        f"明确推荐 {counts.get('clear_yes', 0)} 人，"
        f"中间档 {counts.get('mid', 0)} 人，"
        f"明确不推荐 {counts.get('clear_no', 0)} 人。"
    )

    print_section("推荐名单")
    if not evaluated:
        print("没有进入打分的达人。")
    current_band = None
    for item in evaluated:
        band = str(item.get("band") or "")
        if band != current_band:
            current_band = band
            print(f"\n{BAND_LABELS.get(band, band)}")
        score = item.get("score")
        score_text = f"{score:.2f}" if isinstance(score, (int, float)) else "无法打分"
        tag = item.get("tag") or "未写标签"
        print(f"  · {item.get('nickname') or '未命名'}  {score_text}  {tag}")
        if item.get("error"):
            print(f"    这条理由没有生成：{item['error']}")
        elif item.get("sentence"):
            print(f"    {item['sentence']}")

    nxt = result.get("next_step") or {}
    if nxt.get("display_name"):
        print(f"\n下一步：{nxt['display_name']}。")


def get_interrupt_payload(result: dict) -> dict:
    interrupts = result.get("__interrupt__") or []
    if not interrupts:
        return {}
    value = getattr(interrupts[0], "value", {})
    return value if isinstance(value, dict) else {}


def main() -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": f"cli-{uuid4()}"}}
    result = graph.invoke(
        {
            "brief_path": str(STATIC / "甲方投放需求单_屿光精华.xlsx"),
            "talent_path": str(STATIC / "蒲公英达人数据_屿光精华.xlsx"),
            "policy_revision_count": 0,
        },
        config=config,
    )

    while result.get("__interrupt__"):
        render_result(result)
        payload = get_interrupt_payload(result)
        interrupt_type = payload.get("type")

        if interrupt_type == "client_confirmation":
            answer = input(
                "\n请把客户对上述问题的答复粘贴在这里，然后按回车继续：\n> "
            ).strip()
            if not answer:
                print("尚未收到客户答复，筛选没有继续。")
                return
            result = graph.invoke(
                Command(resume={"answers": answer}),
                config=config,
            )
            continue

        if interrupt_type == "policy_review":
            decision = input(
                "\n如果方案可以采用，请输入“同意”；如需调整，直接写下修改意见：\n> "
            ).strip()
            if not decision:
                print("你尚未同意或提出修改，方案仍在等待审核。")
                return

            if decision.lower() in {"同意", "批准", "通过", "approve", "yes", "y"}:
                review = {"action": "approve"}
            else:
                review = {"action": "revise", "feedback": decision}
                print_section("筛选方案需要修改")
                print("已收到你的修改意见，正在重新生成筛选方案。")
            result = graph.invoke(Command(resume=review), config=config)
            continue

        raise RuntimeError(f"无法识别的人工确认类型：{interrupt_type}")

    out = save_result(result)
    print_section("筛选方案审核通过")
    print("你已同意这套筛选方案，并已完成清洗、筛选、打分和中间档理由。")
    render_shortlist(result)
    print(f"\n本轮状态已保存至：{out}")


def run_cli() -> None:
    """运行命令行程序，并把可恢复的联网故障转成简短提示。"""
    try:
        main()
    except LLMServiceError as exc:
        print(f"\n这次没有连上模型服务：{exc}")
        print("确认网络恢复后重新运行即可，需求文件和达人数据不会受影响。")
        raise SystemExit(1) from None
    except PolicyValidationError as exc:
        print("\n这次生成的筛选方案没有通过完整性检查，因此没有提交给你审核：")
        for error in exc.errors:
            print(f"  · {error}")
        print("请重新运行；如果问题反复出现，需要检查需求内容或生成规则。")
        raise SystemExit(1) from None


if __name__ == "__main__":
    run_cli()
