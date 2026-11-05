"""推荐表Excel导出：支持人工审核预览和审核通过后的交付，不生成内容建议。"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


COLUMNS = (
    ("xhs_id", "达人 ID"), ("nickname", "达人昵称"),
    ("tag", "推荐原因标签"), ("sentence", "推荐理由"),
    ("format", "合作形式"), ("quantity", "投放条数"),
    ("unit_price", "拟定单价（元）"), ("total_price", "小计（元）"),
    ("angle", "内容角度"), ("sub_angle", "子角度"),
    ("details", "内容建议"), ("warnings", "待核实事项"),
    ("remarks", "备注"),
)

# 描述业务来源，不代表这些步骤当前均已实现或已联调。
FIELD_SOURCES = {
    "remarks": "第10步Python读取当前达人图文、视频报价，记录缺失或无效报价；不由模型生成",
    "xhs_id": "第1步读达人表；第9步选定名单后传递",
    "nickname": "第1步读达人表；第9步传递",
    "tag": "第6步明确档代码生成；第7步中间档模型生成；第8步核对/触发重写",
    "sentence": "第6步明确档代码生成；第7步中间档模型生成；第8步核对/触发重写",
    "format": "第9步确定本次合作形式",
    "quantity": "第9步确定投放条数",
    "unit_price": "第9步拟定价格，不直接等同于平台参考价",
    "total_price": "第9步提供小计（可选）；第12步按单价×条数计算或核对",
    "angle": "第10步生成；第11步人工审核",
    "sub_angle": "第10步生成；第11步人工审核",
    "details": "第10步生成并检查客户需求一致性；第11步人工审核",
    "warnings": "第1步加载问题、第3步未核实检查项、第8步审核/缺数提示、第9步方案提示及第10步内容提示；随第11步审核后由主流程合并，第12步展示",
}


def _money(value: Any, field: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{field} 必须为非负金额")
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError(f"{field} 必须为非负金额") from None
    if not number.is_finite() or number < 0 or number != number.quantize(Decimal("0.01")):
        raise ValueError(f"{field} 必须为非负金额，最多两位小数")
    return number


def flatten_recommendation(row: dict[str, Any]) -> dict[str, Any]:
    """将第十步保留的嵌套资料展平为导出字段，不修改输入或生成内容。

    已有顶层字段优先（包括人工清空的值）；仅缺键时读取嵌套字段。
    上游完整对象仍保留在结果中，Excel 仅按 COLUMNS 选列。
    """
    item = dict(row)
    for container, fields in (("talent", ("nickname",)),
                              ("evaluation", ("tag", "sentence"))):
        nested = item.get(container, {})
        if nested is None:
            nested = {}
        if not isinstance(nested, dict):
            raise ValueError(f"{container} 必须为字典或空值")
        for field in fields:
            if field not in item and field in nested:
                item[field] = nested[field]
    return item


def prepare_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """校验已合并的表格行；不改上游价格、不补造理由或内容。"""
    if not isinstance(rows, list) or not rows:
        raise ValueError("推荐数据必须为非空列表")
    result = []
    seen = set()
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"第 {index} 行必须为对象")
        item = flatten_recommendation(row)
        for field in ("xhs_id", "nickname"):
            if not isinstance(item.get(field), str) or not item[field].strip():
                raise ValueError(f"第 {index} 行缺少字符串字段 {field}")
        if item.get("format") not in {"image_text", "video"}:
            raise ValueError(f"第 {index} 行合作形式无效")
        key = (item["xhs_id"], item["format"])
        if key in seen:
            raise ValueError(f"重复投放项：{key}")
        seen.add(key)
        quantity = item.get("quantity")
        if type(quantity) is not int or quantity <= 0:
            raise ValueError(f"第 {index} 行投放条数必须为正整数")
        unit = _money(item.get("unit_price"), "unit_price")
        total = unit * quantity
        if item.get("total_price") is not None and _money(item["total_price"], "total_price") != total:
            raise ValueError(f"第 {index} 行小计与单价乘条数不一致，请先核实")
        item.update(unit_price=unit, total_price=total)
        for field in ("tag", "sentence", "angle", "sub_angle", "details"):
            if item.get(field) is not None and not isinstance(item[field], str):
                raise ValueError(f"第 {index} 行 {field} 必须为字符串或空值")
        remarks = item.get("remarks", [])
        if not isinstance(remarks, list) or any(not isinstance(value, str) for value in remarks):
            raise ValueError(f"第 {index} 行 remarks 必须为字符串列表")
        item["remarks"] = list(dict.fromkeys(value.strip() for value in remarks if value.strip()))
        warnings = item.get("warnings", [])
        if not isinstance(warnings, list) or any(not isinstance(w, str) for w in warnings):
            raise ValueError(f"第 {index} 行 warnings 必须为字符串列表")
        warnings = list(warnings)
        missing = [label for field, label in COLUMNS
                   if field in ("tag", "sentence", "angle", "sub_angle", "details")
                   and not (item.get(field) or "").strip()]
        for label in missing:
            warnings.append(f"缺失：{label}")
        if not item.get("tag") or not item.get("sentence"):
            warnings.append("未附完整推荐标签或理由")
        if any(not item.get(field) for field in ("angle", "sub_angle", "details")):
            warnings.append("内容建议未补全")
        item["warnings"] = list(dict.fromkeys(w for w in warnings if w.strip()))
        result.append(item)
    return result


def export_recommendations(
    rows: list[dict[str, Any]],
    output_path: str | Path,
    *,
    review_status: str,
    overwrite: bool = False,
    preview: bool = False,
) -> Path:
    """导出含备注栏的推荐表；review_status 须由第十一步传入 approved。

    本函数不执行人工审核，调用方负责传入真实审核状态与已合并的数据。
    warnings 应已包含需要随表展示的上游提示；完整审核附件由主流程保留。
    """
    if not preview and review_status != "approved":
        raise ValueError("只有人工审核通过（approved）的数据可导出客户推荐表")
    records = prepare_rows(rows)
    path = Path(output_path)
    if path.suffix.lower() != ".xlsx":
        raise ValueError("输出文件必须为 .xlsx")
    if path.exists() and not overwrite:
        raise FileExistsError(f"文件已存在：{path}")
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.comments import Comment

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "未审核预览" if preview else "达人推荐表"
    sheet.append([label for _, label in COLUMNS])
    for record in records:
        values = []
        for field, _ in COLUMNS:
            value = record.get(field)
            if field == "format":
                value = {"image_text": "图文", "video": "视频"}[value]
            elif field in {"warnings", "remarks"}:
                value = "\n".join(value)
            elif isinstance(value, Decimal):
                value = float(value)
            if isinstance(value, str) and len(value) > 32767:
                raise ValueError(f"{field} 超过 Excel 单元格长度限制")
            values.append(value)
        sheet.append(values)
        # 强制外部文字作为文本，避免以 = 开头的内容被当成公式。
        for cell, value in zip(sheet[sheet.max_row], values):
            if isinstance(value, str):
                cell.data_type = "s"
            cell.alignment = Alignment(vertical="top", wrap_text=True)
        for column in (7, 8):
            sheet.cell(sheet.max_row, column).number_format = '#,##0.00'
        sheet.cell(sheet.max_row, 1).number_format = "@"
    for cell, (field, _) in zip(sheet[1], COLUMNS):
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="305496")
        cell.comment = Comment(FIELD_SOURCES[field], "字段来源")
    sources = workbook.create_sheet("字段来源")
    sources.append(["列名", "字段名", "生成步骤"])
    for field, label in COLUMNS:
        sources.append([label, field, FIELD_SOURCES[field]])
    sources.column_dimensions["A"].width = 22
    sources.column_dimensions["B"].width = 24
    sources.column_dimensions["C"].width = 95
    for cells in sources:
        for cell in cells:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
    sources.freeze_panes = "A2"
    if preview:
        note = workbook.create_sheet("预览说明", 0)
        note.append(["状态", "未审核预览，不作为客户正式交付"])
        note.append(["输入审核状态", str(review_status)])
        note.append(["说明", "本模式仅用于人工检查表格。是否为模拟数据须结合来源和待核实事项核对。"])
        note.column_dimensions["A"].width = 22
        note.column_dimensions["B"].width = 100
        for cells in note:
            for cell in cells:
                cell.data_type = "s"
                cell.alignment = Alignment(wrap_text=True, vertical="top")
    widths = (18, 20, 18, 42, 14, 12, 20, 18, 18, 20, 60, 50, 40)
    from openpyxl.utils import get_column_letter
    for index, width in enumerate(widths, 1):
        sheet.column_dimensions[get_column_letter(index)].width = width
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    path.parent.mkdir(parents=True, exist_ok=True)
    # 默认独占创建，防止意外覆盖已有交付文件。
    with path.open("wb" if overwrite else "xb") as stream:
        workbook.save(stream)
    return path
