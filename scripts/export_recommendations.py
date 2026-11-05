"""从第十一步保存的审核结果导出：python scripts/export_recommendations.py 输入.json 输出.xlsx"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.export_table import export_recommendations


def main() -> None:
    parser = argparse.ArgumentParser(description="第十二步：导出人工审核通过的推荐表")
    parser.add_argument("input", type=Path, help="包含 review_status 和 recommendations 的 JSON")
    parser.add_argument("output", type=Path)
    parser.add_argument("--preview", action="store_true", help="导出带未审核标记的预览，不作为正式交付")
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    path = export_recommendations(
        data["recommendations"], args.output,
        review_status=data.get("review_status", "pending"),
        preview=args.preview,
    )
    print(f"已导出：{path}")


if __name__ == "__main__":
    main()
