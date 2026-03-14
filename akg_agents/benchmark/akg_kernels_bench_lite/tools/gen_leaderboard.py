#!/usr/bin/env python3
"""从 results/ 目录下的成绩 JSON 文件生成排行榜。

用法:
    python gen_leaderboard.py [--results-dir results/] [--output leaderboard.json]

输出:
    - leaderboard.json: 结构化排行榜数据
    - 终端同时打印 markdown 表格
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parent.parent


def load_results(results_dir: Path) -> list[dict]:
    """加载目录下所有 .json 结果文件。"""
    reports = []
    for f in sorted(results_dir.glob("*.json")):
        if f.name == "leaderboard.json":
            continue
        try:
            with open(f) as fh:
                data = json.load(fh)
                if "team_name" in data and "summary" in data:
                    reports.append(data)
        except (json.JSONDecodeError, KeyError):
            print(f"[WARN] 跳过无效文件: {f}")
    return reports


def build_leaderboard(reports: list[dict]) -> dict:
    """构建排行榜数据结构。"""
    entries = []
    for r in reports:
        summary = r["summary"]
        cases_by_tier: dict[str, list[dict]] = {}
        for c in r.get("cases", []):
            tier = c.get("tier", "unknown")
            cases_by_tier.setdefault(tier, []).append({
                "case": c["case"],
                "status": c["status"],
                "speedup": c.get("speedup", 0.0),
                "weighted_score": c.get("weighted_score", 0.0),
            })

        entries.append({
            "team_name": r["team_name"],
            "institution": r.get("meta", {}).get("institution", ""),
            "total_weighted_score": summary["total_weighted_score"],
            "passed": summary["passed"],
            "total": summary["total"],
            "avg_speedup": summary["avg_speedup"],
            "device": r.get("device", ""),
            "timestamp": r.get("timestamp", ""),
            "cases_by_tier": cases_by_tier,
        })

    entries.sort(key=lambda e: e["total_weighted_score"], reverse=True)
    for i, e in enumerate(entries):
        e["rank"] = i + 1

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_teams": len(entries),
        "ranking": entries,
    }


def print_markdown_table(leaderboard: dict):
    """在终端打印 markdown 格式排行榜。"""
    ranking = leaderboard["ranking"]
    if not ranking:
        print("暂无数据")
        return

    print()
    print(f"## AKG Bench Lite 排行榜")
    print(f"> 生成时间: {leaderboard['generated_at']}")
    print(f"> 参赛队伍: {leaderboard['total_teams']}")
    print()
    print("| Rank | Team | Institution | Score | Passed | Avg Speedup |")
    print("|-----:|------|-------------|------:|-------:|------------:|")
    for e in ranking:
        print(
            f"| {e['rank']} "
            f"| {e['team_name']} "
            f"| {e['institution'] or '-'} "
            f"| {e['total_weighted_score']:.1f} "
            f"| {e['passed']}/{e['total']} "
            f"| {e['avg_speedup']:.2f}x |"
        )

    # 各 tier 明细
    all_tiers = set()
    for e in ranking:
        all_tiers.update(e["cases_by_tier"].keys())

    for tier in sorted(all_tiers):
        print()
        print(f"### {tier.upper()} 详细")
        print()

        case_names = set()
        for e in ranking:
            for c in e["cases_by_tier"].get(tier, []):
                case_names.add(c["case"].split("/", 1)[-1])
        case_names_sorted = sorted(case_names)

        header = "| Team | " + " | ".join(case_names_sorted) + " |"
        sep = "|------|" + "|".join(["------:" for _ in case_names_sorted]) + "|"
        print(header)
        print(sep)

        for e in ranking:
            tier_cases = {
                c["case"].split("/", 1)[-1]: c
                for c in e["cases_by_tier"].get(tier, [])
            }
            row = f"| {e['team_name']} "
            for cn in case_names_sorted:
                if cn in tier_cases:
                    c = tier_cases[cn]
                    if c["status"] == "pass":
                        row += f"| {c['speedup']:.2f}x "
                    else:
                        row += "| FAIL "
                else:
                    row += "| - "
            row += "|"
            print(row)

    print()


def main():
    parser = argparse.ArgumentParser(description="生成排行榜")
    parser.add_argument(
        "--results-dir",
        default=str(BENCH_ROOT / "results"),
        help="成绩 JSON 目录 (默认: bench_lite/results/)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="排行榜输出路径 (默认: results_dir/leaderboard.json)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    if not results_dir.exists():
        print(f"[ERROR] 目录不存在: {results_dir}")
        sys.exit(1)

    reports = load_results(results_dir)
    if not reports:
        print(f"[ERROR] 未找到有效的成绩文件: {results_dir}")
        sys.exit(1)

    leaderboard = build_leaderboard(reports)

    output_path = Path(args.output) if args.output else results_dir / "leaderboard.json"
    with open(output_path, "w") as f:
        json.dump(leaderboard, f, indent=2, ensure_ascii=False)
    print(f"[OK] 排行榜已写入: {output_path}")

    print_markdown_table(leaderboard)


if __name__ == "__main__":
    main()
