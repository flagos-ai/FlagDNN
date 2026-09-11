#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Summarize native runner records; never execute or time operator kernels."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True


def main() -> int:
    from run_tests_adapter import benchmark_speedup_summary

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--min-speedup", type=float, default=0.9)
    parser.add_argument("--note", default="")
    args = parser.parse_args()
    if not math.isfinite(args.min_speedup) or args.min_speedup <= 0:
        parser.error("--min-speedup must be positive and finite")
    document = json.loads(args.input.read_text())
    if (
        document.get("platform") != "thead"
        or document.get("schema_version") != 2
    ):
        parser.error("input must be a schema-v2 THead native runner result")
    results = document["results"]
    coverage = document.get("comparable_coverage")
    if not isinstance(coverage, dict):
        parser.error("input lacks comparable coverage accounting")
    cases = []
    operators = []
    for operator, suites in sorted(results.items()):
        benchmark = suites.get("benchmark", {})
        pairs = []
        if benchmark.get("status") == "passed":
            records = benchmark.get("records", {})
            for name, providers in sorted(records.items()):
                if set(providers) != {"flagdnn", "acdnn"}:
                    raise ValueError(
                        f"incomplete provider pair: {operator}/{name}"
                    )
                flagdnn = float(providers["flagdnn"]["median"])
                reference = float(providers["acdnn"]["median"])
                if not all(
                    math.isfinite(v) and v > 0 for v in (flagdnn, reference)
                ):
                    raise ValueError(f"invalid latency: {operator}/{name}")
                ratio = reference / flagdnn
                pairs.append(ratio)
                cases.append({"operator": operator, "case": name,
                              "flagdnn_median_us": flagdnn,
                              "acdnn_median_us": reference, "speedup": ratio,
                              "passed": ratio >= args.min_speedup})
        failed = sum(value < args.min_speedup for value in pairs)
        state = ("below_threshold" if failed else "passed") if pairs else (
            "reference_unavailable" if benchmark.get("status") == "skipped"
            else "not_measured")
        operators.append({
            "operator": operator, "status": state, "case_count": len(pairs),
            "failed_case_count": failed,
            "minimum_speedup": min(pairs) if pairs else None,
            "median_speedup": statistics.median(pairs) if pairs else None,
            "geometric_mean_speedup": (
                statistics.geometric_mean(pairs) if pairs else None
            ),
        })
    performance = benchmark_speedup_summary(
        results, args.min_speedup, coverage
    )
    report = {"source": str(args.input.resolve()),
              "source_timestamp_utc": document.get("timestamp_utc"),
              "note": args.note, "performance": performance,
              "operators": operators, "cases": cases}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "performance.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    for name, rows in (("operators", operators), ("cases", cases)):
        with (args.output_dir / f"{name}.csv").open("w", newline="") as output:
            if rows:
                writer = csv.DictWriter(output, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    lines = ["# THead 算子性能统计", "",
             f"来源：`{args.input.resolve()}`", "",
             f"来源时间：{document.get('timestamp_utc')}", "",
             "speedup = acDNN median / FlagDNN median；"
             f"逐 case 阈值：{args.min_speedup}。", "",
             f"可比较 case：{len(cases)}；低于阈值：{performance['failed_case_count']}；"
             f"性能门禁通过：{performance['gate_passed']}。", "",
             "没有可比较 case 的算子不计为达标。以下是输入记录的统计，不是新的真机测量。", ""]
    if args.note:
        lines += [args.note, ""]
    lines += ["| 算子 | case 数 | 未达标 | 最小 speedup | 几何均值 | 状态 |",
              "|---|---:|---:|---:|---:|---|"]
    for row in operators:
        minimum = f"{row['minimum_speedup']:.6f}" if row['case_count'] else "—"
        mean = (
            f"{row['geometric_mean_speedup']:.6f}"
            if row['case_count'] else "—"
        )
        lines.append(
            f"| {row['operator']} | {row['case_count']} | "
            f"{row['failed_case_count']} | {minimum} | {mean} | "
            f"{row['status']} |"
        )
    (args.output_dir / "performance.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"case_count": len(cases),
                      "failed_case_count": performance["failed_case_count"],
                      "failed_operator_count": sum(
                          r["failed_case_count"] > 0 for r in operators
                      ),
                      "gate_passed": performance["gate_passed"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
