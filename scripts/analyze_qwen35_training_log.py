from __future__ import annotations

import argparse
import json
import re
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_LOG = ROOT / "third_party/MindSpeed-MM/logs/qwen35_0.8B_20260608_110844.log"
ITERATION_PATTERN = re.compile(
    r"iteration\s+(?P<iteration>\d+)/\s*(?P<total>\d+).*?"
    r"elapsed time per iteration \(ms\):\s*(?P<elapsed>[0-9.]+).*?"
    r"global batch size:\s*(?P<gbs>\d+).*?"
    r"loss:\s*(?P<loss>[0-9.Ee+\-]+).*?"
    r"grad norm:\s*(?P<grad_norm>[0-9.Ee+\-]+)"
)
MEMORY_PATTERN = re.compile(
    r"memory \(MB\).*?allocated:\s*(?P<allocated>[0-9.]+).*?"
    r"max allocated:\s*(?P<max_allocated>[0-9.]+).*?"
    r"reserved:\s*(?P<reserved>[0-9.]+).*?max reserved:\s*(?P<max_reserved>[0-9.]+)"
)


def parse_training_log(path: Path) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    memory: list[dict[str, float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = ITERATION_PATTERN.search(line)
        if match:
            values = match.groupdict()
            steps.append(
                {
                    "iteration": int(values["iteration"]),
                    "total_iterations": int(values["total"]),
                    "elapsed_ms": float(values["elapsed"]),
                    "global_batch_size": int(values["gbs"]),
                    "loss": float(values["loss"]),
                    "grad_norm": float(values["grad_norm"]),
                }
            )
        memory_match = MEMORY_PATTERN.search(line)
        if memory_match:
            memory.append({key: float(value) for key, value in memory_match.groupdict().items()})
    return {"steps": steps, "memory_mb": memory}


def summarize(parsed: dict[str, Any], start_step: int, end_step: int) -> dict[str, Any]:
    steps = parsed["steps"]
    selected = [item for item in steps if start_step <= item["iteration"] <= end_step]
    summary: dict[str, Any] = {
        "parsed_steps": len(steps),
        "first_iteration": steps[0]["iteration"] if steps else None,
        "last_iteration": steps[-1]["iteration"] if steps else None,
        "declared_total_iterations": steps[-1]["total_iterations"] if steps else None,
        "measurement_window": {"start_step": start_step, "end_step": end_step, "sample_count": len(selected)},
        "memory_mb": parsed["memory_mb"],
    }
    if not selected:
        summary["performance"] = None
        return summary
    elapsed = [item["elapsed_ms"] for item in selected]
    global_batch_sizes = {item["global_batch_size"] for item in selected}
    global_batch_size = selected[0]["global_batch_size"] if len(global_batch_sizes) == 1 else None
    average_ms = statistics.fmean(elapsed)
    summary["performance"] = {
        "average_step_ms": average_ms,
        "median_step_ms": statistics.median(elapsed),
        "min_step_ms": min(elapsed),
        "max_step_ms": max(elapsed),
        "global_batch_size": global_batch_size,
        "samples_per_second": (global_batch_size * 1000.0 / average_ms) if global_batch_size else None,
    }
    summary["training_curve"] = {
        "first_loss": steps[0]["loss"],
        "last_loss": steps[-1]["loss"],
        "first_grad_norm": steps[0]["grad_norm"],
        "last_grad_norm": steps[-1]["grad_norm"],
    }
    return summary


def relative_error(actual: float, expected: float) -> float:
    denominator = abs(expected)
    if denominator < 1e-12:
        return abs(actual - expected)
    return abs(actual - expected) / denominator


def compare_logs(actual: dict[str, Any], reference: dict[str, Any], tolerance: float) -> dict[str, Any]:
    actual_by_step = {item["iteration"]: item for item in actual["steps"]}
    reference_by_step = {item["iteration"]: item for item in reference["steps"]}
    common_steps = sorted(set(actual_by_step) & set(reference_by_step))
    comparisons: dict[str, Any] = {}
    overall_passed = bool(common_steps)
    for metric in ("loss", "grad_norm"):
        errors = [relative_error(actual_by_step[step][metric], reference_by_step[step][metric]) for step in common_steps]
        metric_result = {
            "common_steps": len(common_steps),
            "mean_relative_error": statistics.fmean(errors) if errors else None,
            "max_relative_error": max(errors) if errors else None,
            "tolerance": tolerance,
            "passed": bool(errors) and max(errors) <= tolerance,
        }
        comparisons[metric] = metric_result
        overall_passed = overall_passed and metric_result["passed"]
    comparisons["passed"] = overall_passed
    return comparisons


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse and validate a MindSpeed-MM Qwen3.5 training log")
    parser.add_argument("log")
    parser.add_argument("--reference-log", default=str(REFERENCE_LOG))
    parser.add_argument("--no-reference", action="store_true")
    parser.add_argument("--start-step", type=int, default=50)
    parser.add_argument("--end-step", type=int, default=100)
    parser.add_argument("--relative-tolerance", type=float, default=0.02)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    log_path = Path(args.log).expanduser().resolve()
    if not log_path.is_file():
        parser.error(f"Log not found: {log_path}")
    parsed = parse_training_log(log_path)
    summary = summarize(parsed, args.start_step, args.end_step)
    expected_steps = summary["declared_total_iterations"] or 100
    complete = bool(parsed["steps"]) and summary["last_iteration"] == expected_steps and len(parsed["steps"]) >= expected_steps

    report: dict[str, Any] = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "log": str(log_path),
        "summary": summary,
        "complete": complete,
        "precision_comparison": None,
        "evidence_class": "project_npu_run" if "third_party/MindSpeed-MM/logs" not in str(log_path) else "official_reference_sample",
    }
    passed = complete or not args.require_complete
    if not args.no_reference:
        reference_path = Path(args.reference_log).expanduser().resolve()
        if not reference_path.is_file():
            parser.error(f"Reference log not found: {reference_path}")
        reference = parse_training_log(reference_path)
        report["reference_log"] = str(reference_path)
        report["precision_comparison"] = compare_logs(parsed, reference, args.relative_tolerance)
        passed = passed and bool(report["precision_comparison"]["passed"])
    report["status"] = "passed" if passed else "failed"
    report["truthfulness_boundary"] = (
        "The bundled reference log is not project NPU evidence. Only a new log produced on the rented Ascend instance "
        "can support a migration or performance claim."
    )

    if args.output:
        output = Path(args.output).expanduser()
        if not output.is_absolute():
            output = ROOT / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
