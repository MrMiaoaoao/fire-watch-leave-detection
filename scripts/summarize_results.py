"""Summarize and gate fire-watch leave-detection JSON reports."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path


EXPECTED_CURRENT = {
    "监火员离岗测试": {
        "total_alerts": 1,
        "warn_times": [96.6],
        "clear_times": [104.3],
    },
    "监火员离岗测试2": {
        "total_alerts": 1,
        "warn_times": [55.3],
        "clear_times": [94.0],
    },
    "监火员离岗测试3": {
        "total_alerts": 0,
        "warn_times": [],
        "clear_times": [],
    },
    "监火员离岗测试4": {
        "total_alerts": 4,
        "warn_times": [13.5, 81.3, 123.5, 152.3],
        "clear_times": [34.2, 100.9, 138.4, 166.1],
    },
}


@dataclass(frozen=True)
class ResultSummary:
    stem: str
    path: Path
    duration_s: float | None
    fps: float | None
    total_alerts: int
    warn_times: list[float]
    clear_times: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize outputs/*_result.json files")
    parser.add_argument("--outputs", default="outputs", help="Output directory containing *_result.json files")
    parser.add_argument("--include-probes", action="store_true", help="Include probe/smoke-test result files")
    parser.add_argument("--strict-current", action="store_true", help="Check the current four-video baseline")
    parser.add_argument("--tolerance", type=float, default=0.2, help="Allowed time difference in seconds")
    return parser.parse_args()


def load_summary(path: Path) -> ResultSummary:
    data = json.loads(path.read_text(encoding="utf-8"))
    monitor = data.get("leave_monitor", {})
    monitor_summary = monitor.get("summary", {})
    alerts = monitor.get("alerts", [])
    warn_times = [float(a["time_s"]) for a in alerts if a.get("type") == "warn"]
    clear_times = [float(a["time_s"]) for a in alerts if a.get("type") == "clear"]
    stem = path.name.removesuffix("_result.json")
    return ResultSummary(
        stem=stem,
        path=path,
        duration_s=monitor_summary.get("total_duration_s"),
        fps=monitor_summary.get("fps"),
        total_alerts=int(monitor_summary.get("total_alerts", len(warn_times))),
        warn_times=warn_times,
        clear_times=clear_times,
    )


def find_results(outputs: Path, include_probes: bool) -> list[ResultSummary]:
    results = []
    for path in sorted(outputs.glob("*_result.json")):
        if not include_probes and "probe" in path.stem.lower():
            continue
        results.append(load_summary(path))
    expected_order = {stem: index for index, stem in enumerate(EXPECTED_CURRENT)}
    return sorted(results, key=lambda item: (expected_order.get(item.stem, 999), item.stem))


def format_times(times: list[float]) -> str:
    if not times:
        return "无"
    return ", ".join(f"{t:.1f}s" for t in times)


def print_table(results: list[ResultSummary]) -> None:
    print("| result | alerts | warn_times | clear_times | duration | fps |")
    print("| --- | ---: | --- | --- | ---: | ---: |")
    for result in results:
        duration = "" if result.duration_s is None else f"{float(result.duration_s):.1f}s"
        fps = "" if result.fps is None else f"{float(result.fps):.3g}"
        print(
            f"| {result.stem} | {result.total_alerts} | "
            f"{format_times(result.warn_times)} | {format_times(result.clear_times)} | "
            f"{duration} | {fps} |"
        )


def times_match(actual: list[float], expected: list[float], tolerance: float) -> bool:
    if len(actual) != len(expected):
        return False
    return all(abs(a - e) <= tolerance for a, e in zip(actual, expected))


def check_current(results: list[ResultSummary], tolerance: float) -> list[str]:
    by_stem = {result.stem: result for result in results}
    errors = []
    for stem, expected in EXPECTED_CURRENT.items():
        result = by_stem.get(stem)
        if result is None:
            errors.append(f"missing result: {stem}")
            continue
        if result.total_alerts != expected["total_alerts"]:
            errors.append(f"{stem}: total_alerts {result.total_alerts} != {expected['total_alerts']}")
        if not times_match(result.warn_times, expected["warn_times"], tolerance):
            errors.append(f"{stem}: warn_times {result.warn_times} != {expected['warn_times']}")
        if not times_match(result.clear_times, expected["clear_times"], tolerance):
            errors.append(f"{stem}: clear_times {result.clear_times} != {expected['clear_times']}")
    return errors


def main() -> int:
    args = parse_args()
    outputs = Path(args.outputs)
    if not outputs.exists():
        print(f"outputs directory not found: {outputs}", file=sys.stderr)
        return 2
    results = find_results(outputs, args.include_probes)
    if not results:
        print(f"no *_result.json files found in: {outputs}", file=sys.stderr)
        return 2

    print_table(results)
    if args.strict_current:
        errors = check_current(results, args.tolerance)
        if errors:
            print("\nBaseline check failed:", file=sys.stderr)
            for error in errors:
                print(f"- {error}", file=sys.stderr)
            return 1
        print("\nBaseline check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
