import argparse
import csv
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


FLOAT_RE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
INSTANCE_ID_RE = re.compile(r"instance_(\d+)")

GUROBI_FINAL_RE = re.compile(
    rf"Best objective\s+({FLOAT_RE}),\s*best bound\s+({FLOAT_RE}),\s*gap\s+({FLOAT_RE})%",
    re.IGNORECASE,
)
GUROBI_TIME_RE = re.compile(
    rf"Explored .*? in\s+({FLOAT_RE})\s+seconds",
    re.IGNORECASE,
)

SCIP_STATUS_RE = re.compile(r"SCIP Status\s*:\s*(.+)")
SCIP_TIME_RE = re.compile(rf"Solving Time \(sec\)\s*:\s*({FLOAT_RE})", re.IGNORECASE)
SCIP_PRIMAL_RE = re.compile(rf"Primal Bound\s*:\s*({FLOAT_RE})", re.IGNORECASE)
SCIP_DUAL_RE = re.compile(rf"Dual Bound\s*:\s*({FLOAT_RE})", re.IGNORECASE)
SCIP_GAP_RE = re.compile(rf"Gap\s*:\s*({FLOAT_RE})\s*%", re.IGNORECASE)


@dataclass
class InstanceResult:
    instance: str
    instance_id: int | None
    objective: float | None
    best_bound: float | None
    gap_pct: float | None
    runtime_s: float | None
    status: str
    solver: str
    source_file: str


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize per-instance evaluation results from a test log directory."
    )
    parser.add_argument("result_dir", type=Path, help="Directory containing instance log files")
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional path to save per-instance results as CSV",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Show top-k best and worst instances by objective value (default: %(default)s)",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Print the full per-instance table",
    )
    return parser.parse_args()


def find_log_files(result_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in result_dir.glob("*.log")
        if path.is_file() and path.name != "test.log"
    )


def extract_instance_id(path: Path) -> int | None:
    match = INSTANCE_ID_RE.search(path.stem)
    if match is None:
        return None
    return int(match.group(1))


def find_last_match(pattern: re.Pattern[str], text: str) -> re.Match[str] | None:
    matches = list(pattern.finditer(text))
    return matches[-1] if matches else None


def detect_gurobi_status(text: str) -> str:
    status_patterns = (
        ("OPTIMAL", r"Optimal solution found"),
        ("TIME_LIMIT", r"Time limit reached"),
        ("INFEASIBLE", r"Model is infeasible"),
        ("UNBOUNDED", r"Model is unbounded"),
        ("INF_OR_UNBD", r"Infeasible or unbounded"),
        ("INTERRUPTED", r"Interrupted"),
    )
    for status, pattern in status_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return status
    return "UNKNOWN"


def normalize_scip_status(raw_status: str | None) -> str:
    if not raw_status:
        return "UNKNOWN"

    text = raw_status.strip().lower()
    if "optimal" in text:
        return "OPTIMAL"
    if "time limit" in text:
        return "TIME_LIMIT"
    if "infeasible" in text and "unbounded" in text:
        return "INF_OR_UNBD"
    if "infeasible" in text:
        return "INFEASIBLE"
    if "unbounded" in text:
        return "UNBOUNDED"
    if "interrupt" in text:
        return "INTERRUPTED"
    return raw_status.strip()


def parse_gurobi_log(text: str, path: Path) -> InstanceResult:
    final_match = find_last_match(GUROBI_FINAL_RE, text)
    time_match = find_last_match(GUROBI_TIME_RE, text)

    objective = float(final_match.group(1)) if final_match else None
    best_bound = float(final_match.group(2)) if final_match else None
    gap_pct = float(final_match.group(3)) if final_match else None
    runtime_s = float(time_match.group(1)) if time_match else None

    return InstanceResult(
        instance=path.stem,
        instance_id=extract_instance_id(path),
        objective=objective,
        best_bound=best_bound,
        gap_pct=gap_pct,
        runtime_s=runtime_s,
        status=detect_gurobi_status(text),
        solver="GUROBI",
        source_file=path.name,
    )


def parse_scip_log(text: str, path: Path) -> InstanceResult:
    status_match = find_last_match(SCIP_STATUS_RE, text)
    time_match = find_last_match(SCIP_TIME_RE, text)
    primal_match = find_last_match(SCIP_PRIMAL_RE, text)
    dual_match = find_last_match(SCIP_DUAL_RE, text)
    gap_match = find_last_match(SCIP_GAP_RE, text)

    return InstanceResult(
        instance=path.stem,
        instance_id=extract_instance_id(path),
        objective=float(primal_match.group(1)) if primal_match else None,
        best_bound=float(dual_match.group(1)) if dual_match else None,
        gap_pct=float(gap_match.group(1)) if gap_match else None,
        runtime_s=float(time_match.group(1)) if time_match else None,
        status=normalize_scip_status(status_match.group(1) if status_match else None),
        solver="SCIP",
        source_file=path.name,
    )


def parse_log_file(path: Path) -> InstanceResult:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if "Gurobi" in text:
        return parse_gurobi_log(text, path)
    if "SCIP Status" in text or "SCIP version" in text:
        return parse_scip_log(text, path)
    raise ValueError(f"Unsupported solver log format: {path.name}")


def is_finite_number(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute percentile of empty list")
    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]

    lower_value = sorted_values[lower]
    upper_value = sorted_values[upper]
    fraction = position - lower
    return lower_value + (upper_value - lower_value) * fraction


def summarize_values(values: Iterable[float]) -> dict[str, float] | None:
    data = sorted(values)
    if not data:
        return None

    mean_value = sum(data) / len(data)
    variance = sum((value - mean_value) ** 2 for value in data) / len(data)

    return {
        "mean": mean_value,
        "std": math.sqrt(variance),
        "min": data[0],
        "q25": percentile(data, 0.25),
        "median": percentile(data, 0.50),
        "q75": percentile(data, 0.75),
        "max": data[-1],
    }


def infer_objective_sense(results: list[InstanceResult]) -> str:
    minimize_votes = 0
    maximize_votes = 0

    for item in results:
        if not is_finite_number(item.objective) or not is_finite_number(item.best_bound):
            continue

        objective = float(item.objective)
        best_bound = float(item.best_bound)
        if best_bound <= objective:
            minimize_votes += 1
        if best_bound >= objective:
            maximize_votes += 1

    if minimize_votes and minimize_votes >= maximize_votes:
        return "minimize"
    if maximize_votes:
        return "maximize"
    return "unknown"


def format_float(value: float | None, digits: int = 4) -> str:
    if not is_finite_number(value):
        return "NA"
    return f"{value:.{digits}f}"


def print_summary(results: list[InstanceResult], result_dir: Path) -> None:
    objectives = [item.objective for item in results if is_finite_number(item.objective)]
    gaps = [item.gap_pct for item in results if is_finite_number(item.gap_pct)]
    runtimes = [item.runtime_s for item in results if is_finite_number(item.runtime_s)]
    status_counts = Counter(item.status for item in results)

    print("=" * 88)
    print(f"Result directory: {result_dir}")
    print(f"Parsed instances: {len(results)}")
    print(f"Inferred objective sense: {infer_objective_sense(results)}")
    print("=" * 88)

    print("Status counts:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status:<16} {count}")

    for title, values in (
        ("Objective", objectives),
        ("Gap (%)", gaps),
        ("Runtime (s)", runtimes),
    ):
        summary = summarize_values(values)
        print()
        print(f"{title} statistics:")
        if summary is None:
            print("  No valid values found.")
            continue
        print(f"  mean   : {summary['mean']:.4f}")
        print(f"  std    : {summary['std']:.4f}")
        print(f"  min    : {summary['min']:.4f}")
        print(f"  q25    : {summary['q25']:.4f}")
        print(f"  median : {summary['median']:.4f}")
        print(f"  q75    : {summary['q75']:.4f}")
        print(f"  max    : {summary['max']:.4f}")


def objective_sort_key(item: InstanceResult) -> tuple[int, float]:
    if is_finite_number(item.objective):
        return (0, float(item.objective))
    return (1, float("inf"))


def instance_sort_key(item: InstanceResult) -> tuple[int, int, str]:
    if item.instance_id is None:
        return (1, 0, item.instance)
    return (0, item.instance_id, item.instance)


def print_ranked_lists(results: list[InstanceResult], top_k: int) -> None:
    valid = [item for item in results if is_finite_number(item.objective)]
    if not valid or top_k <= 0:
        return

    objective_sense = infer_objective_sense(results)
    reverse_best = objective_sense == "maximize"

    best_items = sorted(valid, key=objective_sort_key, reverse=reverse_best)[:top_k]
    worst_items = sorted(valid, key=objective_sort_key, reverse=not reverse_best)[:top_k]

    print()
    print(f"Best {len(best_items)} instances by objective:")
    for item in best_items:
        print(
            f"  instance_{item.instance_id if item.instance_id is not None else item.instance:<4} "
            f"obj={format_float(item.objective)} gap={format_float(item.gap_pct)}% "
            f"time={format_float(item.runtime_s, 2)}s"
        )

    print()
    print(f"Worst {len(worst_items)} instances by objective:")
    for item in worst_items:
        print(
            f"  instance_{item.instance_id if item.instance_id is not None else item.instance:<4} "
            f"obj={format_float(item.objective)} gap={format_float(item.gap_pct)}% "
            f"time={format_float(item.runtime_s, 2)}s"
        )


def print_full_table(results: list[InstanceResult]) -> None:
    rows = []
    for item in sorted(results, key=instance_sort_key):
        instance_label = (
            f"instance_{item.instance_id}" if item.instance_id is not None else item.instance
        )
        rows.append(
            (
                instance_label,
                format_float(item.objective),
                format_float(item.best_bound),
                format_float(item.gap_pct),
                format_float(item.runtime_s, 2),
                item.status,
                item.solver,
            )
        )

    headers = ("Instance", "Objective", "BestBound", "Gap(%)", "Time(s)", "Status", "Solver")
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    print()
    print("Per-instance results:")
    print("  " + "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  " + "  ".join("-" * width for width in widths))
    for row in rows:
        print("  " + "  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def save_csv(results: list[InstanceResult], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "instance",
                "instance_id",
                "objective",
                "best_bound",
                "gap_pct",
                "runtime_s",
                "status",
                "solver",
                "source_file",
            ],
        )
        writer.writeheader()
        for item in sorted(results, key=instance_sort_key):
            writer.writerow(
                {
                    "instance": item.instance,
                    "instance_id": item.instance_id,
                    "objective": item.objective,
                    "best_bound": item.best_bound,
                    "gap_pct": item.gap_pct,
                    "runtime_s": item.runtime_s,
                    "status": item.status,
                    "solver": item.solver,
                    "source_file": item.source_file,
                }
            )


def main() -> None:
    args = parse_arguments()
    result_dir = args.result_dir.expanduser().resolve()

    if not result_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {result_dir}")
    if not result_dir.is_dir():
        raise NotADirectoryError(f"Expected a directory: {result_dir}")

    log_files = find_log_files(result_dir)
    if not log_files:
        raise FileNotFoundError(f"No instance log files found in {result_dir}")

    results = []
    failures = []
    for log_file in log_files:
        try:
            results.append(parse_log_file(log_file))
        except Exception as exc:
            failures.append((log_file.name, str(exc)))

    print_summary(results, result_dir)
    print_ranked_lists(results, args.top_k)

    if args.show_all:
        print_full_table(results)

    if failures:
        print()
        print(f"Failed to parse {len(failures)} files:")
        for file_name, message in failures:
            print(f"  {file_name}: {message}")

    if args.csv is not None:
        csv_path = args.csv.expanduser().resolve()
        save_csv(results, csv_path)
        print()
        print(f"Saved CSV to: {csv_path}")


if __name__ == "__main__":
    main()
