#!/usr/bin/env python3
"""Recompute the 2026-11-17 Qwen2.5 update using only Python's standard library.

From the repository root:
    python 07_scripts/audit_qwen25.py
    python 07_scripts/audit_qwen25.py --output audit_qwen25_results.json

The default input is ../09_qwen25_update relative to this script's directory.
Only base/repro1/repro2 full test predictions support the new results. The
60-question files document a historical reproducibility check, not a new run.
No model inference or training is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

RUNS = ("base", "repro1", "repro2")
EXPECTED_UNITS = 577
EXPECTED_SINGLE = 406
EXPECTED_REPEATED = 171
HISTORICAL_FILES = (
    "repro_check_60.txt",
    "repro60_q25v3_records.jsonl",
    "repro60_records.log",
)
SUMMARY_METRICS = (
    "parse_rate", "overall", "single", "repeated", "binary", "exact"
)
# Archived fractions were rounded to four decimal places (trailing zeros omitted).
SUMMARY_TOLERANCE = 0.00005 + 1e-12


def is_integer(value: Any) -> bool:
    # Booleans are Python ints but are not valid count predictions.
    return isinstance(value, int) and not isinstance(value, bool)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_records(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path.name}:{number}: expected an object")
            unit = row.get("unit_id")
            if not isinstance(unit, str) or not unit:
                raise ValueError(f"{path.name}:{number}: missing unit_id")
            if unit in result:
                raise ValueError(f"{path.name}:{number}: duplicate unit_id {unit!r}")
            if not is_integer(row.get("count")) or row["count"] < 1:
                raise ValueError(f"{path.name}:{number}: invalid ground-truth count")
            if row.get("split") != "test":
                raise ValueError(f"{path.name}:{number}: expected test split")
            for field in ("video_id", "action"):
                if not isinstance(row.get(field), str) or not row[field]:
                    raise ValueError(f"{path.name}:{number}: missing {field}")
            result[unit] = row
    return result


def four_way_correct(row: dict[str, Any]) -> bool:
    prediction = row.get("pred")
    # Four buckets are 1, 2, 3, 4+. Zero is NOT clamped to one.
    # Missing, non-integer, zero, and negative predictions are errors.
    return (
        is_integer(prediction)
        and prediction >= 1
        and min(prediction, 4) == min(row["count"], 4)
    )


def compute(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = list(records.values())
    single = [row for row in rows if row["count"] == 1]
    repeated = [row for row in rows if row["count"] >= 2]
    parsed = [row for row in rows if is_integer(row.get("pred"))]
    counts = {
        "units": len(rows),
        "single_units": len(single),
        "repeated_units": len(repeated),
        "parsed": len(parsed),
        "positive_integer_predictions": sum(row["pred"] >= 1 for row in parsed),
        "nonpositive_integer_predictions": sum(row["pred"] <= 0 for row in parsed),
        "missing_or_noninteger_predictions": len(rows) - len(parsed),
        "overall_correct": sum(four_way_correct(row) for row in rows),
        "single_correct": sum(four_way_correct(row) for row in single),
        "repeated_correct": sum(four_way_correct(row) for row in repeated),
        "answer_one": sum(row["pred"] == 1 for row in parsed),
        "binary_correct": sum(
            (row["pred"] >= 2) == (row["count"] >= 2) for row in parsed
        ),
        "exact_correct": sum(row["pred"] == row["count"] for row in parsed),
    }
    metrics = {
        "parse_rate": counts["parsed"] / counts["units"],
        "overall": counts["overall_correct"] / counts["units"],
        "single": counts["single_correct"] / counts["single_units"],
        "repeated": counts["repeated_correct"] / counts["repeated_units"],
        "answer_one_rate": counts["answer_one"] / counts["units"],
        "binary": counts["binary_correct"] / counts["units"],
        "exact": counts["exact_correct"] / counts["units"],
    }
    return {
        "counts": counts,
        "fractions": metrics,
        "percentages": {key: round(value * 100, 6) for key, value in metrics.items()},
        "percentages_1dp": {key: round(value * 100, 1) for key, value in metrics.items()},
    }


def audit(folder: Path) -> dict[str, Any]:
    errors: list[str] = []
    runs: dict[str, Any] = {}
    records_by_run: dict[str, dict[str, dict[str, Any]]] = {}
    for name in RUNS:
        prediction_path = folder / f"{name}_test_predictions.jsonl"
        summary_path = folder / f"{name}_test_summary.json"
        records = read_records(prediction_path)
        n_single = sum(row["count"] == 1 for row in records.values())
        n_repeated = len(records) - n_single
        if (len(records), n_single, n_repeated) != (
            EXPECTED_UNITS, EXPECTED_SINGLE, EXPECTED_REPEATED
        ):
            errors.append(
                f"{name}: unexpected populations "
                f"{len(records)}/{n_single}/{n_repeated}"
            )
        if not records or not n_single or not n_repeated:
            raise ValueError(f"{name}: empty overall, single, or repeated population")
        records_by_run[name] = records
        result = compute(records)
        summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
        comparisons: dict[str, Any] = {}
        for key in ("units", "parsed"):
            actual = result["counts"][key]
            archived = summary.get(key)
            matches = is_integer(archived) and actual == archived
            comparisons[key] = {
                "recomputed": actual, "archived": archived, "matches": matches
            }
            if not matches:
                errors.append(f"{name}: summary mismatch for {key}")
        for key in SUMMARY_METRICS:
            actual = result["fractions"][key]
            archived = summary.get(key)
            valid_number = (
                isinstance(archived, (int, float))
                and not isinstance(archived, bool)
                and math.isfinite(archived)
            )
            matches = valid_number and abs(actual - archived) <= SUMMARY_TOLERANCE
            comparisons[key] = {
                "recomputed": actual, "archived": archived, "matches": matches
            }
            if not matches:
                errors.append(f"{name}: summary mismatch for {key}")
        if summary.get("split") != "test":
            errors.append(f"{name}: summary split is not test")
        result["summary_comparison"] = comparisons
        result["input_sha256"] = {
            prediction_path.name: sha256(prediction_path),
            summary_path.name: sha256(summary_path),
        }
        runs[name] = result

    baseline = records_by_run["base"]
    identities_match = True
    for name in ("repro1", "repro2"):
        candidate = records_by_run[name]
        if set(candidate) != set(baseline):
            identities_match = False
            errors.append(f"{name}: unit_id set differs from baseline")
        else:
            for unit in baseline:
                for field in ("count", "video_id", "action", "split"):
                    if candidate[unit][field] != baseline[unit][field]:
                        identities_match = False
                        errors.append(f"{name}: {field} differs for {unit!r}")
    return {
        "passed": not errors,
        "records_directory": str(folder.resolve()),
        "protocol": {
            "four_way": "Count buckets 1, 2, 3, 4+; zero is not clamped to one.",
            "missing_or_invalid": "Errors retained in every relevant denominator.",
            "parsed": "Integer prediction, including zero; bool is excluded.",
            "binary": "Integer prediction below 2 versus at least 2.",
            "answer_one_rate": "Number of integer predictions equal to 1 / all units.",
            "summary_precision": "Fractions rounded to 4 decimal places.",
            "historical_files_excluded": list(HISTORICAL_FILES),
            "not_recomputed": ["prior_conditional", "prior_conditional_n", "js_divergence"],
            "scope": "Arithmetic and identity audit only; no inference/training reproduction.",
        },
        "matching_unit_ids_and_ground_truth": identities_match,
        "runs": runs,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records", type=Path,
        default=Path(__file__).resolve().parents[1] / "09_qwen25_update",
        help="Directory containing the three full prediction files and summaries.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    options = parser.parse_args()
    try:
        report = audit(options.records)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        report = {"passed": False, "errors": [f"{type(exc).__name__}: {exc}"]}
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if options.output:
        options.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

