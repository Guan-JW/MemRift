#!/usr/bin/env python3
"""Produce stable aggregate JSON/CSV from run records."""

import argparse
import csv
import json
from pathlib import Path


def scalar_fields(prefix, values):
    return {
        f"{prefix}{key}": value
        for key, value in values.items()
        if value is None or isinstance(value, (str, int, float, bool))
    }


def summarize_record(record):
    row = {
        "run_name": record["run_name"],
        "status": record["status"],
        "exit_code": record["exit_code"],
        "model_logical_id": record["model_logical_id"],
        "checkpoint_logical_id": record["checkpoint_logical_id"],
        "started_at": record["started_at"],
        "ended_at": record["ended_at"],
    }
    row.update(scalar_fields("environment.", record.get("environment") or {}))
    row.update(scalar_fields("result.", record.get("result") or {}))
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    records = []
    for path in sorted(args.results.rglob("run.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if all(key in record for key in ("run_name", "model_logical_id", "checkpoint_logical_id")):
            records.append(record)
    rows = [summarize_record(record) for record in records]
    aggregate = {"schema_version": "1.0", "runs": rows}
    (args.results / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    with (args.results / "aggregate.csv").open("w", newline="", encoding="utf-8") as output:
        fieldnames = sorted({key for row in rows for key in row}) if rows else ["run_name", "status"]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"runs": len(rows), "successful": sum(r["status"] == "success" for r in rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
