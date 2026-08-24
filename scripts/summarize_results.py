#!/usr/bin/env python3
"""Produce stable aggregate JSON/CSV from run records."""

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(args.results.rglob("run.json"))]
    rows = [{
        "run_name": row["run_name"], "status": row["status"], "exit_code": row["exit_code"],
        "model_logical_id": row["model_logical_id"],
        "checkpoint_logical_id": row["checkpoint_logical_id"],
        "started_at": row["started_at"], "ended_at": row["ended_at"],
    } for row in records]
    aggregate = {"schema_version": "1.0", "runs": rows}
    (args.results / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    with (args.results / "aggregate.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]) if rows else ["run_name", "status"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"runs": len(rows), "successful": sum(r["status"] == "success" for r in rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
