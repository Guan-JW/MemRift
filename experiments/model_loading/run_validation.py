#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path

try:
    from experiments.model_loading.driver_utils import (
        WorkerFailure,
        environment_record,
        prepare_outputs,
        run_supervised,
        utc_now,
        write_json,
    )
except ModuleNotFoundError:
    from driver_utils import (
        WorkerFailure,
        environment_record,
        prepare_outputs,
        run_supervised,
        utc_now,
        write_json,
    )


def build_commands(args, script_dir=None):
    script_dir = Path(script_dir or __file__).parent
    root = Path(args.output_root) / args.name
    online = root / "online.json"
    prequant = root / "prequant.json"
    worker = script_dir / "validate_qlora.py"
    common = ["--device", args.device]
    return root, [
        [
            args.python,
            str(worker),
            "--method",
            "online",
            "--model",
            args.model,
            "--output",
            str(online),
            *common,
        ],
        [
            args.python,
            str(worker),
            "--method",
            "prequant",
            "--model",
            args.model,
            "--checkpoint",
            str(Path(args.prepared) / "nf4"),
            "--output",
            str(prequant),
            *common,
        ],
        [
            args.python,
            str(script_dir / "compare_validation.py"),
            "--online",
            str(online),
            "--prequant",
            str(prequant),
            "--output",
            str(root / "comparison.json"),
        ],
    ]


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worker-timeout-seconds", type=float, default=3600)
    parser.add_argument("--min-mem-available-bytes", type=int, default=1024**3)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.worker_timeout_seconds < 0:
        parser.error("--worker-timeout-seconds must be nonnegative")
    if args.min_mem_available_bytes < 0:
        parser.error("--min-mem-available-bytes must be nonnegative")
    root, commands = build_commands(args)
    root.mkdir(parents=True, exist_ok=True)
    outputs = (
        root / "online.json",
        root / "online.logits.pt",
        root / "prequant.json",
        root / "prequant.logits.pt",
        root / "comparison.json",
        root / "driver.json",
    )
    try:
        prepare_outputs(outputs, args.overwrite)
    except (FileExistsError, IsADirectoryError) as error:
        parser.error(str(error))
    driver_path = root / "driver.json"
    driver = {
        "kind": "model_loading_validation",
        "status": "running",
        "cache_state": "warm",
        "cache_dropped": False,
        "started_at": utc_now(),
        "command": [sys.executable, str(Path(__file__)), *(argv if argv is not None else sys.argv[1:])],
        "environment": environment_record(),
        "commands": commands,
        "failures": [],
    }
    write_json(driver_path, driver)
    for index, command in enumerate(commands):
        try:
            run_supervised(
                command, args.worker_timeout_seconds, args.min_mem_available_bytes
            )
        except WorkerFailure as error:
            driver["status"] = "failed"
            driver["finished_at"] = utc_now()
            driver["failures"].append(
                {
                    "stage": ("online", "prequant", "comparison")[index],
                    "command": command,
                    "reason": error.reason,
                    "returncode": error.returncode,
                    "timestamp": utc_now(),
                }
            )
            write_json(driver_path, driver)
            raise
    driver["status"] = "complete"
    driver["finished_at"] = utc_now()
    driver["result_files"] = [path.name for path in outputs[:-1]]
    write_json(driver_path, driver)


if __name__ == "__main__":
    main()
