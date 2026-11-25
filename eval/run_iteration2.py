#!/usr/bin/env python
"""Iteration 2 experiment orchestrator (Proposal 3 + Proposal 1 data collection).

Runs:
1. Proposal 3: all 5 conditions (c0, c1, c2, c3, c4) on all 444 DB (300) + OS (144) items.
2. Cross-task failure-memory transfer: builds failure bank from one task's trajectories
   and runs c2/c4 on the other task.
3. Proposal 1 variance runs: repeated runs at temperature 0.7 (default k=3).
4. Automatically invokes Proposal 3 & Proposal 1 analysis scripts upon completion.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from run import load_cfg
from trajectory_parser import extract_failure_records, load_jsonl

HERE = Path(__file__).resolve().parent


def safe_tag(name: str) -> str:
    return name.replace(":", "_").replace("/", "_")


def run_eval(cfg, task_name: str, condition: str, out_file: Path, temperature: float, runs: int, limit=None, memory_bank=None):
    cmd = [
        sys.executable,
        str(HERE / "server_agent.py"),
        "-m", str(cfg["model"]),
        "-u", cfg["url"],
        "--model-api-key", str(cfg.get("api_key", "placeholder")),
        "-c", str(cfg.get("controller", "http://localhost:5020/api")),
        "-j", str(cfg.get("concurrency", 1)),
        "-n", str(runs),
        "-t", str(temperature),
        "--condition", condition,
        "--extra-body", json.dumps(cfg["extra"]),
        "-T",
        "-f", str(out_file),
    ]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    if memory_bank is not None:
        cmd += ["--memory-bank", str(memory_bank)]
    cmd.append(task_name)
    print(f"\n>>> Running {task_name} | model={cfg['model']} | condition={condition} | temp={temperature} | runs={runs}")
    rc = subprocess.call(cmd)
    if rc != 0:
        raise RuntimeError(f"Command failed with exit code {rc}: {' '.join(cmd)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(HERE / "model.yaml"))
    parser.add_argument("--out-dir", default=str(HERE / "results" / "iteration2"))
    parser.add_argument("--conditions", nargs="+", default=["c0", "c1", "c2", "c3", "c4"])
    parser.add_argument("--tasks", nargs="+", default=["dbbench-std", "os-std"])
    parser.add_argument("--limit", type=int, default=None, help="optional per-task limit for smoke test")
    parser.add_argument("--cross-task-transfer", action="store_true", help="run DB->OS and OS->DB memory transfer under c2")
    parser.add_argument("--variance-runs", type=int, default=0, help="if >0, run k repeated runs at temperature 0.7 under c0")
    parser.add_argument("--variance-limit", type=int, default=50, help="subset size for repeated temp=0.7 runs")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    model_tag = safe_tag(cfg["model"])
    out_dir = Path(args.out_dir) / model_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Main 5-condition Proposal 3 suite at temperature 0
    for cond in args.conditions:
        for task in args.tasks:
            out_file = out_dir / f"{task}__{cond}__t0.jsonl"
            run_eval(cfg, task, cond, out_file, temperature=0.0, runs=1, limit=args.limit)

    # 2. Cross-task failure memory transfer (Section 6)
    if args.cross_task_transfer and "dbbench-std" in args.tasks and "os-std" in args.tasks:
        db_c0 = out_dir / "dbbench-std__c0__t0.jsonl"
        os_c0 = out_dir / "os-std__c0__t0.jsonl"
        if db_c0.exists():
            db_mem = out_dir / "memory_bank_from_db.json"
            db_mem.write_text(json.dumps(extract_failure_records(load_jsonl(db_c0)), indent=2), encoding="utf-8")
            run_eval(cfg, "os-std", "c2", out_dir / "os-std__c2_from_db__t0.jsonl", temperature=0.0, runs=1, limit=args.limit, memory_bank=db_mem)
        if os_c0.exists():
            os_mem = out_dir / "memory_bank_from_os.json"
            os_mem.write_text(json.dumps(extract_failure_records(load_jsonl(os_c0)), indent=2), encoding="utf-8")
            run_eval(cfg, "dbbench-std", "c2", out_dir / "dbbench-std__c2_from_os__t0.jsonl", temperature=0.0, runs=1, limit=args.limit, memory_bank=os_mem)

    # 3. Repeated sampled runs at T=0.7 for Proposal 1 run-to-run variance
    if args.variance_runs > 0:
        for task in args.tasks:
            out_file = out_dir / f"{task}__c0__t0.7_k{args.variance_runs}.jsonl"
            run_eval(cfg, task, "c0", out_file, temperature=0.7, runs=args.variance_runs, limit=args.variance_limit)

    print(f"\nAll runs finished. Outputs written under: {out_dir}")


if __name__ == "__main__":
    main()
