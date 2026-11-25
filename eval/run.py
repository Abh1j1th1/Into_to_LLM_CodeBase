#!/usr/bin/env python
"""Run the benchmark with whatever model is written in eval/model.yaml.

  python eval/run.py check                     test that the model link in model.yaml works
  python eval/run.py dbbench-std 1             run first 1 sample under condition c0
  python eval/run.py os-std --condition c3     run all 144 OS samples under targeted verify gate
  python eval/run.py dbbench-std --condition c4 --limit 10

Optional settings in model.yaml: temperature (default 0), concurrency (default 1),
runs (default 1), controller (default http://localhost:5020/api), extra_body (dict).
"""
import argparse
import json
import os
import subprocess
import sys
import time

import requests
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))


def load_cfg(model_yaml=None):
    path = model_yaml or os.path.join(HERE, "model.yaml")
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    for key in ("url", "model"):
        if not cfg.get(key):
            sys.exit(f"{path}: '{key}' is missing")
    url = str(cfg["url"]).strip().rstrip("/")
    if url.endswith("/chat/completions"):
        url = url[: -len("/chat/completions")]
    if url.count("/") == 2:
        url += "/v1"
    cfg["url"] = url
    extra = dict(cfg.get("extra_body") or {})
    if cfg.get("thinking_off"):
        extra["reasoning_effort"] = "none"
    cfg["extra"] = extra
    return cfg


def check(cfg):
    headers = {
        "Authorization": "Bearer " + str(cfg.get("api_key", "placeholder")),
        "Content-Type": "application/json",
    }
    print(f"link : {cfg['url']}\nmodel: {cfg['model']}\n")
    try:
        r = requests.get(cfg["url"] + "/models", headers=headers, timeout=30)
        print(f"GET /models -> HTTP {r.status_code}")
        if r.status_code == 200:
            ids = [m.get("id") for m in r.json().get("data", [])]
            print("models available on that server:", ids)
            if cfg["model"] not in ids:
                print(f"WARNING: '{cfg['model']}' is not in that list (on Ollama run: ollama pull {cfg['model']})")
    except Exception as e:
        sys.exit(f"FAILED: cannot reach the link ({type(e).__name__}). Is the tunnel/server running?")
    body = {
        "model": cfg["model"],
        "max_tokens": 50,
        "temperature": 0,
        "messages": [{"role": "user", "content": "Reply with just: pong"}],
    }
    body.update(cfg["extra"])
    t = time.time()
    try:
        r = requests.post(cfg["url"] + "/chat/completions", headers=headers, json=body, timeout=95)
    except Exception as e:
        sys.exit(f"FAILED: chat request error ({type(e).__name__}).")
    dt = time.time() - t
    if r.status_code != 200:
        hint = " (timeout: model probably still loading, run check again)" if r.status_code == 524 else ""
        sys.exit(f"FAILED: chat -> HTTP {r.status_code}{hint}\n{r.text[:300]}")
    msg = r.json()["choices"][0]["message"]
    print(f"chat -> OK in {dt:.1f}s, reply: {msg.get('content')!r}")
    print("\nThe model link works.")


def main():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("target", help="'check' or task name (e.g. dbbench-std, os-std)")
    parser.add_argument("pos_limit", nargs="?", type=int, default=None, help="optional positional sample limit")
    parser.add_argument("--config", default=None, help="path to model yaml (default: eval/model.yaml)")
    parser.add_argument("--condition", default="c0", choices=["c0", "c1", "c2", "c3", "c4"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--memory-bank", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    if args.target == "check":
        return check(cfg)

    limit = args.limit if args.limit is not None else args.pos_limit
    runs = args.runs if args.runs is not None else cfg.get("runs", 1)
    temp = args.temperature if args.temperature is not None else cfg.get("temperature", 0)
    conc = args.concurrency if args.concurrency is not None else cfg.get("concurrency", 1)

    cmd = [
        sys.executable,
        os.path.join(HERE, "server_agent.py"),
        "-m",
        str(cfg["model"]),
        "-u",
        cfg["url"],
        "--model-api-key",
        str(cfg.get("api_key", "placeholder")),
        "-c",
        str(cfg.get("controller", "http://localhost:5020/api")),
        "-j",
        str(conc),
        "-n",
        str(runs),
        "-t",
        str(temp),
        "--condition",
        args.condition,
        "--extra-body",
        json.dumps(cfg["extra"]),
        "-T",
    ]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    if args.output_dir:
        cmd += ["-o", args.output_dir]
    if args.output_file:
        cmd += ["-f", args.output_file]
    if args.memory_bank:
        cmd += ["--memory-bank", args.memory_bank]
    if args.verbose:
        cmd += ["-v"]
    cmd.append(args.target)

    print(f"running: model={cfg['model']} task={args.target} condition={args.condition} temp={temp} runs={runs}")
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
