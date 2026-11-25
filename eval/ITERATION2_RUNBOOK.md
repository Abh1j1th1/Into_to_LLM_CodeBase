# Iteration 2 Execution Runbook (Group 12)

This branch (`iteration2.0`) implements the complete Iteration 2 pipeline planned in `Task4_Iteration1_Group12.pdf`:
- **Proposal 3 (Refined Failure-Aware Agent)**: 5 experimental conditions (`c0`–`c4`) + cross-task memory transfer + paired McNemar & `<=1`-command diagnostics.
- **Proposal 1 (Trajectory-Aware Bayesian IRT)**: shared trajectory parser (`eval/trajectory_parser.py`) + Bayesian 1PL/2PL/Trajectory-Aware 2PL/Graded IRT (`eval/irt.py`) + calibration-subset selection, bootstrap rank intervals, MAE/RMSE vs full set, 5-fold CV, and $T=0.7$ variance quantification (`eval/analyze_proposal1.py`).

---

## 1. Prerequisites on Host

1. **Docker storage**: Ensure the Docker root partition has enough free space (or point `/etc/docker/daemon.json` `"data-root"` to a drive with >15 GB free) for `jingbh/agentrl-controller`, `redis:7`, `mysql:8`, `ubuntu`, and worker images.
2. **Start the OS + DB stack**:
   ```bash
   # Portable Linux / macOS compose (starts controller, redis, dbbench-std, os_interaction-std)
   docker compose -f extra/docker-compose.os-db.yml up -d
   # Or on macOS explicitly:
   docker compose -f extra/docker-compose.mac.yml up -d
   ```
   Build local OS task images if not already built:
   ```bash
   for df in default packages ubuntu; do
     docker build -t local-os/$df -f data/os_interaction/res/dockerfiles/$df data/os_interaction/res/dockerfiles
   done
   ```
3. **Configure the LLM endpoint** in `eval/model.yaml` (e.g., Colab Ollama + Cloudflare tunnel, or any OpenAI-compatible API):
   - Model 1: `qwen3:8b` (`thinking_off: true`)
   - Model 2: `llama3.1:8b` (`thinking_off: false`)
   - Extra models for Proposal 1 (~5 configurations): e.g., `qwen3:4b`, `qwen2.5:7b`, `qwen3:8b` with thinking on (`thinking_off: false`), etc.
   Verify connectivity:
   ```bash
   .venv/bin/python eval/run.py check
   ```

---

## 2. Running Proposal 3 (5 Conditions on all 444 DB + OS items)

Run all 5 conditions (`c0`, `c1`, `c2`, `c3`, `c4`) + cross-task transfer (`DB -> OS`, `OS -> DB`) for a configured model:
```bash
.venv/bin/python eval/run_iteration2.py \
  --config eval/model.yaml \
  --cross-task-transfer \
  --variance-runs 3 \
  --variance-limit 50
```
For a quick 10-item smoke test first:
```bash
.venv/bin/python eval/run_iteration2.py --config eval/model.yaml --limit 10
```

Analyze Proposal 3 results (success rate, Wilson CI, paired McNemar exact test vs `c0`, `<=1`-command rate, TLE rate, $P(10, 0.8)$ repetition, helped/harmed flips):
```bash
.venv/bin/python eval/analyze_proposal3.py eval/results/iteration2 \
  --out-json eval/results/iteration2/proposal3_summary.json \
  --out-md eval/results/iteration2/proposal3_report.md
```

---

## 3. Running Proposal 1 (Trajectory-Aware Bayesian IRT on 444 Items)

Once trajectories from ~5 model configurations (plus optional $T=0.7$ repeated runs) are collected under `eval/results/iteration2`:
```bash
.venv/bin/python eval/analyze_proposal1.py eval/results/iteration2 \
  --n-boot 200 \
  --n-seeds 10 \
  --out-json eval/results/iteration2/proposal1_summary.json \
  --out-md eval/results/iteration2/proposal1_report.md
```

---

## 4. Automated Test Suite (No GPU / No Live Tunnel Required)

```bash
.venv/bin/pytest -v eval/tests/test_iteration2.py
.venv/bin/python scripts/validate_lite_configs.py
```
