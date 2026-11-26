# Task 4 – Iteration 2 Empirical Findings (Group 12)

Evaluated live on AgentBench FC (`dbbench-std` + `os-std` Docker environments) across **126 valid temperature-0 episodes** (`sampled_rows_excluded = 0`) on the stratified 40-item benchmark ([stratified_40.json](stratified_40.json): 20 DB + 20 OS items) using local Ollama models (`OLLAMA_CONTEXT_LENGTH=8192`):
- `qwen2.5:3b` (`100% GPU`): complete `c0` + `c3` (`80` episodes).
- `qwen2.5:7b` (`21%/79% CPU/GPU`): partial `c0` (`38` episodes: 20 DB + 18 OS; stopped after `os-std` index `76` hung twice).
- `gemma4:12b` (`8.0 GB` on 6 GB VRAM): partial `c0` (`8` DB episodes; stopped after hanging twice on DB indices `67` and `93`).

---

## 1. Proposal 3 Results (Failure-Aware Agent & Targeted Verify Gate)

Full summary in [proposal3_report.md](results/iteration2/proposal3_report.md) and [proposal3_summary.json](results/iteration2/proposal3_summary.json).

All comparisons are against the repo's own `c0` baseline under identical protocol (`T=0.0`, `stratified_40.json`).

| Model | Condition | Split | N | Success Rate | Paired (`c0% -> cond%`) | <=1 Cmd Rate | TLE Rate | Mean Env Calls | Repeat Rate | P(10, 0.8) | McNemar p vs c0 | Helped / Harmed |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen2.5:3b` | `c0` | OVERALL | 40 | 27.5% (11/40) | — | 50.0% | 0.0% | 2.15 | 12.8% | 50.0% | — | — |
| `qwen2.5:3b` | `c0` | DB | 20 | 40.0% (8/20) | — | 35.0% | 0.0% | 2.60 | 21.2% | 60.0% | — | — |
| `qwen2.5:3b` | `c0` | OS | 20 | 15.0% (3/20) | — | 65.0% | 0.0% | 1.70 | 4.3% | 40.0% | — | — |
| `qwen2.5:3b` | `c3` | OVERALL | 40 | 25.0% (10/40) | 40: 27.5% -> 25.0% | 10.0% | 2.5% | 3.58 | 20.7% | 75.0% | 1.0000 | +4 / -5 |
| `qwen2.5:3b` | `c3` | DB | 20 | 30.0% (6/20) | 20: 40.0% -> 30.0% | 0.0% | 5.0% | 4.80 | 23.3% | 80.0% | 0.6250 | +1 / -3 |
| `qwen2.5:3b` | `c3` | OS | 20 | 20.0% (4/20) | 20: 15.0% -> 20.0% | 20.0% | 0.0% | 2.35 | 18.1% | 70.0% | 1.0000 | +3 / -2 |
| `qwen2.5:7b` | `c0` | OVERALL | 38 | 28.9% | — | 36.8% | 7.9% | 2.53 | 22.3% | 57.9% | — | — |
| `qwen2.5:7b` | `c0` | DB | 20 | 45.0% | — | 10.0% | 0.0% | 2.75 | 31.7% | 85.0% | — | — |
| `qwen2.5:7b` | `c0` | OS | 18 | 11.1% | — | 66.7% | 16.7% | 2.28 | 11.8% | 27.8% | — | — |
| `gemma4:12b` | `c0` | OVERALL | 8 | 50.0% | — | 50.0% | 12.5% | 3.25 | 5.8% | 25.0% | — | — |
| `gemma4:12b` | `c0` | DB | 8 | 50.0% | — | 50.0% | 12.5% | 3.25 | 5.8% | 25.0% | — | — |

Key observations (exploratory; small `N` on local models):
1. **Baseline `c0` frequently stops after `<=1` command**: `qwen2.5:3b` answered in `<=1` environment command in **50.0%** of `c0` episodes overall (**35.0%** DB, **65.0%** OS); `qwen2.5:7b` did so in **36.8%** overall (**10.0%** DB, **66.7%** OS).
2. **No statistically significant accuracy effect from `c3` vs `c0` (`qwen2.5:3b`)**: Overall paired success moved from **27.5% -> 25.0%** (`+4 / -5`, McNemar exact $p = 1.0000$; DB `40.0% -> 30.0%`, `+1 / -3`, $p = 0.6250$; OS `15.0% -> 20.0%`, `+3 / -2`, $p = 1.0000$). With 20 items per task and a 3B model, these differences are consistent with noise.
3. **Verify gate behavior (`c3` confirms the gate fired)**: `c3` reduced `<=1`-command rate from **50.0% -> 10.0%** overall (**35.0% -> 0.0%** DB, **65.0% -> 20.0%** OS), confirming the gate fired (`verify_gate_fired_rate = 67.5%` overall: `80.0%` DB, `55.0%` OS; `verify_ran_new_cmd_rate = 62.5%` overall; `verify_changed_answer_rate = 20.0%` overall). Note that the reduction in `<=1`-command rate is true by construction when the gate holds the first commit and forces a second command.
4. **Compute & repetition trade-off (`qwen2.5:3b`)**: Firing the verify gate increased mean environment calls (**2.15 -> 3.58**), cumulative prompt tokens (**4,241.4 -> 8,403.9**), completion tokens (**486.2 -> 756.4**), mean episode latency (**16.30 s -> 28.42 s**), and `P(10, 0.8)` repetition rate (**50.0% -> 75.0%**).

### Stopped Runs & Exact Errors (`qwen2.5:7b`, `gemma4:12b`)
- **`qwen2.5:7b` (`os-std c0`)**: Completed 20/20 `dbbench-std c0` (`45.0%`) and 18/20 `os-std c0` (`11.1%`). Stopped after `os-std` index `76` hung twice in `server_agent.py` while `agentrl-controller` logged `session 127 expired` (`calling worker os-std#1 to cancel session 127`) and `session 129 expired` (`calling worker os-std#1 to cancel session 129`). `c3` was not reached.
- **`gemma4:12b` (`dbbench-std c0`)**: Cold-load `eval/run.py check` failed once with `ReadTimeout` (`95 s`), then passed (`6.4 s` warm, `50.3 s` with `thinking_off: true`). Completed 8/20 `dbbench-std c0` (`50.0%`), then stopped after hanging twice: first on `session 138` (`dbbench-std` index `67`, `agentrl-controller` logged `session 138 expired` after ~9.6 min with `thinking_off: false`), and second on `session 139` (`dbbench-std` index `93`, stalled >10 min with `thinking_off: true` before Ollama went idle).

---

## 2. Proposal 1 Status & Limitations (Trajectory-Aware Bayesian IRT)

- **Skipped on this run**: `analyze_proposal1.py` was not run because the current `eval/results/iteration2/` dataset contains only **4 partial/complete `model × condition` configs** (`qwen2.5:3b` × `{c0, c3}`, `qwen2.5:7b` × `{c0}` partial, `gemma4:12b` × `{c0}` partial), with only `qwen2.5:3b` having paired `c0`/`c3` coverage, which is insufficient for meaningful IRT rank intervals, subset calibration, or 5-fold cross-validation.
- **Methodological limitations** (for any multi-config Proposal 1 run):
  1. **Circular subset selection**: IRT calibration-subset selection is fit on the same response matrix it is scored against.
  2. **Non-independent CV cells**: The evaluated "models" are a small set of `model × condition` configurations, and 5-fold CV over `(model, item)` cells leaves the same model's other-condition result on the same item in the training fold.

---

## 3. Refinement Log (2 LLM Feedback Rounds)

### Round 1: AgentRL FC Protocol & Runtime Compatibility
- **Issue found**: As documented in [RUN_NOTES.md](file:///C:/Users/abhij/OneDrive/Desktop/Projects/AgentBench_Iteration2/RUN_NOTES.md), the repo's legacy v0.2 client (`src/assigner.py`, `src/client/task.py`, `HTTPAgent`) expected port `5000`, `agent_response` text payloads, and `POST /calculate_overall`, which failed against the FC stack (`agentrl-controller` on port `5020`, `session_id` HTTP headers, incremental `messages`, `tools` schema). In addition, `server_agent.py` `-T` raised `KeyError` on non-tool responses, Docker `network_mode: host` failed on Docker Desktop, and CRLF line endings in `*.sh` prevented worker registration.
- **Feedback given to the model**: Replace the legacy client path with AgentRL's `server_agent.py`, expose port `5020:5020` and route workers via Compose DNS (`http://controller:5020/api`), enforce LF line endings (`*.sh text eol=lf` in `.gitattributes`), sanitize assistant messages (`sanitize_message_for_api`), and benchmark only against the repo's own `c0` baseline rather than the v0.2 leaderboard.
- **Fix**: Implemented `eval/server_agent.py`, `eval/conditions.py`, `eval/run_iteration2.py`, `eval/check.py`, `extra/docker-compose.os-db.yml`, and `.gitattributes`.
- **Effect**: Both `dbbench-std` (`300` items) and `os-std` (`144` items) workers registered with the controller, `eval/run.py check` passed on `qwen2.5:3b` (`tools` supported, `OLLAMA_CONTEXT_LENGTH=8192`), and full `c0`/`c3` episodes ran and resumed cleanly (`80` valid rows).

### Round 2: Evaluation Hygiene, Statistical Reporting & Remaining Spec Gaps
- **Issue found**:
  1. `eval/analyze_proposal3.py` originally grouped only by `(model, condition)`, risking mixing `T=0.7` variance rows into `T=0` `c0` baselines, and lacked a matched-item `Paired` column.
  2. Early write-up language framed `c3` as "eliminating `<=1`-command guessing" (true by construction when the gate holds the first commit) and risked overclaiming accuracy effects on small `N=20` per-task samples.
  3. Unverified/open gaps identified: a smoke-test DB trajectory ran `SELECT Notes ... WHERE Method = 'decision' LIMIT 1` and committed one row, yet scored correct even though the question asked for all matching Notes (either the item has one match or the checker is lenient; unverified); `load_memory_bank` returns the first 3 advisories rather than task-matched ones; Proposal 1's subset selection is circular and its 5-fold CV leaks same-item/other-condition cells.
- **Feedback given to the model**: Filter out `T > 0` rows in `analyze_proposal3.py` using `filename_temperature` + row metadata, add a matched-N `Paired` column, use absolute `--out-dir` paths, reword `c3` conclusions to "confirms the gate fired" with exact McNemar $p$-values, and skip `analyze_proposal1.py` when fewer than a multi-model/condition grid is available.
- **Fix**: Applied `filename_temperature` and `Paired` reporting in [analyze_proposal3.py](file:///C:/Users/abhij/OneDrive/Desktop/Projects/AgentBench_Iteration2/eval/analyze_proposal3.py#L32-L35), re-ran `qwen2.5:3b` (`c0`, `c3`) cleanly under `eval/results/iteration2/`, and updated this document with verified numbers and explicit limitations.
- **Effect**: Confirmed `sampled_rows_excluded = 0` across all `126` `T=0` episodes (`N=40` paired on `qwen2.5:3b`), verified `c3` gate firing (`<=1`-command rate `50.0% -> 10.0%`, `verify_gate_fired_rate = 67.5%`), and accurately reported the non-significant accuracy delta (`27.5% -> 25.0%`, McNemar $p = 1.0000$) and latency/token cost (`16.30 s -> 28.42 s`).
