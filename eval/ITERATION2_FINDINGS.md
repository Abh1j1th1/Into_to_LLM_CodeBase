# Task 4 – Iteration 2 Empirical Findings (Group 12)

Evaluated live on AgentBench FC (`dbbench-std` + `os-std` Docker environments) across **495 valid episodes** on the stratified 40-item benchmark ([stratified_40.json](stratified_40.json): 20 DB across all 10 SQL types + 20 OS across all 7 problem directories `1`–`7`) spanning **5 distinct models** (`gemini-3.5-flash-lite`, `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `openai/gpt-oss-safeguard-20b`, `qwen/qwen3.8-27b`) and **15 model/condition configurations**.

---

## 1. Proposal 3 Results (Failure-Aware Agent & Targeted Verify Gate)

Full summary in [proposal3_report.md](results/iteration2/proposal3_report.md) and [proposal3_summary.json](results/iteration2/proposal3_summary.json).

| Model | Condition | Split | Valid N | Success Rate | <=1 Cmd Rate | TLE Rate | Mean Env Calls | P(10, 0.8) | McNemar p vs c0 | Helped / Harmed |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `gemini-3.5-flash-lite` | `c0` | OVERALL | 39 | 51.3% | 33.3% | 12.8% | 3.62 | 28.2% | — | — |
| `gemini-3.5-flash-lite` | `c0` | DB / OS | 19 / 20 | 63.2% / 40.0% | 36.8% / 30.0% | 5.3% / 20.0% | 3.58 / 3.65 | 31.6% / 25.0% | — | — |
| `gemini-3.5-flash-lite` | `c1` | OVERALL | 40 | 50.0% | 40.0% | 2.5% | 3.17 | 32.5% | 1.0000 | +1 / -2 |
| `gemini-3.5-flash-lite` | `c2` | DB | 20 | **80.0%** | 40.0% | **0.0%** | 3.25 | 25.0% | 0.2500 | **+3 / -0** |
| `gemini-3.5-flash-lite` | `c2` | OS | 8 | 37.5% | 37.5% | 0.0% | 2.88 | 12.5% | 0.5000 | +0 / -2 |
| `openai/gpt-oss-120b` | `c0` | OVERALL | 36 | 58.3% | 75.0% | 2.8% | 1.69 | 11.1% | — | — |
| `openai/gpt-oss-120b` | `c3` | DB | 6 | 66.7% | **0.0%** | 0.0% | 2.67 | 16.7% | 1.0000 | +0 / -0 |
| `openai/gpt-oss-120b` | `c4` | OVERALL | 36 | 58.3% | **0.0%** | 11.1% | 4.25 | 38.9% | 1.0000 | +2 / -3 |
| `openai/gpt-oss-120b` | `c4` | DB | 17 | **76.5%** | **0.0%** | 0.0% | 3.77 | 35.3% | 1.0000 | +1 / -1 |
| `openai/gpt-oss-20b` | `c0` | OVERALL | 38 | 52.6% | 65.8% | 0.0% | 1.95 | 10.5% | — | — |
| `openai/gpt-oss-20b` | `c1` | OVERALL | 38 | 47.4% | 60.5% | 0.0% | 2.08 | 21.1% | 0.6250 | +1 / -3 |
| `openai/gpt-oss-20b` | `c3` | DB / OS | 20 / 13 | 65.0% / 23.1% | **0.0%** / **0.0%** | 0.0% / 38.5% | 3.30 / 5.46 | 45.0% / 69.2% | 1.0000 | +2 / -3 |
| `openai/gpt-oss-20b` | `c4` | DB | 8 | 62.5% | **0.0%** | 0.0% | 3.50 | 50.0% | 1.0000 | +0 / -0 |
| `openai/gpt-oss-safeguard-20b` | `c0` | OVERALL | 82 | 51.2% | 65.8% | 0.0% | 1.79 | 19.5% | — | — |
| `openai/gpt-oss-safeguard-20b` | `c3` | OVERALL | 25 | **64.0%** | **0.0%** | 0.0% | 2.72 | 32.0% | 0.1250 | **+6 / -1** |
| `openai/gpt-oss-safeguard-20b` | `c3` | DB | 19 | **73.7%** | **0.0%** | 0.0% | 2.68 | 31.6% | 0.1250 | **+4 / -0** |
| `openai/gpt-oss-safeguard-20b` | `c4` | DB | 17 | **70.6%** | **0.0%** | 0.0% | 2.59 | 35.3% | 0.5000 | **+2 / -0** |
| `qwen/qwen3.8-27b` | `c0` | OVERALL | 40 | 47.5% | 22.5% | 12.5% | 3.52 | 42.5% | — | — |
| `qwen/qwen3.8-27b` | `c3` | OVERALL | 29 | **55.2%** | **3.5%** | **6.9%** | 4.34 | 62.1% | 1.0000 | +0 / -0 |
| `qwen/qwen3.8-27b` | `c3` | OS | 9 | **44.4%** | 11.1% | 22.2% | 4.11 | 33.3% | 1.0000 | +0 / -0 |

Key takeaways:
1. **Early unverified answering (`<=1` command) generalizes across model families**: In baseline `c0`, `gpt-oss-120b` (75.0%), `gpt-oss-20b` (65.8%), `gpt-oss-safeguard-20b` (65.8%), and `gemini-3.5-flash-lite` (33.3%) frequently answered in `<=1` command (`early_unverified_answer`), whereas `qwen/qwen3.8-27b` had a 22.5% `<=1`-cmd rate and suffered 12.5% TLE (25.0% on OS) and 42.5% $P(10,0.8)$ repetition.
2. **State-conditioned failure memory (`c2`) cuts TLE and boosts DBBench**: On `gemini-3.5-flash-lite`, `c2` raised DBBench success from **63.2% $\to$ 80.0%** (`+3/-0` flips, TLE `5.3% -> 0.0%`), and dropped overall TLE from **12.8% $\to$ 0.0%**, whereas simple repeat warnings (`c1`) alone did not improve accuracy (`51.3% -> 50.0%` on Gemini; `52.6% -> 47.4%` on `gpt-oss-20b`).
3. **Targeted verify gate (`c3`) and combined (`c4`) eliminate `<=1`-command guessing**: Dropped `<=1`-command rates to **0.0%** across all OpenAI-OSS models and `3.5%` on `qwen3.8-27b`, yielding `+6/-1` overall flips on `gpt-oss-safeguard-20b` (`51.2% -> 64.0%`, McNemar $p=0.125$), `70.6% -> 76.5%` DB on `gpt-oss-120b` (`c4`), and `47.5% -> 55.2%` overall on `qwen3.8-27b` (`c3`). Trade-off: extra verification rounds increase token cost and raise OS TLE rates under OS's strict 8-round limit (`gpt-oss-20b c3` OS hit 38.5% TLE; `gpt-oss-120b c4` OS hit 21.1% TLE).

---

## 2. Proposal 1 Results (Trajectory-Aware Bayesian IRT)

Evaluated across **15 observed model/condition configurations** on the 40 stratified items + repeated $T=0.7$ (`k=3`) runs on `openai/gpt-oss-safeguard-20b` ([proposal1_report.md](results/iteration2/proposal1_report.md), [proposal1_summary.json](results/iteration2/proposal1_summary.json)):

- **Held-out 5-fold CV instance prediction**:
  - `outcome_1pl`: accuracy `0.8333`, Brier `0.1092`, log-loss `0.3481`
  - `outcome_2pl`: accuracy `0.8244`, Brier `0.1095`, log-loss `0.3415`
  - `traj_2pl`: accuracy `0.8222`, Brier `0.1185`, log-loss `0.3884`
- **Calibration-subset full-score recovery (10%, 20%, 30% budgets)**:
  - At **20% (`k=8`)**: `info_traj__macro_mean` achieved $\tau = 0.532, \rho = 0.640$ (vs `random__reciprocal_weighted` $\tau = 0.282, \rho = 0.350$), and `info_traj__traj_2pl` / `info_2pl__traj_2pl` cut full-set score error to **MAE = 0.1298, RMSE = 0.1527** (vs `random__reciprocal_weighted` MAE = 0.2260, RMSE = 0.2690).
  - At **30% (`k=12`)**: `info_2pl__traj_2pl` achieved **MAE = 0.0801, RMSE = 0.1011** (vs `random__reciprocal_weighted` MAE = 0.1679, RMSE = 0.2068).
- **Leave-one-environment sensitivity**:
  - Dropping `db`: `traj_2pl` retained $\tau = 0.4831, \rho = 0.6380$ vs `reciprocal_weighted` $\tau = 0.1757, \rho = -0.0072$.
  - Dropping `os`: `traj_2pl` retained $\tau = 0.6952, \rho = 0.8571$ vs `reciprocal_weighted` $\tau = 0.2366, \rho = 0.3907$.
- **Run-to-run variance at `temperature=0.7` (`k=3`, 15 DBBench items, `gpt-oss-safeguard-20b`)**:
  - Mean score across runs: `0.556 ± 0.038` | **Best-of-3: `0.733`** (+17.7 pp) | Mean per-item variance: `0.0889` (`SD = 0.154`).

---

## 3. Quota & Scope Note

All 3 Groq free-tier keys hit Groq's `200,000` TPD per-model limit and `gemini-3.5-flash-lite` hit Google AI Studio's `500` RPD free-tier limit (while `gemini-3.1-flash-lite`, `gemini-3.5-flash`, and `gemini-3.8-flash` returned HTTP `503 UNAVAILABLE` during testing). All remaining runs (including `c2` on Groq, `c3`/`c4` on Gemini, cross-task transfer, or scaling to all 444 items) can be resumed without code changes via `-f` / `eval/run_iteration2.py`.
