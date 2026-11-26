# Failure-Aware Tool-Use Agents and Trajectory-Aware Evaluation on AgentBench FC

**Group 12**

> **Note on consolidation**: Sections 1, 3, 4, 5, 6, and the Abstract below are grounded directly in the verified codebase, [RUN_NOTES.md](../RUN_NOTES.md), [ARCHITECTURE.md](../ARCHITECTURE.md), and our live evaluation outputs under `eval/results/iteration2/` ([ITERATION2_FINDINGS.md](ITERATION2_FINDINGS.md)). Where prior assignment documents (Task 1 Literature Review, Task 2 Problem Formulation, Task 3 Ideation / `Task4_Iteration1_Group12.pdf`) are not present in this repository, `[INSERT ...]` markers indicate where to paste your prior write-ups without inventing unverified citations.

---

## Abstract

Multi-turn tool-use agents evaluated on interactive benchmarks frequently fail either by submitting premature answers after minimal environment inspection or by looping on repeated commands until hitting round limits. We investigate client-side failure-aware interventions and trajectory-aware evaluation on **AgentBench FC** (`dbbench-std` and `os-std`) using local open-weight models served via Ollama. Specifically, we formalize and evaluate a five-condition client-side intervention suite (`c0`–`c4`) featuring repeated-action warnings, state-conditioned failure memory, and a targeted verification gate (`c3`) that holds a model's first commit when $\ge 2$ rounds remain and requires an independent verification command, alongside a Bayesian Item Response Theory (IRT) evaluation framework (`outcome_1pl`, `outcome_2pl`, `traj_2pl`) with Laplace posterior uncertainty. On a stratified 40-item subset (`20` DB + `20` OS, `T=0.0`), `qwen2.5:3b` completed all `80` `c0`/`c3` episodes (`27.5%` overall `c0` vs. `25.0%` `c3`; McNemar exact $p = 1.0000$), confirming that the verify gate reliably fired (`<=1`-command rate dropped from `50.0%` to `10.0%`, gate fired on `67.5%` of episodes) while increasing prompt tokens (`4,241.4` to `8,403.9`) and latency (`16.30 s` to `28.42 s`) without a statistically significant accuracy gain at `N=40`. Larger local models (`qwen2.5:7b` at `28.9%` `c0` over `38` episodes; `gemma4:12b` at `50.0%` DB `c0` over `8` episodes) hit controller session-expiration timeouts (`5`–`10` min) under local 6 GB VRAM constraints. We document the two-round LLM refinement process, protocol compatibility fixes, and methodological limitations of small-sample local evaluation.

---

## 1. Introduction

### 1.1 Background Motivation and Problem Definition

Evaluating Large Language Models (LLMs) as autonomous tool-using agents requires multi-turn interaction with stateful environments such as SQL databases and Linux shells. On **AgentBench FC** (`dbbench-std` with a 15-round limit and `os-std` with an 8-round limit), baseline agents (`c0`) exhibit two prominent behavioral failure modes:
1. **Early unverified answering (`<=1` command)**: The agent submits a final answer after at most one environment call, guessing from partial or unverified observations (observed in `50.0%` of `qwen2.5:3b` `c0` episodes and `36.8%` of `qwen2.5:7b` `c0` episodes).
2. **Repeated-action loops and Time Limit Exceeded (TLE)**: The agent repeats identical or near-identical commands in the same environment state without recovering from errors, empty outputs, or truncated outputs.

Furthermore, standard benchmark aggregation (micro-mean or reciprocal-weighted environment averages) treats all items as equally informative and ignores trajectory-level signals such as repetition or early stopping, while hardware-constrained local evaluation introduces strict latency and session-timeout constraints.

*(Optional: `[INSERT Task 2 Research Gap & Problem Motivation text here]`)*

### 1.2 Contributions

1. **AgentRL FC Protocol Alignment**: Diagnosed and resolved protocol incompatibilities between the legacy v0.2 client and the AgentRL FC controller (`port 5020`, `session_id` headers, incremental `messages`, OpenAI tool-call schemas, and LF line endings in worker scripts).
2. **Client-Side Failure-Aware Intervention Suite (`c0`–`c4`)**: Implemented modular, environment-agnostic client-side conditions in [conditions.py](conditions.py) without modifying task environments or grading logic—covering repeated-action warnings (`c1`), state-conditioned failure memory (`c2`), a budget-aware targeted verify gate (`c3`), and their combination (`c4`).
3. **Trajectory-Aware Bayesian IRT Framework**: Implemented 1PL, 2PL, and trajectory-aware 2PL IRT models ([irt.py](irt.py)) with Gaussian priors, Laplace standard errors, and temperature-filtered paired statistical reporting ([analyze_proposal3.py](analyze_proposal3.py)).
4. **Empirical Local-Hardware Evaluation & Refinement Log**: Evaluated `qwen2.5:3b`, `qwen2.5:7b`, and `gemma4:12b` on a stratified 40-item AgentBench FC subset (`126` valid `T=0.0` episodes), documenting exact paired McNemar tests, compute-latency trade-offs, session-expiration failure modes, and a two-round LLM refinement log.

---

## 2. Related Work

*(Consolidate your filtered Task 1 Literature Review here using the hierarchical structure below. Only verified citations present in the repo/assignment are listed as placeholders.)*

### 2.1 LLM Agent Benchmarks and Tool-Use Evaluation
- **AgentBench**: Multi-environment benchmark for LLMs as agents (THUDM, [arXiv:2308.03688](https://arxiv.org/abs/2308.03688)), adapted in this work via its Function Calling (FC) release on AgentRL.
- `[INSERT filtered Task 1 papers on LLM agent benchmarks]`

### 2.2 Failure-Aware, Self-Verifying, and Memory-Augmented Agents
- `[INSERT filtered Task 1 papers on reflection, failure memory, and verification gates]`
- Reference papers noted in Task 4: DeepScientist (`arXiv:2509.26603`), AutoSOTA (`arXiv:2604.05550`), EvoScientist (`arXiv:2603.08127`), Arbor (`arXiv:2606.11926`) — `[INSERT verified summaries if read/used in Task 1–4]`.

### 2.3 Statistical Evaluation and Item Response Theory for LLMs
- `[INSERT filtered Task 1 papers on Bayesian IRT and adaptive/subset benchmark evaluation]`

---

## 3. Problem Formulation

### 3.1 Multi-Turn Tool-Use Episode with Client-Side Interventions

Each benchmark item $i \in \mathcal{I} = \mathcal{I}_{\text{DB}} \cup \mathcal{I}_{\text{OS}}$ defines an interactive episode over discrete turns $t = 1, \dots, R_{\text{env}}$ with round budget $R_{\text{DB}} = 15$ and $R_{\text{OS}} = 8$. At turn $t$, given conversation history $h_t$, the agent policy $\pi_\theta$ emits an assistant message $a_t$ containing tool calls from $\mathcal{T}_{\text{env}} \cup \mathcal{T}_{\text{commit}}$ (where $\mathcal{T}_{\text{env}} = \{\texttt{execute\_sql}\}$ or $\{\texttt{bash\_action}\}$, and $\mathcal{T}_{\text{commit}} = \{\texttt{commit\_final\_answer}\}$ or $\{\texttt{answer\_action}, \texttt{finish\_action}\}$).

A client-side intervention operator $\Phi_c(h_t, a_t, s_t)$ under condition $c \in \{c_0, c_1, c_2, c_3, c_4\}$ maintains episode state $s_t$ (environment call count $n_{\text{env}}$, action history, and failure records) and transforms the interaction without altering the server environment or binary reward $y_{m,i} \in \{0, 1\}$:
- Under $c_3$ (targeted verify gate), if $a_t$ invokes a commit tool, the gate has not yet fired, and remaining budget $R_{\text{env}} - n_{\text{env}} \ge 2$, $\Phi_{c_3}$ holds the commit and injects a verification prompt requiring an independent $\mathcal{T}_{\text{env}}$ check before re-submission.

### 3.2 Trajectory-Aware Bayesian Item Response Theory

Given an observed binary outcome matrix $Y \in \{0, 1\}^{M \times I}$ across $M$ model-condition configurations and $I$ items, Proposal 1 models item-level pass probabilities via regularized Maximum A Posteriori (MAP) estimation with Gaussian priors and Laplace posterior standard errors:
- **Outcome 1PL / 2PL**:
  $$P(y_{m,i} = 1 \mid \theta_m, a_i, b_i) = \sigma\!\big(a_i (\theta_m - b_i)\big)$$
  where $\theta_m$ is latent ability, $b_i$ is item difficulty, $a_i > 0$ is item discrimination ($a_i = 1$ in 1PL), and $\sigma(\cdot)$ is the logistic sigmoid.
- **Trajectory-Aware 2PL (`traj_2pl`)**: Jointly incorporates auxiliary trajectory features (with parameters $\alpha_0, \alpha_1$) alongside binary outcomes to estimate $\theta_m$ and $\operatorname{SE}(\theta_m)$.

*(Optional: `[INSERT exact Task 2 mathematical formulation if different from eval/irt.py]`)*

---

## 4. Methodology

### 4.1 Dataset and Environments

Evaluations are conducted live on **AgentBench FC** using Dockerized `dbbench-std` (MySQL containers, `300` total items) and `os-std` (Ubuntu bash containers, `144` total items) orchestrated by `agentrl-controller` and Redis. To enable controlled local evaluation under 6 GB VRAM constraints, we use the stratified 40-item benchmark ([stratified_40.json](stratified_40.json)) comprising **20 DB items** and **20 OS items**, evaluated at greedy decoding ($T = 0.0$, concurrency $1$, `OLLAMA_CONTEXT_LENGTH=8192`).

### 4.2 Failure-Aware Agent Conditions (`c0`–`c4`)

All interventions operate purely on the client side ([conditions.py](conditions.py), [server_agent.py](server_agent.py)):
- **`c0` (Baseline)**: Standard multi-turn tool-calling loop.
- **`c1` (Repeated-Action Warning)**: Detects exact or near-identical ($\text{ROUGE-L } F_1 \ge 0.85$) commands against recent steps and appends a warning advisory.
- **`c2` (State-Conditioned Failure Memory)**: Logs structured failure records for execution errors, empty non-mutating outputs, truncated outputs, repeated actions, and missing tool calls, injecting episode-local (and optional cross-task) advisories.
- **`c3` (Targeted Verify Gate)**: Intercepts the first commit attempt when $\ge 2$ rounds remain, requiring the agent to map question constraints to observed outputs and execute a second independent environment command before re-committing.
- **`c4` (Combined)**: Combines `c2` failure memory with the `c3` verify gate.

### 4.3 Statistical Analysis and Verification Protocol

Results are parsed and filtered by [analyze_proposal3.py](analyze_proposal3.py), which excludes any rows sampled at $T > 0$ (via both row metadata and filename temperature tags) so the `c0` baseline remains strictly $T = 0$. Condition effects are evaluated on **matched item pairs** using **exact two-sided McNemar's test**, alongside behavioral diagnostics (`<=1`-command answer rate, TLE rate, mean environment calls, exact repeat rate, $P(10, 0.8)$ repetition rate, gate-firing rate, token usage, and latency).

---

## 5. Experimentation, Results, and Discussion

### 5.1 Experimental Setup

We tested three local Ollama models on Windows 11 (Ryzen 7 7840HS, 16 GB RAM, RTX 3050 6 GB):
- `qwen2.5:3b` (`1.9 GB`, `100% GPU`, `tools` supported): full `c0` and `c3` runs (`80` episodes).
- `qwen2.5:7b` (`4.7 GB`, `21%/79% CPU/GPU`, `tools` supported): `c0` run (`38` completed episodes before stopping on repeated session expiration).
- `gemma4:12b` (`8.0 GB`, `tools` and `thinking` supported): `c0` run (`8` completed DB episodes before stopping on repeated stalls/session expirations).

### 5.2 Quantitative Results

All comparisons are made solely against the repo's own `c0` baseline under identical protocol (not against the v0.2 AgentBench leaderboard).

| Model | Condition | Split | N | Success Rate | Paired (`c0% -> c3%`) | <=1 Cmd Rate | TLE Rate | Mean Env Calls | Repeat Rate | P(10, 0.8) | McNemar p vs c0 | Helped / Harmed |
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

### 5.3 Discussion

1. **Prevalence of `<=1`-command answers in `c0`**: Both `qwen2.5:3b` (`50.0%` overall; `65.0%` OS) and `qwen2.5:7b` (`36.8%` overall; `66.7%` OS) frequently committed answers after at most one environment call.
2. **Verify gate (`c3`) mechanics vs. accuracy**: On `qwen2.5:3b`, `c3` reduced `<=1`-command answers from `50.0%` to `10.0%` (`35.0% -> 0.0%` DB, `65.0% -> 20.0%` OS), confirming that the gate fired (`verify_gate_fired_rate = 67.5%` overall, `verify_ran_new_cmd_rate = 62.5%`, `verify_changed_answer_rate = 20.0%`). Note that reducing `<=1`-command rate is true by construction when the gate holds the first commit. In accuracy terms, the paired difference was not statistically significant (`27.5% -> 25.0%`, `+4 / -5`, McNemar exact $p = 1.0000$; DB $p = 0.6250$; OS $p = 1.0000$). With `N=20` per task and a 3B model, small fluctuations are indistinguishable from noise.
3. **Compute and repetition overhead**: Forcing verification on `qwen2.5:3b` increased mean environment calls (`2.15 -> 3.58`), cumulative prompt tokens (`4,241.4 -> 8,403.9`), completion tokens (`486.2 -> 756.4`), mean latency (`16.30 s -> 28.42 s`), and $P(10, 0.8)$ repetition (`50.0% -> 75.0%`).
4. **Hardware and session-timeout bottlenecks**: `qwen2.5:7b` stopped on `os-std` index `76` after `agentrl-controller` expired `session 127` and `session 129` (leaving `server_agent.py` hung on `/interact`). `gemma4:12b` exceeded 6 GB VRAM and stalled past controller session limits on `dbbench-std` indices `67` (`session 138` expired) and `93` (`session 139` stalled $>10$ min even with `thinking_off: true`).
5. **Two-round LLM refinement**:
   - *Round 1 (Protocol & Runtime)*: Migrated from the incompatible v0.2 client to AgentRL `server_agent.py` on port `5020`, fixed Compose networking, enforced LF line endings for worker scripts, and added tool-call/message sanitization.
   - *Round 2 (Evaluation Hygiene & Reporting)*: Patched `analyze_proposal3.py` to exclude $T > 0$ rows via `filename_temperature`, added matched-`N` `Paired` columns and McNemar tests, corrected `c3` claims to "confirms the gate fired," and documented unverified gaps (e.g., a smoke-test DB query using `LIMIT 1` scoring correct; `load_memory_bank` returning the first 3 advisories rather than task-matched ones).
6. **Methodological limitations**: Proposal 1 (`analyze_proposal1.py`) was skipped because only `qwen2.5:3b` has paired `c0`/`c3` coverage; more broadly, Proposal 1's subset selection is fit on the same matrix it is scored against (circular), and 5-fold CV over `(model, item)` cells leaks other-condition results for the same item.

---

## 6. Conclusion

Client-side failure-aware interventions can be cleanly integrated into AgentBench FC without modifying task environments or grading logic. Across `126` verified temperature-0 episodes on local open-weight models, baseline `c0` agents frequently submit answers after $\le 1$ environment command (`50.0%` on `qwen2.5:3b`, `36.8%` on `qwen2.5:7b`). Enabling the targeted verify gate (`c3`) on `qwen2.5:3b` reliably enforces an additional verification step (`<=1`-command rate reduced to `10.0%`, gate fired on `67.5%` of episodes) at roughly $2\times$ prompt-token and latency cost, while yielding no statistically significant accuracy improvement over `c0` at `N=40` (`27.5% -> 25.0%`, McNemar $p = 1.0000$). Scaling local evaluation to larger models (`7B`–`12B`) on consumer 6 GB VRAM hardware is constrained by controller session timeouts during slow multi-turn episodes. Future work requires multi-model, multi-condition runs with higher VRAM budgets to evaluate `c1`/`c2`/`c4` and validate Proposal 1's IRT calibration across a full model population.
