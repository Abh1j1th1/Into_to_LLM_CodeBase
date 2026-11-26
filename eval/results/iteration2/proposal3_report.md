# Proposal 3 Evaluation Report (Failure-Aware Agent Conditions)

Temperature-0 runs only. `Paired` = matched items vs c0 (N: c0% -> condition%); use it, not the unmatched Success Rate, to compare a condition with c0.

## Model: `gemma4:12b`

| Condition | Split | N | Success Rate | Paired | <=1 Cmd Rate | TLE Rate | Mean Env Calls | Repeat Rate | P(10,0.8) | McNemar p vs c0 | Helped / Harmed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `c0` | OVERALL | 8 | 50.0% | — | 50.0% | 12.5% | 3.25 | 5.8% | 25.0% | — | — |
| `c0` | DB | 8 | 50.0% | — | 50.0% | 12.5% | 3.25 | 5.8% | 25.0% | — | — |

## Model: `qwen2.5:3b`

| Condition | Split | N | Success Rate | Paired | <=1 Cmd Rate | TLE Rate | Mean Env Calls | Repeat Rate | P(10,0.8) | McNemar p vs c0 | Helped / Harmed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `c0` | OVERALL | 40 | 27.5% | — | 50.0% | 0.0% | 2.15 | 12.8% | 50.0% | — | — |
| `c0` | DB | 20 | 40.0% | — | 35.0% | 0.0% | 2.60 | 21.2% | 60.0% | — | — |
| `c0` | OS | 20 | 15.0% | — | 65.0% | 0.0% | 1.70 | 4.3% | 40.0% | — | — |
| `c3` | OVERALL | 40 | 25.0% | 40: 27.5% -> 25.0% | 10.0% | 2.5% | 3.58 | 20.7% | 75.0% | 1.0000 | +4 / -5 |
| `c3` | DB | 20 | 30.0% | 20: 40.0% -> 30.0% | 0.0% | 5.0% | 4.80 | 23.3% | 80.0% | 0.6250 | +1 / -3 |
| `c3` | OS | 20 | 20.0% | 20: 15.0% -> 20.0% | 20.0% | 0.0% | 2.35 | 18.1% | 70.0% | 1.0000 | +3 / -2 |

## Model: `qwen2.5:7b`

| Condition | Split | N | Success Rate | Paired | <=1 Cmd Rate | TLE Rate | Mean Env Calls | Repeat Rate | P(10,0.8) | McNemar p vs c0 | Helped / Harmed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `c0` | OVERALL | 38 | 28.9% | — | 36.8% | 7.9% | 2.53 | 22.3% | 57.9% | — | — |
| `c0` | DB | 20 | 45.0% | — | 10.0% | 0.0% | 2.75 | 31.7% | 85.0% | — | — |
| `c0` | OS | 18 | 11.1% | — | 66.7% | 16.7% | 2.28 | 11.8% | 27.8% | — | — |
