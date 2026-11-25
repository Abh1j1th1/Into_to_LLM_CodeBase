"""End-to-end and unit tests for Iteration 2 (Proposal 3 + Proposal 1) pipelines."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from aiohttp import web

EVAL_DIR = Path(__file__).resolve().parents[1]
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import analyze_proposal1
import analyze_proposal3
import conditions
import irt
import server_agent
import trajectory_parser


# ---------------------------------------------------------------------------
# 1. Trajectory Parser Tests
# ---------------------------------------------------------------------------

def test_rouge_l_and_p_nt():
    assert trajectory_parser.rouge_l_f1("SELECT COUNT(*) FROM t", "SELECT COUNT(*) FROM t") == 1.0
    sim, flag = trajectory_parser.compute_p_nt(
        ["bash_action:ls -la /var/log", "bash_action:ls -la /var/log", "bash_action:cat /etc/hosts"],
        n=10,
        threshold=0.8,
    )
    assert sim == 1.0 and flag is True


def test_parse_trajectory_early_answer_and_verify_gate():
    # Plain OS episode answering after 1 command (failure)
    rec_c0 = {
        "model": "qwen3:8b",
        "condition": "c0",
        "task_index": "std-001-0",
        "run_number": 1,
        "result": 0,
        "messages": [
            {"role": "system", "content": "OS prompt"},
            {"role": "user", "content": "Count Alice's sales in trades.log"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "1", "type": "function", "function": {"name": "bash_action", "arguments": '{"script": "tail -n 100 trades.log"}'}}],
            },
            {"role": "tool", "tool_call_id": "1", "content": "The output of the OS:\n\nAlice BUY 1\nAlice SELL 2"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "2", "type": "function", "function": {"name": "answer_action", "arguments": '{"answer": "10"}'}}],
            },
        ],
        "metadata": {"condition": "c0", "task_type": "os"},
    }
    f0 = trajectory_parser.parse_trajectory(rec_c0)
    assert f0.task_type == "os"
    assert f0.num_env_calls == 1
    assert f0.answered_le_1_cmd is True
    assert f0.failure_mode == "early_unverified_answer"
    assert f0.partial_credit_level == 1

    # Targeted verify gate episode: held back first commit, ran independent grep, changed answer, passed
    rec_c3 = {
        "model": "qwen3:8b",
        "condition": "c3",
        "task_index": "std-001-0",
        "run_number": 1,
        "result": 1,
        "messages": [
            {"role": "system", "content": "OS prompt"},
            {"role": "user", "content": "Count Alice's sales in trades.log"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "1", "type": "function", "function": {"name": "bash_action", "arguments": '{"script": "tail -n 100 trades.log"}'}}],
            },
            {"role": "tool", "tool_call_id": "1", "content": "The output of the OS:\n\nAlice BUY 1\nAlice SELL 2"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "2", "type": "function", "function": {"name": "answer_action", "arguments": '{"answer": "10"}'}}],
            },
            {"role": "tool", "tool_call_id": "2", "content": f"{trajectory_parser.VERIFY_GATE_MARKER} Answer held before submission."},
            {"role": "user", "content": f"{trajectory_parser.VERIFY_GATE_MARKER} Run a second independent check."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "3", "type": "function", "function": {"name": "bash_action", "arguments": '{"script": "grep -c \"Alice SELL\" trades.log"}'}}],
            },
            {"role": "tool", "tool_call_id": "3", "content": "The output of the OS:\n\n4"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "4", "type": "function", "function": {"name": "answer_action", "arguments": '{"answer": "4"}'}}],
            },
        ],
        "metadata": {
            "condition": "c3",
            "task_type": "os",
            "verify_gate_fired": True,
            "first_answer": "10",
            "final_answer": "4",
        },
    }
    f3 = trajectory_parser.parse_trajectory(rec_c3)
    assert f3.success == 1
    assert f3.num_env_calls == 2
    assert f3.answered_le_1_cmd is False
    assert f3.verify_gate_fired is True
    assert f3.verify_ran_new_command is True
    assert f3.verify_changed_answer is True
    assert f3.failure_mode == "completed"
    assert f3.partial_credit_level == 3


def test_extract_failure_records_for_cross_task():
    rec = {
        "model": "qwen3:8b",
        "condition": "c0",
        "task_index": "0",
        "run_number": 1,
        "result": 0,
        "messages": [
            {"role": "user", "content": "Query sales"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "1", "type": "function", "function": {"name": "execute_sql", "arguments": '{"query": "SELECT bad_col FROM t"}'}}],
            },
            {"role": "tool", "tool_call_id": "1", "content": "Error executing query: Unknown column 'bad_col'"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "2", "type": "function", "function": {"name": "commit_final_answer", "arguments": '{"answers": ["0"]}'}}],
            },
        ],
    }
    extracted = trajectory_parser.extract_failure_records([rec])
    outcomes = {r["outcome"] for r in extracted}
    assert "execution_error" in outcomes
    assert "early_unverified_answer" in outcomes


# ---------------------------------------------------------------------------
# 2. Condition Logic & Budget Safety Tests
# ---------------------------------------------------------------------------

def test_verify_gate_respects_remaining_round_budget():
    state = conditions.EpisodeState(condition="c3", task_type="os", round_limit=8, env_calls=7)
    commit_msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "answer_action", "arguments": '{"answer": "5"}'}}],
    }
    # At 7/8 rounds used, firing verify gate would force TLE on round 8, so it must skip safely
    assert conditions.should_fire_verify_gate(state, commit_msg) is False
    assert state.verify_gate_skipped_low_budget is True

    state_ok = conditions.EpisodeState(condition="c3", task_type="os", round_limit=8, env_calls=1)
    assert conditions.should_fire_verify_gate(state_ok, commit_msg) is True


# ---------------------------------------------------------------------------
# 3. End-to-End Async HTTP Test: server_agent.py x Mock Controller x Mock LLM
# ---------------------------------------------------------------------------

def test_server_agent_all_conditions_e2e(tmp_path):
    async def run_e2e():
        sessions = {}
        llm_requests = []

        async def handle_sync(req):
            return web.json_response({"ok": True})

        async def handle_indices(req):
            name = req.query.get("name", "")
            if name == "os-std":
                return web.json_response(["std-001-0", "std-001-1"])
            return web.json_response([0, 1])

        async def handle_start(req):
            body = await req.json()
            sid = f"sid-{len(sessions) + 1}"
            task_name = body["name"]
            idx = str(body["index"])
            if task_name == "os-std":
                tools = [
                    {"type": "function", "function": {"name": "bash_action", "parameters": {"type": "object"}}},
                    {"type": "function", "function": {"name": "answer_action", "parameters": {"type": "object"}}},
                ]
                prompt = "Count Alice SELL lines in trades.log" if idx == "std-001-0" else "Check /tmp/missing"
            else:
                tools = [
                    {"type": "function", "function": {"name": "execute_sql", "parameters": {"type": "object"}}},
                    {"type": "function", "function": {"name": "commit_final_answer", "parameters": {"type": "object"}}},
                ]
                prompt = "Total revenue from sales_2025"

            sessions[sid] = {"task_name": task_name, "index": idx, "rounds": 0}
            return web.json_response(
                {
                    "messages": [
                        {"role": "system", "content": "Agent prompt"},
                        {"role": "user", "content": prompt},
                    ],
                    "tools": tools,
                    "finish": False,
                },
                headers={"session_id": sid},
            )

        async def handle_interact(req):
            sid = req.headers["session_id"]
            sess = sessions[sid]
            sess["rounds"] += 1
            body = await req.json()
            msg = body["messages"][0]
            # Verify non-standard keys like 'reasoning' were stripped
            assert "reasoning" not in msg

            tc = (msg.get("tool_calls") or [{}])[0]
            fn = (tc.get("function") or {}).get("name", "")
            args = json.loads((tc.get("function") or {}).get("arguments", "{}") or "{}")
            tc_id = tc.get("id", "t1")

            if fn == "answer_action":
                ans = str(args.get("answer", ""))
                reward = 1 if (sess["index"] == "std-001-0" and ans == "4") or (sess["index"] == "std-001-1" and ans == "0") else 0
                return web.json_response({"messages": [], "finish": True, "reward": reward})
            if fn == "commit_final_answer":
                ans = args.get("answers", [])
                reward = 1 if ans == ["42"] else 0
                return web.json_response({"messages": [], "finish": True, "reward": reward})

            if fn == "bash_action":
                cmd = args.get("script", "")
                if "tail" in cmd:
                    obs = "The output of the OS:\n\nAlice BUY 1\nAlice SELL 2"
                elif "grep" in cmd:
                    obs = "The output of the OS:\n\n4"
                elif "missing" in cmd:
                    obs = "The output of the OS:\n\nls: cannot access '/tmp/missing': No such file or directory"
                else:
                    obs = "The output of the OS is empty."
                return web.json_response(
                    {"messages": [{"role": "tool", "tool_call_id": tc_id, "content": obs}], "finish": False, "reward": 0}
                )

            if fn == "execute_sql":
                q = args.get("query", "")
                if "bad_col" in q:
                    obs = "Error executing query: Unknown column 'bad_col'"
                else:
                    obs = "[(42,)]"
                return web.json_response(
                    {"messages": [{"role": "tool", "tool_call_id": tc_id, "content": obs}], "finish": False, "reward": 0}
                )
            return web.json_response({"messages": [], "finish": True, "reward": 0})

        async def handle_chat(req):
            body = await req.json()
            llm_requests.append(body)
            msgs = body["messages"]
            # Check no raw reasoning leaked into subsequent messages
            for m in msgs:
                assert "reasoning" not in m

            tools = [t["function"]["name"] for t in body.get("tools") or []]
            last_text = str(msgs[-1].get("content", ""))
            user_prompt = str(msgs[1].get("content", ""))

            # OS behavior:
            # - std-001-0: initially rushes with tail -> wrong answer 10; if verify gate fires, runs grep -> 4!
            # - std-001-1: checks /tmp/missing -> No such file; if failure memory triggers, checks existence -> 0!
            if "bash_action" in tools:
                if "Alice" in user_prompt:
                    if trajectory_parser.VERIFY_GATE_MARKER in last_text:
                        msg = {
                            "role": "assistant",
                            "content": "",
                            "reasoning": "Verifying with grep",
                            "tool_calls": [{"id": f"call_{len(msgs)}", "type": "function", "function": {"name": "bash_action", "arguments": '{"script": "grep -c SELL trades.log"}'}}],
                        }
                    elif "4" in last_text and any("grep" in json.dumps(m) for m in msgs):
                        msg = {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [{"id": f"call_{len(msgs)}", "type": "function", "function": {"name": "answer_action", "arguments": '{"answer": "4"}'}}],
                        }
                    elif len(msgs) <= 3:
                        msg = {
                            "role": "assistant",
                            "content": "",
                            "reasoning": "Single quick command",
                            "tool_calls": [
                                {"id": f"call_{len(msgs)}", "type": "function", "function": {"name": "bash_action", "arguments": '{"script": "tail -n 100 trades.log"}'}},
                                {"id": f"call_{len(msgs)}_extra", "type": "function", "function": {"name": "bash_action", "arguments": '{"script": "echo extra"}'}},
                            ],
                        }
                    else:
                        msg = {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [{"id": f"call_{len(msgs)}", "type": "function", "function": {"name": "answer_action", "arguments": '{"answer": "10"}'}}],
                        }
                else:
                    if trajectory_parser.VERIFY_GATE_MARKER in last_text:
                        msg = {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [{"id": f"call_{len(msgs)}", "type": "function", "function": {"name": "bash_action", "arguments": '{"script": "ls /tmp/missing || true"}'}}],
                        }
                    elif any("|| true" in json.dumps(m) for m in msgs):
                        msg = {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [{"id": f"call_{len(msgs)}", "type": "function", "function": {"name": "answer_action", "arguments": '{"answer": "0"}'}}],
                        }
                    elif "No such file" in last_text and ("[FAILURE MEMORY]" in last_text or "[REPEATED-ACTION WARNING]" in last_text):
                        msg = {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [{"id": f"call_{len(msgs)}", "type": "function", "function": {"name": "answer_action", "arguments": '{"answer": "0"}'}}],
                        }
                    elif "No such file" in last_text:
                        seen_missing = sum("missing" in json.dumps(m) for m in msgs if m.get("role") == "assistant")
                        if seen_missing < 2:
                            msg = {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [{"id": f"call_{len(msgs)}", "type": "function", "function": {"name": "bash_action", "arguments": '{"script": "ls /tmp/missing"}'}}],
                            }
                        else:
                            msg = {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [{"id": f"call_{len(msgs)}", "type": "function", "function": {"name": "answer_action", "arguments": '{"answer": "99"}'}}],
                            }
                    else:
                        msg = {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [{"id": f"call_{len(msgs)}", "type": "function", "function": {"name": "bash_action", "arguments": '{"script": "ls /tmp/missing"}'}}],
                        }
            else:
                # DB behavior: bad_col first; if failure memory triggers -> fixed query -> 42; else loops on bad_col
                if "[FAILURE MEMORY]" in last_text or "[(42,)]" in last_text:
                    if "[(42,)]" in last_text:
                        msg = {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [{"id": f"call_{len(msgs)}", "type": "function", "function": {"name": "commit_final_answer", "arguments": '{"answers": ["42"]}'}}],
                        }
                    else:
                        msg = {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [{"id": f"call_{len(msgs)}", "type": "function", "function": {"name": "execute_sql", "arguments": '{"query": "SELECT SUM(rev) FROM sales_2025"}'}}],
                        }
                elif trajectory_parser.VERIFY_GATE_MARKER in last_text:
                    msg = {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"id": f"call_{len(msgs)}", "type": "function", "function": {"name": "execute_sql", "arguments": '{"query": "SELECT SUM(rev) FROM sales_2025"}'}}],
                    }
                else:
                    msg = {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"id": f"call_{len(msgs)}", "type": "function", "function": {"name": "execute_sql", "arguments": '{"query": "SELECT bad_col FROM sales_2025"}'}}],
                    }

            return web.json_response({
                "choices": [{"message": msg}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 20},
            })

        app = web.Application()
        app.router.add_post("/api/sync_all", handle_sync)
        app.router.add_get("/api/get_indices", handle_indices)
        app.router.add_post("/api/start_sample", handle_start)
        app.router.add_post("/api/interact", handle_interact)
        app.router.add_post("/v1/chat/completions", handle_chat)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]

        try:
            for cond in ("c0", "c1", "c2", "c3", "c4"):
                out_os = tmp_path / f"os-std__{cond}__t0.jsonl"
                args = server_agent.argparse_args_for_test if hasattr(server_agent, "argparse_args_for_test") else None
                ns = type("Args", (), {
                    "model": "mock-qwen3:8b",
                    "model2": "",
                    "model_api_base": f"http://127.0.0.1:{port}/v1",
                    "model2_api_base": "",
                    "model_api_key": "test",
                    "temperature": 0.0,
                    "run_all": False,
                    "truncate_multiple_toolcall": True,
                    "controller": f"http://127.0.0.1:{port}/api",
                    "concurrency": 2,
                    "runs": 1,
                    "output_dir": str(tmp_path),
                    "output_file": str(out_os),
                    "extra_body": "{}",
                    "limit": None,
                    "sample_frac": None,
                    "seed": 42,
                    "condition": cond,
                    "memory_bank": None,
                    "max_turns": 24,
                    "max_completion_tokens": 256,
                    "strip_reasoning": True,
                    "verbose": False,
                    "range": None,
                    "index": None,
                    "use_id": False,
                    "task_name": ["os-std"],
                })()
                server_agent.args = ns
                await server_agent.run_tasks(ns)

                out_db = tmp_path / f"dbbench-std__{cond}__t0.jsonl"
                ns.output_file = str(out_db)
                ns.task_name = ["dbbench-std"]
                await server_agent.run_tasks(ns)
        finally:
            await runner.cleanup()

    asyncio.run(run_e2e())

    paths = analyze_proposal3.collect_jsonl_paths([str(tmp_path)])
    report = analyze_proposal3.analyze_files(paths)
    m = report["models"]["mock-qwen3:8b"]
    # c0 fails OS std-001-0 (answered in <=1 cmd) and DB (bad_col loop) -> 0.0%
    assert m["conditions"]["c0"]["overall"]["success_rate"] == 0.0
    # c3 fires verify gate on OS std-001-0 and flips 0 -> 1
    assert m["conditions"]["c3"]["os"]["verify_gate_fired_rate"] > 0.0
    assert m["conditions"]["c3"]["os"]["success_rate"] > m["conditions"]["c0"]["os"]["success_rate"]
    # c4 combines failure memory + verify gate -> solves both OS items and DB items (100%)
    assert m["conditions"]["c4"]["overall"]["success_rate"] == 1.0
    assert m["mcnemar_vs_c0"]["c4"]["overall"]["helped_wrong_to_right"] == 4
    assert m["mcnemar_vs_c0"]["c4"]["overall"]["harmed_right_to_wrong"] == 0


# ---------------------------------------------------------------------------
# 4. Proposal 3 McNemar Exact Test Sanity Check
# ---------------------------------------------------------------------------

def test_mcnemar_exact():
    base = {f"db::{i}": (1 if i < 2 else 0) for i in range(20)}
    cond = {f"db::{i}": (1 if i < 12 else 0) for i in range(20)}
    res = analyze_proposal3.mcnemar_test(base, cond)
    assert res["helped_wrong_to_right"] == 10
    assert res["harmed_right_to_wrong"] == 0
    assert res["mcnemar_exact_p"] < 0.01


# ---------------------------------------------------------------------------
# 5. Proposal 1 IRT & Calibration-Subset Pipeline Test (5 models x 444 items)
# ---------------------------------------------------------------------------

def test_proposal1_irt_444_items():
    rng = np.random.default_rng(123)
    true_thetas = np.array([-1.2, -0.4, 0.2, 0.8, 1.4])
    models = [f"model_{i}" for i in range(5)]
    n_db, n_os = 300, 144
    I = n_db + n_os
    item_envs = ["db"] * n_db + ["os"] * n_os
    items = [f"db::{i}" for i in range(n_db)] + [f"os::{i}" for i in range(n_os)]

    b_true = np.concatenate([rng.normal(-0.3, 1.0, n_db), rng.normal(0.6, 1.0, n_os)])
    a_true = np.exp(rng.normal(0.0, 0.3, I))

    feats = []
    for m_idx, m_name in enumerate(models):
        for i_idx, it in enumerate(items):
            eta = a_true[i_idx] * (true_thetas[m_idx] - b_true[i_idx])
            p = 1.0 / (1.0 + np.exp(-eta))
            # Simulated 3 runs for first 30 items to test variance quantification
            n_runs = 3 if i_idx < 30 else 1
            for r in range(1, n_runs + 1):
                y = int(rng.random() < p)
                # Trajectory process quality correlates with latent eta even when y == 0
                z_latent = 1.0 / (1.0 + np.exp(-(0.5 + 0.8 * eta + rng.normal(0, 0.35))))
                env_calls = 3 if y == 1 else (1 if z_latent < 0.4 else 2)
                feats.append(
                    trajectory_parser.TrajectoryFeatures(
                        model=m_name,
                        condition="c0",
                        task_type=item_envs[i_idx],
                        task_index=it.split("::")[1],
                        item_id=it,
                        run_number=r,
                        success=y,
                        status_valid=True,
                        num_assistant_turns=env_calls + 1,
                        num_tool_calls=env_calls + 1,
                        num_env_calls=env_calls,
                        num_commit_calls=1,
                        answered_le_1_cmd=env_calls <= 1,
                        tle=False,
                        exact_repeat_count=0 if y == 1 else 1,
                        exact_repeat_rate=0.0 if y == 1 else 0.3,
                        max_rouge_l_p10=0.2,
                        rouge_l_repeat_flag=False,
                        error_count=0 if y == 1 else 1,
                        empty_output_count=0,
                        truncated_output_count=0,
                        no_tool_call_count=0,
                        parse_error_count=0,
                        first_error_step=None if y == 1 else 1,
                        steps_to_first_valid_obs=1,
                        verify_gate_fired=False,
                        verify_ran_new_command=False,
                        verify_changed_answer=False,
                        verify_outcome="not_triggered",
                        failure_mode="completed" if y == 1 else "wrong_answer_after_exploration",
                        process_quality=round(float(z_latent), 4),
                        partial_credit_level=3 if y == 1 else (2 if z_latent > 0.5 else 1),
                        total_chars=500,
                        prompt_tokens=100,
                        completion_tokens=40,
                        latency_s=1.0,
                    )
                )

    report = analyze_proposal1.analyze_proposal1(feats, budgets=(0.10, 0.20, 0.30), n_seeds=3, n_boot=30, seed=42)
    assert report["n_models"] == 5
    assert report["n_items"] == 444
    assert report["env_counts"] == {"db": 300, "os": 144}

    # Check theta ordering recovers true model ordering
    thetas_est = {row["model"]: row["traj_2pl_theta"] for row in report["models"]}
    ordered = [thetas_est[f"model_{i}"] for i in range(5)]
    assert ordered == sorted(ordered)

    # Check held-out CV & trajectory-aware beats or stays competitive on Brier/log-loss
    assert "traj_2pl" in report["heldout_cv"]
    assert report["heldout_cv"]["traj_2pl"]["log_loss"] <= report["heldout_cv"]["outcome_2pl"]["log_loss"] + 0.03

    # Check variance quantification ran on the 30 repeated items
    assert report["run_to_run_variance"]["model_0"]["n_repeated_items"] == 30
