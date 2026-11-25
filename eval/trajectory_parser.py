"""Shared trajectory parser for AgentBench FC (DBBench + OS Interaction).

Used by both:
- Proposal 3 (Failure-aware agent evaluation, McNemar & <=1-cmd diagnostics, failure-record extraction)
- Proposal 1 (Trajectory-aware Bayesian IRT & calibration-subset selection)
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


ENV_TOOLS = {"execute_sql", "bash_action"}
COMMIT_TOOLS = {"commit_final_answer", "answer_action", "finish_action"}
VERIFY_GATE_MARKER = "[TARGETED VERIFICATION CHECK]"

ERROR_PATTERNS = (
    r"\berror executing query\b",
    r"\berror:\s*failed to parse tool call\b",
    r"\berror:\s*sql execution timed out\b",
    r"\binvalid function call\b",
    r"\bsyntax error\b",
    r"\byou have an error in your sql syntax\b",
    r"\bunknown column\b",
    r"\btable .* doesn't exist\b",
    r"\bno such file or directory\b",
    r"\bcommand not found\b",
    r"\bpermission denied\b",
    r"\bcannot access\b",
    r"\btraceback \(most recent call last\)",
    r"\bunrecognized option\b",
    r"\binvalid option\b",
    r"\bmysql.*error\b",
)
_ERROR_RE = re.compile("|".join(ERROR_PATTERNS), re.IGNORECASE)

EMPTY_OUTPUT_STRINGS = {
    "",
    "no response from database.",
    "the output of the os is empty.",
    "[]",
    "{}",
    "none",
    "null",
}


def normalize_command(text: str) -> str:
    """Normalize SQL/bash command strings for duplicate and state matching."""
    return " ".join((text or "").strip().split())


def tokenize_for_rouge(text: str) -> List[str]:
    """Tokenize text into whitespace/punctuation-aware tokens for Rouge-L."""
    return re.findall(r"[A-Za-z0-9_./:-]+|[^\w\s]", (text or "").lower())


def rouge_l_f1(a: str, b: str) -> float:
    """Compute exact token-level Rouge-L F1 similarity in [0, 1]."""
    ta = tokenize_for_rouge(a)
    tb = tokenize_for_rouge(b)
    if not ta or not tb:
        return 0.0
    if ta == tb:
        return 1.0
    # Space-efficient DP for LCS length
    m, n = len(ta), len(tb)
    if m < n:
        ta, tb = tb, ta
        m, n = n, m
    prev = [0] * (n + 1)
    for tok_a in ta:
        curr = [0] * (n + 1)
        for j, tok_b in enumerate(tb, start=1):
            if tok_a == tok_b:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = prev[j] if prev[j] >= curr[j - 1] else curr[j - 1]
        prev = curr
    lcs = prev[n]
    if lcs == 0:
        return 0.0
    prec = lcs / len(ta)
    rec = lcs / len(tb)
    return 2.0 * prec * rec / (prec + rec)


def compute_p_nt(responses: List[str], n: int = 10, threshold: float = 0.8) -> Tuple[float, bool]:
    """AgentBench App. J.2.4 repetition metric P(n, t).

    Examines the last `n` assistant action/response strings and returns
    (max_pairwise_rouge_l, any_pair_ge_threshold).
    """
    window = [r for r in responses[-n:] if r]
    if len(window) < 2:
        return 0.0, False
    max_sim = 0.0
    for i in range(len(window)):
        for j in range(i + 1, len(window)):
            sim = rouge_l_f1(window[i], window[j])
            if sim > max_sim:
                max_sim = sim
    return max_sim, max_sim >= threshold


def is_error_output(text: str) -> bool:
    """Check whether a tool observation indicates an execution/parse error."""
    if not text:
        return False
    return bool(_ERROR_RE.search(text))


_SQL_MUTATING_RE = re.compile(r"^\s*(?:INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|REPLACE|TRUNCATE)\b", re.IGNORECASE)
_BASH_MUTATING_RE = re.compile(
    r"^\s*(?:mkdir|touch|cp|mv|rm|chmod|chown|chgrp|useradd|usermod|userdel|groupadd|ln|cd|export|unset|kill|pkill|systemctl|service)\b|(?<![<>])>(?!=)",
    re.IGNORECASE,
)


def is_mutating_command(tool_name: str, cmd: str) -> bool:
    """Return True for DML/DDL SQL or side-effecting bash commands where empty output is normal."""
    s = (cmd or "").strip()
    if tool_name == "execute_sql":
        return bool(_SQL_MUTATING_RE.match(s))
    if tool_name == "bash_action":
        # Check last statement if chained with && or ;
        last_part = re.split(r"(?:&&|;|\|\|)", s)[-1].strip()
        return bool(_BASH_MUTATING_RE.match(last_part))
    return False


def is_empty_output(text: str) -> bool:
    """Check whether a tool observation is effectively empty."""
    cleaned = (text or "").strip().lower()
    if cleaned in EMPTY_OUTPUT_STRINGS:
        return True
    if cleaned.startswith("the output of the os:"):
        body = cleaned[len("the output of the os:") :].strip()
        return body in EMPTY_OUTPUT_STRINGS
    return False


def is_truncated_output(text: str) -> bool:
    """Check whether OS/DB output was truncated."""
    return "[truncated because the output is too long]" in (text or "").lower()


def infer_task_type(record: Dict[str, Any]) -> str:
    """Infer 'os' or 'db' from task_index, metadata, or tool names."""
    meta = record.get("metadata") or {}
    if meta.get("task_type") in ("os", "db"):
        return meta["task_type"]
    idx = str(record.get("task_index", ""))
    if idx.startswith(("std-", "train-", "dev-")):
        return "os"
    for msg in record.get("messages") or []:
        for tc in msg.get("tool_calls") or []:
            name = (tc.get("function") or {}).get("name", "")
            if name in ("bash_action", "answer_action", "finish_action"):
                return "os"
            if name in ("execute_sql", "commit_final_answer"):
                return "db"
    return "db" if idx.isdigit() else "unknown"


def _parse_tool_args(tc: Dict[str, Any]) -> Tuple[str, Dict[str, Any], str]:
    fn = tc.get("function") or {}
    name = fn.get("name", "")
    raw_args = fn.get("arguments", "{}")
    if isinstance(raw_args, dict):
        args_dict = raw_args
    else:
        try:
            parsed = json.loads(raw_args)
            args_dict = parsed if isinstance(parsed, dict) else {"value": parsed}
        except Exception:
            args_dict = {"raw": str(raw_args)}
    if name == "execute_sql":
        cmd = str(args_dict.get("query") or next(iter(args_dict.values()), ""))
    elif name == "bash_action":
        cmd = str(args_dict.get("script") or next(iter(args_dict.values()), ""))
    elif name == "commit_final_answer":
        cmd = json.dumps(args_dict.get("answers", args_dict), ensure_ascii=False)
    elif name in ("answer_action", "finish_action"):
        cmd = str(args_dict.get("answer") or args_dict.get("thought") or next(iter(args_dict.values()), ""))
    else:
        cmd = json.dumps(args_dict, ensure_ascii=False)
    return name, args_dict, cmd


@dataclass
class TrajectoryFeatures:
    model: str
    condition: str
    task_type: str
    task_index: str
    item_id: str
    run_number: int
    success: int
    status_valid: bool
    num_assistant_turns: int
    num_tool_calls: int
    num_env_calls: int
    num_commit_calls: int
    answered_le_1_cmd: bool
    tle: bool
    exact_repeat_count: int
    exact_repeat_rate: float
    max_rouge_l_p10: float
    rouge_l_repeat_flag: bool
    error_count: int
    empty_output_count: int
    truncated_output_count: int
    no_tool_call_count: int
    parse_error_count: int
    first_error_step: Optional[int]
    steps_to_first_valid_obs: Optional[int]
    verify_gate_fired: bool
    verify_ran_new_command: bool
    verify_changed_answer: bool
    verify_outcome: str
    failure_mode: str
    process_quality: float
    partial_credit_level: int
    total_chars: int
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    temperature: float = 0.0
    env_commands: List[str] = field(default_factory=list)
    error_messages: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def parse_trajectory(record: Dict[str, Any]) -> TrajectoryFeatures:
    """Parse a single JSONL trajectory record into structured step/episode features."""
    model = str(record.get("model", "unknown"))
    meta = record.get("metadata") or {}
    condition = str(record.get("condition") or meta.get("condition") or "c0")
    task_type = infer_task_type(record)
    task_index = str(record.get("task_index", ""))
    item_id = f"{task_type}::{task_index}"
    run_number = int(record.get("run_number", 1) or 1)

    raw_result = record.get("result", 0)
    status_valid = raw_result != "error"
    try:
        success = 1 if float(raw_result) >= 1.0 else 0
    except Exception:
        success = 0

    messages: List[Dict[str, Any]] = record.get("messages") or []
    tool_outputs_by_id: Dict[str, str] = {}
    for msg in messages:
        if msg.get("role") == "tool" and msg.get("tool_call_id"):
            tool_outputs_by_id[str(msg["tool_call_id"])] = str(msg.get("content", ""))

    num_assistant_turns = 0
    num_tool_calls = 0
    num_env_calls = 0
    num_commit_calls = 0
    no_tool_call_count = 0
    parse_error_count = 0
    error_count = 0
    empty_output_count = 0
    truncated_output_count = 0
    first_error_step: Optional[int] = None
    steps_to_first_valid_obs: Optional[int] = None

    env_commands: List[str] = []
    normalized_env_cmds: List[str] = []
    seen_env_cmds = set()
    exact_repeat_count = 0
    action_strings_for_rouge: List[str] = []
    error_messages: List[str] = []
    commit_answers: List[str] = []

    verify_gate_fired = bool(meta.get("verify_gate_fired", False))
    gate_turn_idx: Optional[int] = None
    env_calls_after_gate = 0

    for msg_idx, msg in enumerate(messages):
        role = msg.get("role")
        content = str(msg.get("content") or "")
        if role == "tool" and VERIFY_GATE_MARKER in content:
            verify_gate_fired = True
            if gate_turn_idx is None:
                gate_turn_idx = msg_idx
        if role == "user" and (
            "no tool calls found" in content.lower()
            or "no executable tool calls found" in content.lower()
        ):
            no_tool_call_count += 1

        if role != "assistant":
            continue
        num_assistant_turns += 1
        tcs = msg.get("tool_calls") or []
        if not tcs:
            if content:
                action_strings_for_rouge.append(content)
            continue

        for tc in tcs:
            num_tool_calls += 1
            tc_id = str(tc.get("id", ""))
            name, _, cmd = _parse_tool_args(tc)
            if name in ENV_TOOLS:
                action_strings_for_rouge.append(f"{name}:{cmd}")
            obs = tool_outputs_by_id.get(tc_id, "")

            if VERIFY_GATE_MARKER in obs:
                verify_gate_fired = True
                if gate_turn_idx is None:
                    gate_turn_idx = msg_idx

            if name in ENV_TOOLS:
                num_env_calls += 1
                if gate_turn_idx is not None and msg_idx > gate_turn_idx:
                    env_calls_after_gate += 1
                norm = normalize_command(cmd)
                env_commands.append(cmd)
                normalized_env_cmds.append(norm)
                if norm in seen_env_cmds:
                    exact_repeat_count += 1
                else:
                    seen_env_cmds.add(norm)

                obs_err = is_error_output(obs)
                obs_empty = is_empty_output(obs) and not is_mutating_command(name, cmd)
                obs_trunc = is_truncated_output(obs)
                if "failed to parse tool call" in obs.lower():
                    parse_error_count += 1
                if obs_err:
                    error_count += 1
                    error_messages.append(obs[:200])
                    if first_error_step is None:
                        first_error_step = num_env_calls
                if obs_empty:
                    empty_output_count += 1
                if obs_trunc:
                    truncated_output_count += 1
                if not obs_err and not obs_empty and steps_to_first_valid_obs is None:
                    steps_to_first_valid_obs = num_env_calls

            elif name in COMMIT_TOOLS:
                num_commit_calls += 1
                commit_answers.append(normalize_command(cmd))

    # Determine if verification gate ran and whether answer/commands changed
    verify_ran_new_command = verify_gate_fired and env_calls_after_gate > 0
    first_ans = meta.get("first_answer")
    final_ans = meta.get("final_answer")
    if first_ans is not None and final_ans is not None:
        verify_changed_answer = normalize_command(str(first_ans)) != normalize_command(str(final_ans))
    elif len(commit_answers) >= 2:
        verify_changed_answer = commit_answers[0] != commit_answers[-1]
    else:
        verify_changed_answer = False

    if not verify_gate_fired:
        verify_outcome = "not_triggered"
    else:
        baseline_ans = meta.get("baseline_c0_success")
        if baseline_ans is not None:
            b_ok = bool(int(baseline_ans))
            if not b_ok and success:
                verify_outcome = "helped_wrong_to_right"
            elif b_ok and not success:
                verify_outcome = "harmed_right_to_wrong"
            elif success:
                verify_outcome = "unchanged_right"
            else:
                verify_outcome = "unchanged_wrong"
        else:
            verify_outcome = "changed_success" if (verify_changed_answer and success) else (
                "changed_fail" if verify_changed_answer else ("unchanged_right" if success else "unchanged_wrong")
            )

    round_limit = 8 if task_type == "os" else 15
    # Server finishes on commit_final_answer / answer_action / finish_action unless limit reached.
    # If the last assistant message was a commit call (forwarded) -> not TLE.
    last_commit_forwarded = False
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            tcs = msg.get("tool_calls") or []
            if tcs:
                last_name = (tcs[0].get("function") or {}).get("name", "")
                if last_name in COMMIT_TOOLS:
                    last_commit_forwarded = True
            break
    tle = bool(
        status_valid
        and success == 0
        and not last_commit_forwarded
        and num_env_calls >= min(round_limit, 8 if task_type == "os" else 15)
    )

    exact_repeat_rate = (exact_repeat_count / num_env_calls) if num_env_calls > 0 else 0.0
    max_rouge_l_p10, rouge_l_repeat_flag = compute_p_nt(action_strings_for_rouge, n=10, threshold=0.8)
    answered_le_1_cmd = num_env_calls <= 1

    # Causal / diagnostic failure taxonomy
    if not status_valid:
        failure_mode = "infra_or_api_error"
    elif success == 1:
        failure_mode = "completed"
    elif tle or exact_repeat_count >= 2 or rouge_l_repeat_flag:
        failure_mode = "repeated_action_loop_tle"
    elif answered_le_1_cmd or steps_to_first_valid_obs is None:
        if no_tool_call_count > 0 or parse_error_count > 0:
            failure_mode = "format_or_no_tool_error"
        elif error_count > 0 and num_env_calls >= 1:
            failure_mode = "execution_or_syntax_error"
        else:
            failure_mode = "early_unverified_answer"
    elif no_tool_call_count > 0 or parse_error_count > 0:
        failure_mode = "format_or_no_tool_error"
    elif error_count > 0 and error_count >= max(1, num_env_calls // 2):
        failure_mode = "execution_or_syntax_error"
    else:
        failure_mode = "wrong_answer_after_exploration"

    # Process quality in [0, 1] (purely process-based, without penalizing concise 1-shot SQL/bash)
    valid_ratio = (
        max(0, num_env_calls - error_count - empty_output_count) / num_env_calls
        if num_env_calls > 0
        else 0.0
    )
    clean_exec_signal = 1.0 if (num_env_calls >= 1 and error_count == 0 and not tle) else (0.4 if num_env_calls >= 1 else 0.0)
    non_loop_signal = max(0.0, 1.0 - exact_repeat_rate)
    process_quality = round(
        0.45 * valid_ratio + 0.30 * clean_exec_signal + 0.25 * non_loop_signal,
        4,
    )

    if success == 1:
        partial_credit_level = 3
    elif num_env_calls >= 2 and steps_to_first_valid_obs is not None and not tle and exact_repeat_count == 0:
        partial_credit_level = 2
    elif num_env_calls >= 1 and steps_to_first_valid_obs is not None:
        partial_credit_level = 1
    else:
        partial_credit_level = 0

    total_chars = sum(len(str(m.get("content") or "")) for m in messages)
    return TrajectoryFeatures(
        model=model,
        condition=condition,
        task_type=task_type,
        task_index=task_index,
        item_id=item_id,
        run_number=run_number,
        success=success,
        status_valid=status_valid,
        num_assistant_turns=num_assistant_turns,
        num_tool_calls=num_tool_calls,
        num_env_calls=num_env_calls,
        num_commit_calls=num_commit_calls,
        answered_le_1_cmd=answered_le_1_cmd,
        tle=tle,
        exact_repeat_count=exact_repeat_count,
        exact_repeat_rate=round(exact_repeat_rate, 4),
        max_rouge_l_p10=round(max_rouge_l_p10, 4),
        rouge_l_repeat_flag=rouge_l_repeat_flag,
        error_count=error_count,
        empty_output_count=empty_output_count,
        truncated_output_count=truncated_output_count,
        no_tool_call_count=no_tool_call_count,
        parse_error_count=parse_error_count,
        first_error_step=first_error_step,
        steps_to_first_valid_obs=steps_to_first_valid_obs,
        verify_gate_fired=verify_gate_fired,
        verify_ran_new_command=verify_ran_new_command,
        verify_changed_answer=verify_changed_answer,
        verify_outcome=verify_outcome,
        failure_mode=failure_mode,
        process_quality=process_quality,
        partial_credit_level=partial_credit_level,
        total_chars=total_chars,
        prompt_tokens=int(meta.get("prompt_tokens", 0) or 0),
        completion_tokens=int(meta.get("completion_tokens", 0) or 0),
        latency_s=float(meta.get("latency_s", 0.0) or 0.0),
        temperature=float(meta.get("temperature", 0.0) or 0.0),
        env_commands=env_commands,
        error_messages=error_messages,
    )


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def parse_jsonl_file(path: str | Path, condition_override: Optional[str] = None) -> List[TrajectoryFeatures]:
    feats = []
    for rec in load_jsonl(path):
        if condition_override:
            rec = dict(rec)
            rec["condition"] = condition_override
        feats.append(parse_trajectory(rec))
    return feats


def extract_failure_records(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract generalized (state, action, outcome, advisory) records from dev/source logs.

    Supports cross-task transfer (Section 6 of Task4_Iteration1_Group12.pdf) without leaking
    test instances.
    """
    extracted: List[Dict[str, Any]] = []
    seen_keys = set()
    for rec in records:
        feat = parse_trajectory(rec)
        if feat.success == 1:
            continue
        messages = rec.get("messages") or []
        tool_outputs_by_id = {
            str(m["tool_call_id"]): str(m.get("content", ""))
            for m in messages
            if m.get("role") == "tool" and m.get("tool_call_id")
        }
        prev_state = "INITIAL_STATE"
        for m in messages:
            if m.get("role") != "assistant":
                continue
            for tc in m.get("tool_calls") or []:
                tc_id = str(tc.get("id", ""))
                name, _, cmd = _parse_tool_args(tc)
                obs = tool_outputs_by_id.get(tc_id, "")
                if name in ENV_TOOLS:
                    outcome = None
                    if is_error_output(obs):
                        outcome = "execution_error"
                    elif is_empty_output(obs):
                        outcome = "empty_output"
                    elif is_truncated_output(obs):
                        outcome = "truncated_output"
                    if outcome:
                        key = (feat.task_type, outcome, normalize_command(cmd)[:80])
                        if key not in seen_keys:
                            seen_keys.add(key)
                            extracted.append({
                                "source_task": feat.task_type,
                                "state_sig": prev_state[:160],
                                "action": normalize_command(cmd)[:160],
                                "outcome": outcome,
                                "observation_snippet": obs[:160],
                                "advisory": _generalized_advisory(feat.task_type, outcome, cmd),
                            })
                    prev_state = f"{normalize_command(cmd)[:80]} -> {obs[:80]}"
        if feat.answered_le_1_cmd:
            key = (feat.task_type, "early_answer", "le_1_cmd")
            if key not in seen_keys:
                seen_keys.add(key)
                extracted.append({
                    "source_task": feat.task_type,
                    "state_sig": "1_OR_FEWER_ENV_COMMANDS",
                    "action": "COMMIT_ANSWER",
                    "outcome": "early_unverified_answer",
                    "observation_snippet": "",
                    "advisory": "Cross-task warning: previous episodes failed after answering with <=1 command or partial output without independent verification.",
                })
    return extracted


def _generalized_advisory(task_type: str, outcome: str, cmd: str) -> str:
    if outcome == "execution_error":
        return f"Prior {task_type.upper()} failure: command/query `{normalize_command(cmd)[:70]}` caused an execution error; verify syntax/schema/paths before retrying."
    if outcome == "empty_output":
        return f"Prior {task_type.upper()} failure: `{normalize_command(cmd)[:70]}` returned empty output; check filters, case sensitivity, or target path."
    if outcome == "truncated_output":
        return f"Prior {task_type.upper()} failure: `{normalize_command(cmd)[:70]}` output was truncated; inspect full counts or filtered slices instead of relying on partial output."
    return f"Prior {task_type.upper()} failure on `{normalize_command(cmd)[:70]}`."
