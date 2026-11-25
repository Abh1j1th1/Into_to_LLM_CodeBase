"""Proposal 3 experimental conditions for AgentBench FC (DBBench + OS Interaction).

Conditions (all client-side; tasks and scoring remain unchanged):
- c0: Plain agent baseline
- c1: Simple repeated-action warning
- c2: Within-episode state-conditioned failure memory (+ optional cross-task memory bank)
- c3: Targeted verify gate (holds back first commit; requires condition mapping + independent check)
- c4: Failure memory + targeted verify gate (c2 + c3)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from trajectory_parser import (
    COMMIT_TOOLS,
    ENV_TOOLS,
    VERIFY_GATE_MARKER,
    _parse_tool_args,
    is_empty_output,
    is_error_output,
    is_mutating_command,
    is_truncated_output,
    normalize_command,
    rouge_l_f1,
)

VALID_CONDITIONS = ("c0", "c1", "c2", "c3", "c4")

CONDITION_DESCRIPTIONS = {
    "c0": "Plain baseline agent",
    "c1": "Repeated-action warning",
    "c2": "State-conditioned failure memory",
    "c3": "Targeted verify gate",
    "c4": "Failure memory + targeted verify gate",
}


@dataclass
class FailureRecord:
    state_sig: str
    action_sig: str
    outcome_type: str
    summary: str
    step: int


@dataclass
class EpisodeState:
    condition: str
    task_type: str  # "db" or "os"
    round_limit: int
    env_calls: int = 0
    last_state_sig: str = "INITIAL_STATE"
    seen_actions: Dict[str, List[Tuple[int, str]]] = field(default_factory=dict)
    recent_commands: List[str] = field(default_factory=list)
    failure_records: List[FailureRecord] = field(default_factory=list)
    failure_memory_hits: int = 0
    repeat_warnings_issued: int = 0
    verify_gate_fired: bool = False
    verify_gate_skipped_low_budget: bool = False
    first_answer: Optional[str] = None
    final_answer: Optional[str] = None
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cross_task_injected: bool = False


def infer_task_type_from_name(task_name: str, tools: Optional[List[dict]] = None) -> str:
    low = (task_name or "").lower()
    if "os" in low:
        return "os"
    if "db" in low:
        return "db"
    for t in tools or []:
        fn = (t.get("function") or {}).get("name", "")
        if fn in ("bash_action", "answer_action", "finish_action"):
            return "os"
        if fn in ("execute_sql", "commit_final_answer"):
            return "db"
    return "db"


def default_round_limit(task_type: str) -> int:
    return 8 if task_type == "os" else 15


def load_memory_bank(path: Optional[str | Path], target_task_type: str, max_items: int = 3) -> List[Dict[str, Any]]:
    """Load generalized cross-task/dev failure records from a JSON file."""
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Memory bank file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return []
    # Prefer records matching target task if available, else cross-task transfer records
    prioritized = [r for r in data if r.get("advisory")]
    return prioritized[:max_items]


def build_cross_task_preamble(memory_items: List[Dict[str, Any]]) -> Optional[str]:
    if not memory_items:
        return None
    lines = ["[FAILURE MEMORY - PRIOR EPISODE LESSONS]"]
    for idx, item in enumerate(memory_items[:3], start=1):
        lines.append(f"{idx}. {item['advisory']}")
    lines.append("Keep these pitfalls in mind while choosing commands and verifying your answer.")
    return "\n".join(lines)


def sanitize_message_for_api(msg: Dict[str, Any], strip_reasoning: bool = True) -> Dict[str, Any]:
    """Return a clean OpenAI-schema message dict safe for both OpenAI APIs and AgentRL controller."""
    allowed_keys = {"role", "content", "tool_calls", "tool_call_id", "name"}
    cleaned = {}
    for k, v in msg.items():
        if v is None:
            continue
        if strip_reasoning and k not in allowed_keys:
            continue
        cleaned[k] = v
    if "content" not in cleaned:
        cleaned["content"] = ""
    return cleaned


def is_commit_message(message: Dict[str, Any]) -> bool:
    for tc in message.get("tool_calls") or []:
        name = (tc.get("function") or {}).get("name", "")
        if name in COMMIT_TOOLS:
            return True
    return False


def extract_commit_answer(message: Dict[str, Any]) -> Optional[str]:
    for tc in message.get("tool_calls") or []:
        name, _, cmd = _parse_tool_args(tc)
        if name in COMMIT_TOOLS:
            return cmd
    return None


def should_fire_verify_gate(state: EpisodeState, message: Dict[str, Any]) -> bool:
    """Decide whether to hold back the agent's first answer submission for targeted verification."""
    if state.condition not in ("c3", "c4"):
        return False
    if state.verify_gate_fired:
        return False
    if not is_commit_message(message):
        return False
    # Need >= 2 remaining server rounds (1 for independent verify command + 1 to re-submit)
    remaining = state.round_limit - state.env_calls
    if remaining < 2:
        state.verify_gate_skipped_low_budget = True
        return False
    return True


def build_verify_gate_messages(
    state: EpisodeState,
    held_message: Dict[str, Any],
    original_question: str = "",
) -> List[Dict[str, Any]]:
    """Construct synthetic tool responses + targeted verification prompt for held-back commit."""
    state.verify_gate_fired = True
    state.first_answer = extract_commit_answer(held_message)

    extra_messages: List[Dict[str, Any]] = []
    tool_calls = held_message.get("tool_calls") or []
    commit_names = []
    for tc in tool_calls:
        tc_id = str(tc.get("id", "verify_hold"))
        fn_name = (tc.get("function") or {}).get("name", "commit")
        commit_names.append(fn_name)
        extra_messages.append({
            "role": "tool",
            "tool_call_id": tc_id,
            "content": f"{VERIFY_GATE_MARKER} Answer held before submission. Verify before calling {fn_name} again.",
        })

    env_tool = "execute_sql" if state.task_type == "db" else "bash_action"
    commit_tool = "commit_final_answer" if state.task_type == "db" else (commit_names[0] if commit_names else "answer_action")

    lines = [f"{VERIFY_GATE_MARKER} Before your answer `{state.first_answer}` is submitted, perform this targeted check:"]
    if state.env_calls <= 1:
        lines.append(
            f"- WARNING: You have run only {state.env_calls} `{env_tool}` command(s) so far. "
            "Do NOT guess or submit from a single partial output (e.g., head/tail or unverified query)."
        )
    lines.extend([
        "1. List every condition and constraint in the question (target entity/user, specific action/status filter, time/range, and exact output format such as a single number vs. list).",
        "2. Point to the exact command/query output you already observed that satisfies each condition. If any file/table was truncated or only partially read, inspect it completely.",
        f"3. Run a SECOND, INDEPENDENT `{env_tool}` command/query (using a different filter, aggregation, or verification check) to confirm your count, value, or database state.",
        f"4. Only after checking the new output, call `{commit_tool}` with the verified final answer.",
    ])
    if original_question:
        lines.append(f"Question reminder: {original_question[:300]}")

    extra_messages.append({"role": "user", "content": "\n".join(lines)})
    return extra_messages


def process_env_interaction(
    state: EpisodeState,
    assistant_message: Dict[str, Any],
    server_messages: List[Dict[str, Any]],
) -> Optional[str]:
    """Update episode state after an environment turn and return optional advisory warning."""
    tool_outputs_by_id = {
        str(m.get("tool_call_id", "")): str(m.get("content", ""))
        for m in server_messages
        if m.get("role") == "tool"
    }

    advisories: List[str] = []
    for tc in assistant_message.get("tool_calls") or []:
        tc_id = str(tc.get("id", ""))
        name, _, cmd = _parse_tool_args(tc)
        if name not in ENV_TOOLS:
            if name in COMMIT_TOOLS:
                state.final_answer = cmd
            continue

        state.env_calls += 1
        norm_cmd = normalize_command(cmd)
        obs = tool_outputs_by_id.get(tc_id, "")
        obs_err = is_error_output(obs)
        obs_empty = is_empty_output(obs) and not is_mutating_command(name, cmd)
        obs_trunc = is_truncated_output(obs)

        prev_occurrences = state.seen_actions.get(norm_cmd, [])
        is_exact_repeat = len(prev_occurrences) > 0
        is_near_repeat = (
            not is_exact_repeat
            and len(state.recent_commands) > 0
            and max(rouge_l_f1(norm_cmd, prev) for prev in state.recent_commands[-3:]) >= 0.85
        )

        # Condition c1: simple repeated-action warning
        if state.condition == "c1" and (is_exact_repeat or is_near_repeat):
            state.repeat_warnings_issued += 1
            advisories.append(
                "[REPEATED-ACTION WARNING] You already ran this exact or near-identical command/query earlier. "
                "Do not repeat the same step—inspect the output above and try a different command or verify your assumptions."
            )

        # Conditions c2 and c4: state-conditioned failure memory
        if state.condition in ("c2", "c4"):
            outcome_type: Optional[str] = None
            reason_msg: Optional[str] = None

            if obs_err:
                outcome_type = "execution_error"
                reason_msg = f"Command `{norm_cmd[:80]}` produced an error: {obs.strip()[:140]}. Fix syntax, column/table names, or paths."
            elif obs_empty:
                outcome_type = "empty_output"
                reason_msg = f"Command `{norm_cmd[:80]}` returned empty output. Broaden filters, check case/spelling, or verify file/table existence."
            elif obs_trunc:
                outcome_type = "truncated_output"
                reason_msg = f"Command `{norm_cmd[:80]}` output was truncated. Do not infer totals from partial output—filter, count, or paginate explicitly."
            elif is_exact_repeat or is_near_repeat:
                prev_step, prev_obs = prev_occurrences[-1] if prev_occurrences else (state.env_calls - 1, "")
                outcome_type = "repeated_action"
                reason_msg = (
                    f"Repeated command `{norm_cmd[:80]}` (previously run at step {prev_step} "
                    f"with result `{prev_obs[:80]}`). Repeating it in the same state will not change the outcome."
                )

            if outcome_type and reason_msg:
                state.failure_memory_hits += 1
                rec = FailureRecord(
                    state_sig=state.last_state_sig[:120],
                    action_sig=norm_cmd[:120],
                    outcome_type=outcome_type,
                    summary=reason_msg,
                    step=state.env_calls,
                )
                # Deduplicate identical failure summaries
                if not any(r.action_sig == rec.action_sig and r.outcome_type == rec.outcome_type for r in state.failure_records):
                    state.failure_records.append(rec)
                advisories.append(_format_failure_memory_warning(state.failure_records, latest=rec))

        state.seen_actions.setdefault(norm_cmd, []).append((state.env_calls, obs.strip()[:120]))
        state.recent_commands.append(norm_cmd)
        state.last_state_sig = f"{norm_cmd[:80]} => {obs.strip()[:80]}"

    # Also catch empty tool calls in c2/c4
    if state.condition in ("c2", "c4") and not (assistant_message.get("tool_calls") or []):
        state.failure_memory_hits += 1
        advisories.append(
            "[FAILURE MEMORY] You did not issue a tool call. Always respond with one of the provided tools rather than plain text."
        )

    return "\n".join(advisories) if advisories else None


def _format_failure_memory_warning(records: List[FailureRecord], latest: FailureRecord) -> str:
    lines = [f"[FAILURE MEMORY] {latest.summary}"]
    prior = [r for r in records[:-1]][-3:]
    if prior:
        lines.append("Earlier failed/uninformative steps in this episode (avoid repeating):")
        for r in prior:
            lines.append(f"- Step {r.step} [{r.outcome_type}]: `{r.action_sig[:60]}`")
    return "\n".join(lines)
