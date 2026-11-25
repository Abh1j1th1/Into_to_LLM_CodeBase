# Adapted from https://github.com/THUDM/AgentRL (examples/eval/server_agent.py).
# Iteration 2 additions:
# - Proposal 3 experimental conditions (--condition c0|c1|c2|c3|c4) and --memory-bank
# - KeyError fix for -T (--truncate-multiple-toolcall) and stripping non-standard assistant keys
# - Token/latency/verify-gate metadata logging in JSONL output
import asyncio
import json
import os
import random
import re
import sys
import time
import traceback
from argparse import ArgumentParser
from datetime import datetime
from json import JSONDecodeError
from os.path import dirname, join
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from aiohttp import ClientSession
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

from conditions import (
    EpisodeState,
    VALID_CONDITIONS,
    build_cross_task_preamble,
    build_verify_gate_messages,
    default_round_limit,
    extract_commit_answer,
    infer_task_type_from_name,
    is_commit_message,
    load_memory_bank,
    process_env_interaction,
    sanitize_message_for_api,
    should_fire_verify_gate,
)

EXTRA_BODY: Dict[str, Any] = {}  # set from --extra-body


class RateLimitError(Exception):
    def __init__(self, wait_s: float, message: str = "Rate limited", is_daily: bool = False):
        super().__init__(message)
        self.wait_s = wait_s
        self.is_daily = is_daily


def _parse_retry_after(header_val: Optional[str]) -> float:
    if not header_val:
        return 15.0
    s = str(header_val).strip().lower()
    total = 0.0
    if m := re.search(r"([0-9]+(?:\.[0-9]+)?)h", s):
        total += float(m.group(1)) * 3600.0
    if m := re.search(r"([0-9]+(?:\.[0-9]+)?)m(?!s)", s):
        total += float(m.group(1)) * 60.0
    if m := re.search(r"([0-9]+(?:\.[0-9]+)?)ms", s):
        total += float(m.group(1)) / 1000.0
    elif m := re.search(r"([0-9]+(?:\.[0-9]+)?)s", s):
        total += float(m.group(1))
    elif total == 0.0:
        if m := re.search(r"([0-9]+(?:\.[0-9]+)?)", s):
            total = float(m.group(1))
    return max(2.0, min(total + 1.5, 120.0)) if total > 0 else 15.0


class OpenAIAgent:
    def __init__(
        self,
        model: str,
        base_url: str,
        model2: str = "",
        base_url2: str = "",
        api_key: str = "placeholder",
        verbose: bool = False,
        max_completion_tokens: int = 1024,
    ):
        self.model = model
        self.verbose = verbose
        self.base_url = base_url
        self.api_key = api_key
        self.max_completion_tokens = max_completion_tokens

        self.model2 = model2
        self.base_url2 = base_url2

        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.client_session = ClientSession()

    async def call(self, url, model, messages, tools, temperature) -> Tuple[dict, dict]:
        payload = {
            "model": model,
            "messages": messages,
            "tool_choice": "auto",
            "tools": tools,
            "temperature": temperature,
            "max_completion_tokens": self.max_completion_tokens,
        }
        payload.update(EXTRA_BODY)
        response = await self.client_session.post(
            url + "/chat/completions",
            json=payload,
            headers={
                "Authorization": "Bearer " + self.api_key,
                "Content-Type": "application/json",
                "Accept-Encoding": "gzip, deflate",
            },
        )
        try:
            data = await response.json(content_type=None)
        except JSONDecodeError:
            text = await response.text()
            print(f"Error: {text}", flush=True)
            raise
        if response.status == 429:
            retry_after = response.headers.get("retry-after") or response.headers.get("x-ratelimit-reset-tokens")
            wait_s = _parse_retry_after(retry_after)
            is_daily = "tokens per day" in str(data).lower() or "requests per day" in str(data).lower()
            raise RateLimitError(wait_s, f"HTTP 429: {data}", is_daily=is_daily)
        if response.status != 200:
            print(payload)
            print("Error", data, flush=True)
        response.raise_for_status()
        message = data["choices"][0]["message"]
        if message.get("content") is None:
            message["content"] = ""
        message = {k: v for k, v in message.items() if v is not None}
        usage = data.get("usage") or {}
        return message, usage

    async def chat_completion(
        self, messages: List[dict], tools: Optional[List[dict]] = None, temperature=None
    ) -> Tuple[dict, dict]:
        max_retries = 12
        for tries in range(max_retries):
            try:
                if self.model2 and random.random() < 0.5:
                    return await self.call(self.base_url2, self.model2, messages, tools, temperature)
                return await self.call(self.base_url, self.model, messages, tools, temperature)
            except RateLimitError as e:
                if e.is_daily or tries == max_retries - 1:
                    raise
                if self.verbose:
                    print(f"Rate limited, sleeping {e.wait_s:.1f}s (attempt {tries + 1}/{max_retries})", flush=True)
                await asyncio.sleep(min(e.wait_s, 75.0))
            except Exception:
                if tries == max_retries - 1:
                    raise
                if self.verbose:
                    traceback.print_exc()
                    print(f"API call failed (attempt {tries + 1}/{max_retries})", flush=True)
                await asyncio.sleep(min(2.0 * (tries + 1), 15.0))
        raise NotImplementedError("Unreachable code")


async def worker(
    args,
    session: ClientSession,
    agent: OpenAIAgent,
    task_index: str,
    memory_items: Optional[List[dict]] = None,
) -> Tuple[List[dict], Union[float, int, Exception], str, Dict[str, Any]]:
    t0 = time.monotonic()
    text = ""
    while True:
        try:
            response = await session.post(
                f"{args.controller}/start_sample",
                json={"name": args.task_name[0], "index": task_index},
            )
            text = await response.text()
            response.raise_for_status()
            session_id = response.headers["session_id"]
            break
        except Exception:
            if args.verbose:
                print(f"Failed to start sample for task {task_index}, {text=}", file=sys.stderr)
                traceback.print_exc()
            await asyncio.sleep(1)

    messages: List[dict] = []
    task_type = infer_task_type_from_name(args.task_name[0])
    state = EpisodeState(
        condition=args.condition,
        task_type=task_type,
        round_limit=default_round_limit(task_type),
    )

    try:
        data = await response.json()
        messages.extend(data.get("messages") or [])
        tools = data.get("tools") or []
        if not messages or data.get("finish"):
            state.latency_s = round(time.monotonic() - t0, 3)
            return messages, RuntimeError(f"Task worker failed during start_sample (finish={data.get('finish')}, empty messages)"), session_id, _episode_metadata(state)
        task_type = infer_task_type_from_name(args.task_name[0], tools)
        state.task_type = task_type
        state.round_limit = default_round_limit(task_type)

        original_question = ""
        for m in messages:
            if m.get("role") == "user" and m.get("content"):
                original_question = str(m["content"])

        if state.condition in ("c2", "c4") and memory_items:
            preamble = build_cross_task_preamble(memory_items)
            if preamble:
                messages.append({"role": "user", "content": preamble})
                state.cross_task_injected = True

        if args.verbose:
            print(f"session_id: {session_id}")
            print(f"{messages=} {tools=}", flush=True)

        for _ in range(args.max_turns):
            api_messages = [
                sanitize_message_for_api(turn, strip_reasoning=args.strip_reasoning)
                for turn in messages
            ]
            raw_message, usage = await agent.chat_completion(
                api_messages, tools, temperature=args.temperature
            )
            state.prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
            state.completion_tokens += int(usage.get("completion_tokens", 0) or 0)

            message = sanitize_message_for_api(raw_message, strip_reasoning=args.strip_reasoning)

            if args.truncate_multiple_toolcall and message.get("tool_calls"):
                message["tool_calls"] = message["tool_calls"][:1]
                if not message["tool_calls"]:
                    del message["tool_calls"]

            if args.verbose:
                print(f"{message=}", flush=True)

            # Proposal 3 targeted verify gate (c3/c4): intercept first commit before /interact
            if should_fire_verify_gate(state, message):
                messages.append(message)
                gate_msgs = build_verify_gate_messages(state, message, original_question)
                messages.extend(gate_msgs)
                if args.verbose:
                    print("[verify_gate] held first answer and injected targeted check", flush=True)
                continue

            messages.append(message)
            if is_commit_message(message):
                state.final_answer = extract_commit_answer(message)

            response = await session.post(
                f"{args.controller}/interact",
                headers={"session_id": session_id},
                json={"messages": [message]},
            )
            response.raise_for_status()
            data = await response.json()

            if args.verbose:
                print(*data["messages"], sep="\n", flush=True)

            if data["finish"]:
                # Still count final env turn if any non-commit tool was sent
                process_env_interaction(state, message, data.get("messages") or [])
                state.latency_s = round(time.monotonic() - t0, 3)
                return messages, data["reward"], session_id, _episode_metadata(state)
            else:
                server_msgs = data.get("messages") or []
                messages.extend(server_msgs)
                advisory = process_env_interaction(state, message, server_msgs)
                if advisory:
                    messages.append({"role": "user", "content": advisory})

        state.latency_s = round(time.monotonic() - t0, 3)
        return messages, 0, session_id, _episode_metadata(state)
    except Exception as e:
        try:
            await session.post(
                f"{args.controller}/cancel", headers={"session_id": session_id}
            )
        except Exception:
            traceback.print_exc()
        state.latency_s = round(time.monotonic() - t0, 3)
        return messages, e, session_id, _episode_metadata(state)


def _episode_metadata(state: EpisodeState) -> Dict[str, Any]:
    return {
        "condition": state.condition,
        "task_type": state.task_type,
        "temperature": args.temperature,
        "env_calls": state.env_calls,
        "verify_gate_fired": state.verify_gate_fired,
        "verify_gate_skipped_low_budget": state.verify_gate_skipped_low_budget,
        "first_answer": state.first_answer,
        "final_answer": state.final_answer,
        "failure_memory_hits": state.failure_memory_hits,
        "repeat_warnings_issued": state.repeat_warnings_issued,
        "cross_task_injected": state.cross_task_injected,
        "prompt_tokens": state.prompt_tokens,
        "completion_tokens": state.completion_tokens,
        "latency_s": state.latency_s,
    }


async def get_model_name(url):
    session = ClientSession()
    response = await session.get(f"{url}/models")
    response.raise_for_status()
    data = await response.json(content_type=None)
    await session.close()
    return data["data"][0]["id"]


def parse_model_name(name):
    if match := re.search(r"/([^/]+?)/global_step_(\d+?)/", name):
        model_name = f"{match.group(1)}-step{match.group(2)}"
    elif match := re.search(r"/([^/]+?)/", name):
        model_name = match.group(1)
    else:
        model_name = name
    return model_name.replace("/", "__")


async def run_tasks(args):
    if args.model_api_base and not args.model:
        args.model = await get_model_name(args.model_api_base)
        print(f"Model name: {args.model}", flush=True)
    if args.model2_api_base and not args.model2:
        args.model2 = await get_model_name(args.model2_api_base)
        print(f"Model2 name: {args.model2}", flush=True)

    agent = OpenAIAgent(
        model=args.model,
        base_url=args.model_api_base,
        model2=args.model2,
        base_url2=args.model2_api_base,
        api_key=args.model_api_key,
        verbose=args.verbose,
        max_completion_tokens=args.max_completion_tokens,
    )

    if args.model2:
        model = f"{parse_model_name(args.model)}-{parse_model_name(args.model2)}"
    else:
        model = parse_model_name(args.model)
    if args.run_all:
        args.output_dir = join(
            args.output_dir,
            f'{args.task_name[0]}-{model}-{args.condition}-{args.temperature}-{datetime.now().strftime("%m%d%H%M")}',
        )
    os.makedirs(args.output_dir, exist_ok=True)
    if not args.output_file:
        name = f'{args.task_name[0]}-{model}-{args.condition}-{args.temperature}-{datetime.now().strftime("%m%d%H%M")}.jsonl'
    else:
        name = args.output_file
    output_file = name if os.path.isabs(name) else join(args.output_dir, name)
    os.makedirs(dirname(output_file), exist_ok=True)

    # Get task indices
    if args.index:
        task_indices = [args.index]
        args.runs = 1
    elif getattr(args, "indices_file", None):
        with open(args.indices_file, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            task_indices = list(loaded[args.task_name[0]])
        else:
            task_indices = list(loaded)
    elif args.range:
        start, end = map(int, args.range.split(","))
        task_indices = [i for i in range(start, end)]
    else:
        async with ClientSession() as session:
            response = await session.get(
                f"{args.controller}/get_indices", params={"name": args.task_name[0]}
            )
        response.raise_for_status()
        task_indices = [i for i in await response.json() if i != -1]

    if args.sample_frac is not None and 0.0 < args.sample_frac < 1.0:
        rng = random.Random(args.seed)
        k = max(1, int(round(len(task_indices) * args.sample_frac)))
        task_indices = sorted(rng.sample(list(task_indices), k), key=str)

    if args.limit:
        task_indices = task_indices[: args.limit]

    target_task_type = infer_task_type_from_name(args.task_name[0])
    memory_items = load_memory_bank(args.memory_bank, target_task_type) if args.memory_bank else []

    if args.run_all:
        assert args.model2, "model2 is required for all"
        agent1 = OpenAIAgent(
            model=args.model,
            base_url=args.model_api_base,
            api_key=args.model_api_key,
            verbose=args.verbose,
            max_completion_tokens=args.max_completion_tokens,
        )
        of1 = join(args.output_dir, f"{parse_model_name(args.model)}.jsonl")
        t1 = task(of1, agent1, task_indices, args.model, args.concurrency // 2, memory_items)
        agent2 = OpenAIAgent(
            model=args.model2,
            base_url=args.model2_api_base,
            api_key=args.model_api_key,
            verbose=args.verbose,
            max_completion_tokens=args.max_completion_tokens,
        )
        of2 = join(args.output_dir, f"{parse_model_name(args.model2)}.jsonl")
        t2 = task(of2, agent2, task_indices, args.model2, args.concurrency // 2, memory_items)
        await asyncio.gather(t1, t2)
        output_file = join(args.output_dir, "cross.jsonl")

    try:
        await task(output_file, agent, task_indices, model, args.concurrency, memory_items)
    finally:
        await agent.client_session.close()


async def task(output_file, agent, task_indices, model_name, concurrency, memory_items=None):
    session = ClientSession()
    try:
        response = await session.post(f"{args.controller}/sync_all")
        response.raise_for_status()

        columns = [
            "model",
            "condition",
            "task_index",
            "run_number",
            "result",
            "messages",
            "metadata",
            "timestamp",
            "sid",
        ]
        if os.path.exists(output_file):
            df = pd.read_json(output_file, lines=True)
            for col in columns:
                if col not in df.columns:
                    df[col] = args.condition if col == "condition" else ({} if col == "metadata" else None)
            for obj_col in ("model", "condition", "task_index", "result", "messages", "metadata", "timestamp", "sid"):
                df[obj_col] = df[obj_col].astype(object)
        else:
            df = pd.DataFrame(columns=columns)

        completed_runs = set(
            (str(row["task_index"]), int(row["run_number"]), str(row.get("condition", args.condition)))
            for _, row in df.iterrows()
            if str(row["model"]) == model_name and row["result"] != "error"
        )

        semaphore = asyncio.Semaphore(concurrency)

        async def sem_task(task_index: str, run_number: int):
            async with semaphore:
                if args.verbose:
                    print(f"\n========== task {task_index} run {run_number} ({args.condition}) ==========\n", flush=True)

                try:
                    messages, result, sid, meta = await worker(args, session, agent, task_index, memory_items)
                except Exception as e:
                    messages = []
                    result = e
                    sid = ""
                    meta = {"condition": args.condition}

                if isinstance(result, Exception):
                    print(f"Error in task {task_index} run {run_number}:", file=sys.stderr, flush=True)
                    traceback.print_exception(type(result), result, result.__traceback__)
                    result = "error"
                else:
                    tqdm.write(f"Completed task {task_index} run {run_number} [{args.condition}] -> {result}")

                if result == "error" and len(messages) == 0:
                    return

                mask = (
                    (df["model"].astype(str) == str(model_name))
                    & (df["condition"].astype(str) == str(args.condition))
                    & (df["task_index"].astype(str) == str(task_index))
                    & (df["run_number"] == run_number)
                )
                idx = df[mask].index
                now_ts = pd.Timestamp.now("UTC")
                if len(idx) > 0:
                    df.at[idx[0], "result"] = result
                    df.at[idx[0], "messages"] = messages
                    df.at[idx[0], "metadata"] = meta
                    df.at[idx[0], "timestamp"] = now_ts
                    df.at[idx[0], "sid"] = sid
                else:
                    df.loc[len(df)] = [
                        model_name,
                        args.condition,
                        task_index,
                        run_number,
                        result,
                        messages,
                        meta,
                        now_ts,
                        sid,
                    ]
                df.sort_values(by=["model", "condition", "task_index", "run_number"]).to_json(
                    output_file, index=False, orient="records", lines=True, date_format="iso"
                )

        coroutines = []
        for run_number in range(1, args.runs + 1):
            for task_index in task_indices:
                if (str(task_index), run_number, str(args.condition)) in completed_runs:
                    continue
                coroutines.append(sem_task(task_index, run_number))

        print(
            f"running {model_name} [{args.condition}] => {output_file} "
            f"({len(task_indices)} items, {args.runs} runs, concurrency={concurrency})",
            flush=True,
        )
        try:
            if coroutines:
                await tqdm.gather(*coroutines, position=0)
        finally:
            df = df.sort_values(by=["model", "condition", "task_index", "run_number"])
            df.to_json(output_file, index=False, orient="records", lines=True, date_format="iso")

        valid_df = df[df["result"] != "error"]
        valid = len(valid_df)
        if valid > 0:
            avg = valid_df["result"].mean()
            std = valid_df.groupby(["run_number"])["result"].mean().std()
            bon = valid_df.groupby(["task_index"])["result"].max().mean()
            print(f"Valid: {valid} Avg: {avg:.3f} ± {0.0 if pd.isna(std) else std:.3f} | Best of n: {bon:.3f}")
        else:
            print("Valid: 0")
    finally:
        await session.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="", help="name of the model to be evaluated")
    parser.add_argument("-m2", "--model2", type=str, default="", help="the second model if doing cross sampling")
    parser.add_argument("-u", "--model-api-base", type=str, default=os.getenv("OPENAI_API_BASE"), help="base URL for the model API")
    parser.add_argument("-u2", "--model2-api-base", type=str, default="", help="base URL for the second model API")
    parser.add_argument("--model-api-key", type=str, default=os.getenv("OPENAI_API_KEY", "placeholder"), help="API key for the model API")
    parser.add_argument("-t", "--temperature", type=float, default=0.8)
    parser.add_argument("-a", "--run-all", action="store_true")
    parser.add_argument("-T", "--truncate-multiple-toolcall", action="store_true")
    parser.add_argument("-c", "--controller", type=str, default="http://localhost:5020/api", help="controller URL")
    parser.add_argument("-j", "--concurrency", type=int, default=64, help="number of concurrent tasks to run")
    parser.add_argument("-n", "--runs", type=int, default=4, help="number of runs for each task")
    parser.add_argument("-o", "--output-dir", type=str, default=join(dirname(__file__), "results"), help="directory to store the results")
    parser.add_argument("-f", "--output-file", type=str, default=None, help="output file path/name to write or resume")
    parser.add_argument("--extra-body", type=str, default="{}", help="JSON object merged into every chat request")
    parser.add_argument("--limit", type=int, default=None, help="only run the first N sample indices")
    parser.add_argument("--indices-file", type=str, default=None, help="JSON file of task indices (list or {task_name: list})")
    parser.add_argument("--sample-frac", type=float, default=None, help="random deterministic fraction of indices to sample")
    parser.add_argument("--seed", type=int, default=42, help="random seed for --sample-frac")
    parser.add_argument("--condition", type=str, default="c0", choices=VALID_CONDITIONS, help="Proposal 3 condition")
    parser.add_argument("--memory-bank", type=str, default=None, help="optional JSON file of cross-task/dev failure records for c2/c4")
    parser.add_argument("--max-turns", type=int, default=28, help="safety cap on client LLM turns per episode")
    parser.add_argument("--max-completion-tokens", type=int, default=1024, help="max_completion_tokens per LLM turn")
    parser.add_argument("--no-strip-reasoning", dest="strip_reasoning", action="store_false", help="keep non-standard message keys")
    parser.set_defaults(strip_reasoning=True)
    parser.add_argument("-v", "--verbose", action="store_true", help="print more information")
    parser.add_argument("-r", "--range", help="range of indices a,b", type=str, default=None)
    parser.add_argument("-i", "--index", type=str, required=False, help="test only one index")
    parser.add_argument("-d", "--use-id", action="store_true")
    parser.add_argument("task_name", type=str, nargs=1, help="name of the task to be evaluated")

    args = parser.parse_args()
    EXTRA_BODY.update(json.loads(args.extra_body))

    asyncio.run(run_tasks(args))
