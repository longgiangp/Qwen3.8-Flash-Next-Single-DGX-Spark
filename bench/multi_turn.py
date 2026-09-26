#!/usr/bin/env python3
"""Per-turn TTFT and prefix-cache reuse across a growing multi-turn conversation.

Every other correctness/perf tool here sends one prompt (or the same prompt
twice). Real agent sessions keep one conversation open and append a turn at a
time -- exactly the shape MTP's eagle-style block drop affects: with it
active, the prefix cache discards the trailing matched block of a request and
recomputes it every turn (one full Mamba block, 1,664 tokens at MTP 3 with
this profile's dtypes), so a warm turn is not as warm as it should be.
files/patch_block_drop.py (MTP_DISABLE_BLOCK_DROP=1) removes that back-off.

This tool measures, not asserts: it reports per-turn-index TTFT and
cached_tokens across several independent conversations, so a run against a
server with the knob off (baseline) can be compared to one with it on. Save
--output from both and diff the "by_turn" medians.

Each turn appends the model's own previous reply to the conversation (a real
assistant message, not synthetic filler), so cache growth matches an actual
session. Uses /v1/chat/completions with streaming so TTFT is the time to the
first content/reasoning delta, like agent_load.py's load mode.

Only loopback HTTP endpoints are accepted, like the other bench tools.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import secrets
import statistics
import sys
import time
import urllib.request
from datetime import datetime, timezone

import runtime_validation as runtime

TASKS = [
    "Summarize the main idea of the reference material above in one sentence.",
    "Name one technical term used in the reference material and define it briefly.",
    "What tone does the reference material use? Answer in a few words.",
    "Suggest one follow-up question a reader might have, in one sentence.",
    "Restate your previous answer in exactly five words.",
    "List two topics from the reference material, comma-separated.",
    "Is the reference material formal or informal? One word.",
    "Give a one-sentence critique of the reference material.",
]


def is_healthy(base: str) -> bool:
    try:
        request = urllib.request.Request(base + "/health")
        with runtime.opener().open(request, timeout=5) as response:
            return response.status == 200
    except Exception:  # noqa: BLE001
        return False


def stream_chat(base: str, model: str, messages: list[dict], max_tokens: int) -> dict:
    """One streaming chat. Raises on any transport/HTTP failure."""
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0,
               "stream": True, "stream_options": {"include_usage": True}}
    import time
    started = time.perf_counter()
    first = None
    content = ""
    prompt_tokens = None
    cached_tokens = None
    request = urllib.request.Request(
        base + "/v1/chat/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with runtime.opener().open(request, timeout=600) as response:
        for event in runtime.iter_sse(response):
            usage = event.get("usage")
            if usage:
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                details = usage.get("prompt_tokens_details") or {}
                cached_tokens = details.get("cached_tokens", cached_tokens)
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                if first is None and (delta.get("content") or delta.get("reasoning_content")
                                      or delta.get("reasoning")):
                    first = time.perf_counter() - started
                content += delta.get("content") or ""
    return {"ttft": first, "content": content, "prompt_tokens": prompt_tokens, "cached_tokens": cached_tokens}


def run_conversation(args: argparse.Namespace, system_prompt: str, trial: int) -> list[dict]:
    salt = f"Session {secrets.token_hex(8)}.\n"  # keeps trials cache-independent (turn 0 is always cold)
    messages = [{"role": "system", "content": salt + system_prompt}]
    turns = []
    for n in range(args.turns):
        messages.append({"role": "user", "content": f"Trial {trial}-{n}: {TASKS[n % len(TASKS)]}"})
        row = {"turn": n, "crashed": False, "error": None}
        try:
            reply = stream_chat(args.base_url, args.model, messages, args.max_tokens)
        except Exception as exc:  # noqa: BLE001 - a transport failure is itself a result
            row.update(crashed=True, error=f"{type(exc).__name__}: {exc}")
            turns.append(row)
            if not args.continue_on_crash:
                break
            messages.append({"role": "assistant", "content": ""})
            continue
        row.update(ttft=reply["ttft"], prompt_tokens=reply["prompt_tokens"], cached_tokens=reply["cached_tokens"])
        turns.append(row)
        messages.append({"role": "assistant", "content": reply["content"]})
        if not is_healthy(args.base_url):
            turns.append({"turn": n + 1, "crashed": True, "error": "server unhealthy after this turn"})
            break
    return turns


def summarize(trials: list[list[dict]]) -> dict:
    by_turn: dict[int, dict] = {}
    for conversation in trials:
        for row in conversation:
            if row["crashed"]:
                continue
            bucket = by_turn.setdefault(row["turn"], {"ttft": [], "cached_tokens": [], "prompt_tokens": []})
            if row.get("ttft") is not None:
                bucket["ttft"].append(row["ttft"])
            if row.get("cached_tokens") is not None:
                bucket["cached_tokens"].append(row["cached_tokens"])
            if row.get("prompt_tokens") is not None:
                bucket["prompt_tokens"].append(row["prompt_tokens"])
    result = {}
    for turn, bucket in sorted(by_turn.items()):
        result[str(turn)] = {
            "n": len(bucket["ttft"]),
            "ttft_median_s": round(statistics.median(bucket["ttft"]), 3) if bucket["ttft"] else None,
            "ttft_max_s": round(max(bucket["ttft"]), 3) if bucket["ttft"] else None,
            "cached_tokens_median": statistics.median(bucket["cached_tokens"]) if bucket["cached_tokens"] else None,
            "prompt_tokens_median": statistics.median(bucket["prompt_tokens"]) if bucket["prompt_tokens"] else None,
        }
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("--base-url", default="http://127.0.0.1:8888")
    result.add_argument("--model", default=runtime.MODEL)
    result.add_argument("--corpus", type=pathlib.Path, default=runtime.ROOT / "README.md")
    result.add_argument("--system-tokens", type=int, default=6_000,
                        help="reference material carried as the system prompt")
    result.add_argument("--turns", type=int, default=6, help="messages per conversation")
    result.add_argument("--trials", type=int, default=5, help="independent conversations")
    result.add_argument("--max-tokens", type=int, default=256)
    result.add_argument("--continue-on-crash", action="store_true",
                        help="keep the conversation going after a crashed turn (off by default)")
    result.add_argument("--output", type=pathlib.Path)
    return result


def main() -> None:
    args = parser().parse_args()
    try:
        args.base_url = runtime.validate_base_url(args.base_url)
    except ValueError as exc:
        parser().error(str(exc))
    if not args.corpus.is_file():
        parser().error(f"Corpus file does not exist: {args.corpus}")
    if not 2 <= args.turns <= 20:
        parser().error("--turns must be between 2 and 20")
    if not 1 <= args.trials <= 20:
        parser().error("--trials must be between 1 and 20")
    runtime.check_backend(args.base_url, args.model)

    sample = runtime.corpus_text(args.corpus, 40_000)[:40_000]
    per_token = len(sample) / runtime.tokenize_count(args.base_url, args.model, sample)
    corpus = runtime.corpus_text(args.corpus, int(args.system_tokens * per_token * 1.1) + 1000)
    system_prompt = "You are a helpful assistant. Reference material follows.\n" + corpus[
        : int(args.system_tokens * per_token)
    ]

    trials = []
    for t in range(args.trials):
        conversation = run_conversation(args, system_prompt, t)
        trials.append(conversation)
        for row in conversation:
            status = "CRASH" if row["crashed"] else "ok"
            extra = f" ({row['error']})" if row.get("error") else f" cached={row.get('cached_tokens')}"
            print(f"trial {t + 1}/{args.trials} turn {row['turn']}: {status}"
                  + (f" ttft={row['ttft']:.2f}s" if row.get("ttft") is not None else "") + extra,
                  file=sys.stderr, flush=True)

    report = {
        "schema": 1,
        "model": args.model,
        "system_tokens_target": args.system_tokens,
        "turns": args.turns,
        "completed": datetime.now(timezone.utc).isoformat(),
        "by_turn": summarize(trials),
        "trials": trials,
        "generated_text_retained": False,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
