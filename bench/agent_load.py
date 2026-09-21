#!/usr/bin/env python3
"""Agent-shaped checks for the single-endpoint cutover (loopback only).

  load      N concurrent streaming chats that share one long system prompt (like
            agents with the same instructions) plus a short unique task each.
            Reports TTFT/total per level and the peak of vllm:num_requests_waiting.
  features  tool call (plain + streamed), reasoning on/off via chat_template_kwargs,
            and an image input.

    python3 bench/agent_load.py load --levels 1 2 4 6 8 10 --prefix-tokens 20000
    python3 bench/agent_load.py features

Neither mode reads or stores generated text beyond what it needs to check.
"""

from __future__ import annotations

import argparse
import base64
import json
import pathlib
import re
import secrets
import statistics
import struct
import sys
import threading
import time
import zlib
from datetime import datetime, timezone

import runtime_validation as runtime


def waiting_now(base: str) -> float | None:
    try:
        with runtime.opener().open(base + "/metrics", timeout=5) as response:
            for line in response.read().decode().splitlines():
                if line.startswith("vllm:num_requests_waiting"):
                    return float(line.split()[-1])
    except Exception:
        return None
    return None


def stream_chat(base: str, model: str, messages: list[dict], max_tokens: int, extra: dict | None = None) -> dict:
    """One streaming chat; TTFT is the first reasoning or content delta."""
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0,
               "stream": True, "stream_options": {"include_usage": True}, **(extra or {})}
    started = time.perf_counter()
    first = None
    completion_tokens = 0
    content = ""
    tool_calls: dict[int, dict] = {}
    request = runtime.urllib.request.Request(
        base + "/v1/chat/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with runtime.opener().open(request, timeout=3600) as response:
        for event in runtime.iter_sse(response):
            usage = event.get("usage")
            if usage:
                completion_tokens = usage.get("completion_tokens", completion_tokens)
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                if first is None and (delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning")
                                      or delta.get("tool_calls")):
                    first = time.perf_counter() - started
                content += delta.get("content") or ""
                for call in delta.get("tool_calls") or []:
                    slot = tool_calls.setdefault(call.get("index", 0), {"name": "", "arguments": ""})
                    fn = call.get("function") or {}
                    slot["name"] += fn.get("name") or ""
                    slot["arguments"] += fn.get("arguments") or ""
    return {"ttft": first, "total": time.perf_counter() - started, "tokens": completion_tokens,
            "content": content, "tool_calls": list(tool_calls.values())}


def run_load(args: argparse.Namespace) -> dict:
    corpus = runtime.corpus_text(args.corpus, max(50_000, args.prefix_tokens * 8))
    per_token = len(corpus[:40_000]) / runtime.tokenize_count(args.base_url, args.model, corpus[:40_000])
    system = "You are a code assistant. Reference material follows.\n" + corpus[: int(args.prefix_tokens * per_token)]
    system = f"Session {secrets.token_hex(8)}.\n" + system  # cold for the first request, shared after

    def task(n: int) -> list[dict]:
        return [{"role": "system", "content": system},
                {"role": "user", "content": f"Task {n}-{secrets.token_hex(3)}: name one topic from the reference material in a sentence."}]

    prime = stream_chat(args.base_url, args.model, task(0), args.max_tokens)
    report = {"prefix_tokens": args.prefix_tokens, "prime_ttft_cold_s": round(prime["ttft"] or 0, 2), "levels": []}
    for level in args.levels:
        peak = [0.0]
        stop = threading.Event()

        def sampler():
            while not stop.is_set():
                value = waiting_now(args.base_url)
                if value is not None:
                    peak[0] = max(peak[0], value)
                stop.wait(0.5)

        thread = threading.Thread(target=sampler, daemon=True)
        thread.start()
        results: list[dict | None] = [None] * level
        errors: list[str] = []

        def one(i: int):
            try:
                results[i] = stream_chat(args.base_url, args.model, task(i + 1), args.max_tokens)
            except Exception as exc:  # noqa: BLE001 - reported, never hidden
                errors.append(f"{type(exc).__name__}: {exc}")

        workers = [threading.Thread(target=one, args=(i,)) for i in range(level)]
        began = time.perf_counter()
        for w in workers:
            w.start()
        for w in workers:
            w.join()
        wall = time.perf_counter() - began
        stop.set()
        thread.join()
        ok = [r for r in results if r]
        ttfts = sorted(r["ttft"] for r in ok if r["ttft"] is not None)
        row = {
            "concurrency": level, "ok": len(ok), "errors": errors[:3], "wall_s": round(wall, 2),
            "ttft_median_s": round(statistics.median(ttfts), 2) if ttfts else None,
            "ttft_max_s": round(max(ttfts), 2) if ttfts else None,
            "tokens_per_s_aggregate": round(sum(r["tokens"] for r in ok) / wall, 1) if ok else None,
            "peak_waiting": peak[0],
        }
        report["levels"].append(row)
        print(json.dumps(row), file=sys.stderr, flush=True)
    return report


def solid_png(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


TOOL = {"type": "function", "function": {
    "name": "write_file", "description": "Write a file.",
    "parameters": {"type": "object", "properties": {
        "filePath": {"type": "string"}, "content": {"type": "string"}}, "required": ["filePath", "content"]}}}


def run_features(args: argparse.Namespace) -> dict:
    base, model = args.base_url, args.model
    checks: dict[str, dict] = {}

    def chat(messages, **extra):
        return runtime.request_json(base, "/v1/chat/completions",
                                    {"model": model, "messages": messages, "temperature": 0, **extra}, timeout=600)

    ask = [{"role": "user", "content": "Create the file /tmp/hello.txt containing the text hi. Use the tool."}]
    message = chat(ask, tools=[TOOL], tool_choice="auto", max_tokens=1024)["choices"][0]["message"]
    calls = message.get("tool_calls") or []
    parsed = None
    if calls:
        try:
            parsed = json.loads(calls[0]["function"]["arguments"])
        except ValueError:
            parsed = None
    checks["tool_call"] = {"passed": bool(parsed and parsed.get("filePath") == "/tmp/hello.txt"), "calls": len(calls)}

    streamed = stream_chat(base, model, ask, 1024, {"tools": [TOOL], "tool_choice": "auto"})
    try:
        streamed_args = json.loads(streamed["tool_calls"][0]["arguments"]) if streamed["tool_calls"] else None
    except ValueError:
        streamed_args = None
    checks["tool_call_streamed"] = {"passed": bool(streamed_args and streamed_args.get("filePath") == "/tmp/hello.txt")}

    question = [{"role": "user", "content": "What is 17 times 23? Answer with the number only."}]
    on = chat(question, max_tokens=2048)["choices"][0]["message"]
    off = chat(question, max_tokens=2048, chat_template_kwargs={"enable_thinking": False})["choices"][0]["message"]
    reasoning = lambda m: (m.get("reasoning_content") or m.get("reasoning") or "")  # noqa: E731
    checks["thinking_default_on"] = {"passed": bool(reasoning(on)) and "391" in (on.get("content") or "")}
    checks["thinking_off_switch"] = {"passed": not reasoning(off) and "391" in (off.get("content") or "")}

    image = "data:image/png;base64," + base64.b64encode(solid_png(128, 128, (220, 20, 20))).decode()
    try:
        vision = chat([{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image}},
            {"type": "text", "text": "What colour is this image? One word."}]}],
            max_tokens=1024, chat_template_kwargs={"enable_thinking": False})["choices"][0]["message"]
        checks["image_input"] = {"passed": bool(re.search(r"red", vision.get("content") or "", re.I))}
    except Exception as exc:  # noqa: BLE001
        checks["image_input"] = {"passed": False, "error": f"{type(exc).__name__}: {str(exc)[:120]}"}
    return checks


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("mode", choices=("load", "features"))
    result.add_argument("--base-url", default="http://127.0.0.1:8888")
    result.add_argument("--model", default=runtime.MODEL)
    result.add_argument("--corpus", type=pathlib.Path, default=runtime.ROOT / "README.md")
    result.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4, 6, 8, 10])
    result.add_argument("--prefix-tokens", type=int, default=20_000)
    result.add_argument("--max-tokens", type=int, default=256)
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
    runtime.check_backend(args.base_url, args.model)
    body = run_load(args) if args.mode == "load" else run_features(args)
    report = {"schema": 1, "mode": args.mode, "model": args.model,
              "completed": datetime.now(timezone.utc).isoformat(), "result": body}
    if args.mode == "features":
        report["passed"] = all(c["passed"] for c in body.values())
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
