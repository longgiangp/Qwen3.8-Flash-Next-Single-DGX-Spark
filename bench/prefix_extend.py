#!/usr/bin/env python3
"""Prefix-cache correctness when a request EXTENDS a cached prefix with a
different length, not just repeats it identically.

Every other correctness check in this repo (cache_correctness.py, needle.py)
warms the cache by repeating the *same* prompt. Real conversations extend a
prompt instead: turn 2 is turn 1's content plus more. vLLM issue #54173
reports a GB10-specific CUBLAS_STATUS_INTERNAL_ERROR / illegal memory access
in the Mamba/GDN path specifically on that path -- a cache-hit resume where
the new request extends a cached prefix to a *different* length. Identical
repeats (what needle.py sends) do not exercise it.

Each case sends a base prompt (cold), then a second request whose message is
the base prompt's exact text plus more filler and a question -- a byte-exact
prefix extension. Extension sizes are chosen around this deployment's Mamba
block size (1664 tokens, from the server's "Setting attention block size"
log line; override with --mamba-block-size if the profile changes) so both
sub-block and multi-block, boundary-crossing extensions are covered.

Detects three kinds of failure:
  - crash: the extension request (or the base request) raises -- timeout,
    connection reset, HTTP error. Checked immediately; by default this stops
    the whole run, because a crashed server answers every later case wrong
    for reasons that have nothing to do with prefix extension.
  - no cache reuse: usage.prompt_tokens_details.cached_tokens on the
    extension is not positive, so the shared prefix was recomputed --
    a functional regression, not memory corruption, but worth flagging.
  - wrong recall: a passphrase planted at the start of the shared (base)
    prefix is not found after the extension, while the base call itself
    did find it.

Requests go through /v1/chat/completions, like needle.py.
Only loopback HTTP endpoints are accepted, like the other bench tools.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import secrets
import sys
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone

import runtime_validation as runtime

QUESTION = (
    "\n\nQuestion: what is the secret code stated near the start of the text "
    "above? Answer with the code only."
)


@dataclass
class Case:
    label: str
    base_tokens: int
    extend_tokens: int


def default_cases(block: int) -> list[Case]:
    """Sub-block, one-block, boundary-crossing and multi-block extensions."""
    return [
        Case("sub-block", 4_000, max(block // 8, 64)),
        Case("one-block", 4_000, block),
        Case("boundary-cross", 4_000, block + 40),
        Case("multi-block", 4_000, block * 3),
        Case("large-base-sub-block", 40_000, max(block // 8, 64)),
        Case("large-extension", 4_000, 30_000),
    ]


def parse_cases(value: str, block: int) -> list[Case]:
    cases = []
    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = item.split(",")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(f"case must be base,extend: {item!r}")
        try:
            base, extend = int(parts[0]), int(parts[1])
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"case must be base,extend integers: {item!r}") from exc
        cases.append(Case(f"{base}+{extend}", base, extend))
    if not cases:
        raise argparse.ArgumentTypeError("no cases given")
    return cases


def build_filler(base: str, model: str, corpus: pathlib.Path, target_tokens: int) -> tuple[str, float]:
    """Real text sized to comfortably cover target_tokens, calibrated with the serving tokenizer."""
    sample = runtime.corpus_text(corpus, 40_000)[:40_000]
    per_token = len(sample) / runtime.tokenize_count(base, model, sample)
    text = runtime.corpus_text(corpus, int(target_tokens * per_token * 1.1) + 1000)
    return text, per_token


def slice_chars(filler: str, tokens: int, per_token: float) -> str:
    return filler[: max(int(tokens * per_token), 1)]


def is_healthy(base: str) -> bool:
    try:
        request = urllib.request.Request(base + "/health")
        with runtime.opener().open(request, timeout=5) as response:
            return response.status == 200
    except Exception:  # noqa: BLE001 - any failure to reach /health means "not healthy"
        return False


def ask(base: str, model: str, content: str, max_tokens: int) -> dict:
    """One chat call. Raises on any transport/HTTP failure -- callers treat that as a crash."""
    response = runtime.request_json(
        base,
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": content}],
         "max_tokens": max_tokens, "temperature": 0},
        timeout=600,
    )
    text = response["choices"][0]["message"].get("content") or ""
    usage = response.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    return {
        "text": text,
        "finish_reason": response["choices"][0].get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": details.get("cached_tokens"),
    }


def trial(args: argparse.Namespace, filler: str, per_token: float, case: Case) -> dict:
    passphrase = "-".join(secrets.token_hex(2) for _ in range(3))
    salt = secrets.token_hex(16)
    base_filler = slice_chars(filler, case.base_tokens, per_token)
    base_text = f"Validation salt {salt}.\nThe secret code is {passphrase}. Remember it.\n\n" + base_filler
    extend_filler = slice_chars(filler[len(base_filler):], case.extend_tokens, per_token)

    result: dict = {
        "case": case.label, "base_tokens": case.base_tokens, "extend_tokens": case.extend_tokens,
        "crashed": False, "error": None,
    }
    try:
        base_reply = ask(args.base_url, args.model, base_text, args.max_tokens)
    except Exception as exc:  # noqa: BLE001 - any transport failure is the signal we're looking for
        result.update(crashed=True, error=f"base call: {type(exc).__name__}: {exc}")
        return result
    result["base_found"] = passphrase in base_reply["text"]
    result["base_cached_tokens"] = base_reply["cached_tokens"]
    result["base_prompt_tokens"] = base_reply["prompt_tokens"]

    if not is_healthy(args.base_url):
        result.update(crashed=True, error="server unhealthy after the base call")
        return result

    try:
        ext_reply = ask(args.base_url, args.model, base_text + extend_filler + QUESTION, args.max_tokens)
    except Exception as exc:  # noqa: BLE001
        result.update(crashed=True, error=f"extension call: {type(exc).__name__}: {exc}")
        return result
    result["extend_found"] = passphrase in ext_reply["text"]
    result["extend_cached_tokens"] = ext_reply["cached_tokens"]
    result["extend_prompt_tokens"] = ext_reply["prompt_tokens"]
    result["reused_cache"] = bool(ext_reply["cached_tokens"])
    result["wrong_recall"] = bool(result["base_found"]) and not result["extend_found"]
    result["passed"] = (not result["wrong_recall"]) and result["reused_cache"]
    return result


def verdict(item: dict) -> str:
    if item["crashed"]:
        return "CRASH"
    if item["wrong_recall"]:
        return "RECALL-FAIL"
    if not item["reused_cache"]:
        return "NO-CACHE-HIT"
    return "ok"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("--base-url", default="http://127.0.0.1:8888")
    result.add_argument("--model", default=runtime.MODEL)
    result.add_argument("--corpus", type=pathlib.Path, default=runtime.ROOT / "README.md")
    result.add_argument("--mamba-block-size", type=int, default=1664,
                        help="from the server's 'Setting attention block size' log line")
    result.add_argument("--cases", type=str, default=None,
                        help="override the default cases: 'base,extend;base,extend;...' in tokens")
    result.add_argument("--trials", type=int, default=3, help="trials per case")
    result.add_argument("--max-tokens", type=int, default=2048)
    result.add_argument(
        "--continue-on-crash", action="store_true",
        help="keep running later cases after a crash (off by default: a crashed server answers "
             "every later case wrong for reasons that have nothing to do with prefix extension)",
    )
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
    if not 1 <= args.trials <= 20:
        parser().error("--trials must be between 1 and 20")
    runtime.check_backend(args.base_url, args.model)

    cases = parse_cases(args.cases, args.mamba_block_size) if args.cases else default_cases(args.mamba_block_size)
    max_tokens_needed = max(c.base_tokens + c.extend_tokens for c in cases)
    filler, per_token = build_filler(args.base_url, args.model, args.corpus, max_tokens_needed)

    trials: list[dict] = []
    stopped_early = False
    for n in range(args.trials):
        for case in cases:
            item = trial(args, filler, per_token, case)
            trials.append(item)
            print(
                f"trial {n + 1}/{args.trials} case {case.label:>20}: {verdict(item)}"
                + (f"  ({item['error']})" if item.get("error") else ""),
                file=sys.stderr, flush=True,
            )
            if item["crashed"] and not args.continue_on_crash:
                stopped_early = True
                break
        if stopped_early:
            break

    report = {
        "schema": 1,
        "mamba_block_size": args.mamba_block_size,
        "model": args.model,
        "stopped_early": stopped_early,
        "completed": datetime.now(timezone.utc).isoformat(),
        "passed": bool(trials) and not stopped_early and all(t["passed"] for t in trials if not t["crashed"]),
        "trials": trials,
        "generated_text_retained": False,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    if stopped_early:
        sys.exit(1)


if __name__ == "__main__":
    main()
