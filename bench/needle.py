#!/usr/bin/env python3
"""Needle-in-a-haystack at fixed depths, cold and warm, repeated.

Each trial plants a random passphrase at a depth (percent of the prompt), asks
for it, then repeats the *identical* prompt. The first call is cold (a random
salt at the top defeats any earlier cache); the repeat goes through the
prefix-cache / Mamba-state-restore path that patch_mamba_state_idx.py fixes.
A wrong or empty answer on the warm call but not the cold one points at a bad
restored state. The deepest positions are flaky even without caching (see the
README), so read the per-depth rates over many trials, never a single run.

Only loopback HTTP endpoints are accepted, like the other bench tools.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import secrets
import sys
from datetime import datetime, timezone

import runtime_validation as runtime

QUESTION = (
    "\n\nQuestion: what is the secret passphrase stated in the text above? "
    "Answer with the passphrase only. /no_think"
)


def parse_depths(value: str) -> list[int]:
    try:
        depths = [int(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("depths must be comma-separated integers") from exc
    if not depths or any(d < 0 or d > 100 for d in depths):
        raise argparse.ArgumentTypeError("depths are percentages from 0 through 100")
    return depths


def build_prompt(filler: str, depth_pct: int, salt: str, passphrase: str) -> str:
    """Salt on top, needle inserted at depth_pct of the filler, question last."""
    cut = len(filler) * depth_pct // 100
    # Cut on a line break so the needle is its own sentence.
    cut = filler.rfind("\n", 0, cut) + 1 if depth_pct else 0
    needle = f"\nThe secret passphrase is {passphrase}. Remember it.\n"
    return f"Validation salt {salt}.\n" + filler[:cut] + needle + filler[cut:] + QUESTION


def build_filler(base: str, model: str, corpus: pathlib.Path, target_tokens: int) -> tuple[str, int]:
    """Real text sized to about target_tokens, calibrated with the serving tokenizer."""
    sample = runtime.corpus_text(corpus, 40_000)[:40_000]
    per_token = len(sample) / runtime.tokenize_count(base, model, sample)
    text = runtime.corpus_text(corpus, int(target_tokens * per_token * 1.05))
    return text[: int(target_tokens * per_token)], int(per_token * 100)


def ask(base: str, model: str, prompt: str, max_tokens: int) -> dict:
    response = runtime.request_json(
        base,
        "/v1/completions",
        {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0},
        timeout=3600,
    )
    text = response["choices"][0].get("text", "")
    usage = response.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    return {
        "text": text,
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": details.get("cached_tokens"),
    }


def trial(args: argparse.Namespace, filler: str, depth: int) -> dict:
    passphrase = "-".join(secrets.token_hex(2) for _ in range(3))
    prompt = build_prompt(filler, depth, secrets.token_hex(16), passphrase)
    cold = ask(args.base_url, args.model, prompt, args.max_tokens)
    warm = ask(args.base_url, args.model, prompt, args.max_tokens)
    return {
        "depth_pct": depth,
        "cold_found": passphrase in cold["text"],
        "warm_found": passphrase in warm["text"],
        "cold_cached_tokens": cold["cached_tokens"],
        "warm_cached_tokens": warm["cached_tokens"],
        "prompt_tokens": cold["prompt_tokens"],
        # The passphrase itself is not reported; it is random and worthless.
    }


def summarize(trials: list[dict]) -> dict:
    by_depth: dict[int, dict] = {}
    for item in trials:
        row = by_depth.setdefault(
            item["depth_pct"], {"trials": 0, "cold_found": 0, "warm_found": 0, "warm_cache_hits": 0}
        )
        row["trials"] += 1
        row["cold_found"] += item["cold_found"]
        row["warm_found"] += item["warm_found"]
        row["warm_cache_hits"] += bool(item["warm_cached_tokens"])
    return {str(depth): row for depth, row in sorted(by_depth.items())}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--base-url", default="http://127.0.0.1:8888")
    result.add_argument("--model", default=runtime.MODEL)
    result.add_argument("--corpus", type=pathlib.Path, default=runtime.ROOT / "README.md")
    result.add_argument("--tokens", type=int, default=262_000,
                        help="approximate prompt size (262000 native; ~510000 needs YARN=1)")
    result.add_argument("--depths", type=parse_depths, default=parse_depths("5,50,95"))
    result.add_argument("--trials", type=int, default=3, help="trials per depth")
    result.add_argument("--max-tokens", type=int, default=32)
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
    if not 1 <= args.trials <= 50:
        parser().error("--trials must be between 1 and 50")
    runtime.check_backend(args.base_url, args.model)

    filler, _ = build_filler(args.base_url, args.model, args.corpus, args.tokens)
    trials = []
    for n in range(args.trials):
        for depth in args.depths:
            item = trial(args, filler, depth)
            trials.append(item)
            print(
                f"trial {n + 1}/{args.trials} depth {depth:>3}%: cold={'ok' if item['cold_found'] else 'MISS'} "
                f"warm={'ok' if item['warm_found'] else 'MISS'} cached={item['warm_cached_tokens']}",
                file=sys.stderr, flush=True,
            )
    report = {
        "schema": 1,
        "started_tokens_target": args.tokens,
        "model": args.model,
        "completed": datetime.now(timezone.utc).isoformat(),
        "by_depth": summarize(trials),
        "trials": trials,
        "generated_text_retained": False,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
