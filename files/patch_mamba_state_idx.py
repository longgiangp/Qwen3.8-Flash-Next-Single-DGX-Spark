#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fix Mamba prefix-cache block-size confusion (align mode) on the pinned image.

vLLM's EngineCore overwrites cache_config.block_size with the *smallest* KV
group block size. On Qwen3.8-Flash-Next that is the QSA raw-key ring (8/16
tokens), while the Mamba recurrent-state block is 1600 (float32 SSM) or 1664
(bfloat16 SSM). Two consumers still treat cache_config.block_size as the Mamba
block size:

  1. v1/worker/gpu/model_states/mamba_hybrid.py, MambaHybridModelState
     .add_request: seeds the state-slot index of a prefix-cache hit with
     (num_computed_tokens - 1) // block_size. With the ring size that column is
     far outside the row, resolves to the null block, and an all-zero state is
     "restored" -- a crash at best, silently wrong output at worst.
  2. v1/core/sched/scheduler.py, _mamba_block_aligned_split: aligns prefill
     chunks to the ring size instead of the Mamba block, so states are almost
     never captured at a real boundary and cold long prefills cache nothing.

The worker takes the Mamba block size from the Mamba spec (or
cache_config.mamba_block_size); the scheduler uses its own self.block_size,
the LCM of all group block sizes, which is the Mamba block size on this model.

This fixes the block-size half only. The state-copy race (vllm#50729) lives in
v1/worker/mamba_utils.py and is a separate change; see the check in the README.

Reference implementations that reached the same fix independently:
techfury90/qwen3.8-Flash-DGX and blazux/qwen3.8-Flash-DGX, src/patch_mamba_block_size.py.

Outputs: files/mamba_hybrid_patched.py, files/scheduler_patched.py
"""
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def patch(name: str, edits: list[tuple[str, str]]) -> None:
    orig = os.path.join(HERE, f"{name}.orig")
    dest = os.path.join(HERE, name)
    if not os.path.exists(orig):
        sys.exit(f"{name}: missing {orig} (start.sh extracts it from the image)")
    src = open(orig).read()
    for i, (old, new) in enumerate(edits):
        count = src.count(old)
        if count != 1:
            sys.exit(
                f"{name}: anchor {i} not unique/missing (count={count}):\n{old[:180]}"
            )
        src = src.replace(old, new)
    try:
        ast.parse(src)
    except SyntaxError as exc:
        sys.exit(f"{name}: patched source does not parse: {exc}")
    open(dest, "w").write(src)
    print(f"patched {name}")


# Module-level helper injected into mamba_hybrid.py. It never returns the
# scheduler block size when a real Mamba block size is available, and warns
# once when it has to fall back (that is the buggy value in align mode).
MAMBA_BS_HELPER = '''

def _resolve_mamba_block_size(model_state) -> int:
    """Mamba state block size, not cache_config.block_size (min group size)."""
    spec = getattr(model_state, "_mamba_spec", None)
    block_size = getattr(spec, "block_size", None)
    if not block_size:
        block_size = getattr(model_state.cache_config, "mamba_block_size", None)
    if not block_size:
        from vllm.logger import init_logger

        init_logger(__name__).warning_once(
            "mamba state idx: no Mamba block size found; falling back to "
            "cache_config.block_size. Prefix-cache hits may restore a wrong state."
        )
        block_size = model_state.cache_config.block_size
    return block_size
'''


def main() -> None:
    patch("mamba_hybrid_patched.py", [
        # The seed of the state-slot index on a prefix-cache hit.
        (
            "                (new_req_data.num_computed_tokens - 1) // self.cache_config.block_size\n",
            "                (new_req_data.num_computed_tokens - 1)\n"
            "                // _resolve_mamba_block_size(self)\n",
        ),
        # Helper goes right before the class it serves.
        (
            "\nclass MambaHybridModelState(",
            MAMBA_BS_HELPER + "\n\nclass MambaHybridModelState(",
        ),
    ])
    patch("scheduler_patched.py", [
        # Chunk boundaries must land on Mamba block boundaries.
        (
            "        block_size = self.cache_config.block_size\n"
            "        # The last block-aligned position whose state can be cached.",
            "        # Scheduler block size (LCM of the groups) is the Mamba block size;\n"
            "        # cache_config.block_size is the smallest group's (QSA ring).\n"
            "        block_size = self.block_size\n"
            "        # The last block-aligned position whose state can be cached.",
        ),
    ])
    print("ok")


if __name__ == "__main__":
    main()
