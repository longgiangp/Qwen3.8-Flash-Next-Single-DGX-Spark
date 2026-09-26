#!/usr/bin/env python3
"""CPU-only: start.sh actually wires the determinism and block-drop backports.

Guards the plumbing (env reading, docker run mounts/env, JSON merge, chaining
order) against accidental edits; files/patch_determinism.py and
files/patch_block_drop.py have their own behavioural tests.
"""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START = (ROOT / "start.sh").read_text()
ENV_SAMPLE = (ROOT / ".env.sample").read_text()


class DeterminismWiringTests(unittest.TestCase):
    def test_env_vars_are_read_and_snapshot_for_env_precedence(self):
        self.assertIn('VLLM_QSA_DET_TOPK="${VLLM_QSA_DET_TOPK:-}"', START)
        self.assertIn('VLLM_MOE_DET_FINALIZE="${VLLM_MOE_DET_FINALIZE:-}"', START)
        snapshot = re.search(r"_ENV_SNAPSHOT_VARS=\((.*?)\)", START, re.S).group(1)
        self.assertIn("VLLM_QSA_DET_TOPK", snapshot)
        self.assertIn("VLLM_MOE_DET_FINALIZE", snapshot)

    def test_patch_runs_after_qsa_fp8_patch(self):
        # patch_determinism.py chains onto qsa_ops_patched.py; it must run after
        # patch_qsa_fp8_kv.py produces that file. Exact invocations, not any mention
        # (the knob's doc comment references patch_determinism.py earlier by name).
        qsa = START.index('python3 "$SCRIPT_DIR/files/patch_qsa_fp8_kv.py"')
        det = START.index('python3 "$SCRIPT_DIR/files/patch_determinism.py"')
        self.assertLess(qsa, det)

    def test_moe_file_extracted_patched_and_mounted_once(self):
        self.assertIn('extract "$MOE_CUTLASS_PKG" "$DET_DIR/orig/flashinfer_cutlass_moe.py"', START)
        self.assertEqual(START.count('-v $DET_DIR/flashinfer_cutlass_moe.py:$MOE_CUTLASS_PKG:ro'), 1)

    def test_fused_finalize_env_pair_is_gated_on_the_knob(self):
        self.assertIn("VLLM_FLASHINFER_MOE_FUSED_FINALIZE=0", START)
        self.assertIn('[[ "$VLLM_MOE_DET_FINALIZE" == 1 ]]', START)


class BlockDropWiringTests(unittest.TestCase):
    def test_knob_defaults_off_and_is_validated(self):
        self.assertIn('MTP_DISABLE_BLOCK_DROP="${MTP_DISABLE_BLOCK_DROP:-0}"', START)
        self.assertIn('"$MTP_DISABLE_BLOCK_DROP" == "0" || "$MTP_DISABLE_BLOCK_DROP" == "1"', START)
        match = re.search(r"^MTP_DISABLE_BLOCK_DROP=(\S+)", ENV_SAMPLE, re.M)
        self.assertEqual(match.group(1), "0")  # not yet measured on this host; opt-in

    def test_gated_on_knob_and_mtp(self):
        self.assertIn(
            '[[ "$MTP_DISABLE_BLOCK_DROP" == "1" && "$MTP_NUM_SPECULATIVE_TOKENS" -gt 0 ]]', START
        )

    def test_json_key_merges_into_speculative_config(self):
        self.assertIn('_SPEC_ARGMAX+=\',"disable_eagle_block_drop":true\'', START)

    def test_runs_before_the_mamba_patch_and_scheduler_gets_one_mount(self):
        # Exact invocations, not any mention: each section's own doc comment names
        # the other script (explaining the chaining) ahead of its real invocation.
        block_drop = START.index('python3 "$SCRIPT_DIR/files/patch_block_drop.py" "$BLOCK_DROP_DIR/orig"')
        mamba = START.index('python3 "$SCRIPT_DIR/files/patch_mamba_state_idx.py"')
        self.assertLess(block_drop, mamba)
        # scheduler.py is chained into $PATCHED_SCHED; the block-drop mount list
        # must skip it, or the later mount would silently drop our mamba fix.
        self.assertIn('"$f" != "v1/core/sched/scheduler.py"', START)

    def test_block_drop_mounts_variable_is_referenced_in_docker_run(self):
        self.assertIn("BLOCK_DROP_MOUNTS+=", START)
        self.assertIn("$BLOCK_DROP_MOUNTS", START)


if __name__ == "__main__":
    unittest.main()
