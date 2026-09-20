#!/usr/bin/env python3
"""CPU-only guard checks for the vendored mamba_utils fix (vllm#50729 + bounds guard)."""
import ast
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXED = ROOT / "files" / "mamba_utils_guarded.py"
START = (ROOT / "start.sh").read_text()


class MambaUtilsFixTests(unittest.TestCase):
    def test_vendored_file_parses_and_has_both_fixes(self):
        source = FIXED.read_text()
        ast.parse(source)
        self.assertIn("is_left_overlap", source)             # #50729 overlap handling
        self.assertIn("tl.debug_barrier()", source)
        self.assertIn("mamba state-copy guard", source)      # bounds guard telemetry

    def test_start_sh_gates_on_the_image_original_hash(self):
        pinned = re.search(r'^MAMBA_UTILS_ORIG_SHA256="([0-9a-f]{64})"$', START, re.M)
        self.assertIsNotNone(pinned)
        self.assertIn('"$_got" != "$MAMBA_UTILS_ORIG_SHA256"', START)
        # a mismatch must abort, never fall through to mounting an unverified file
        gate = START[START.index('"$_got" != "$MAMBA_UTILS_ORIG_SHA256"'):]
        self.assertLess(gate.index("err "), gate.index("MAMBA_UTILS_MOUNT_SRC="))

    def test_mount_only_when_enabled(self):
        self.assertEqual(START.count('MAMBA_UTILS_MOUNT_SRC="$PATCHED_MAMBA_UTILS"'), 1)
        self.assertIn("${MAMBA_UTILS_MOUNT_SRC:+-v $MAMBA_UTILS_MOUNT_SRC:$MAMBA_UTILS_PKG:ro}", START)

    def test_attribution_present(self):
        notice = (ROOT / "files" / "THIRD_PARTY.md").read_text()
        for needle in ("Apache-2.0", "vllm#50729", "blazux", "MAMBA_UTILS_ORIG_SHA256"):
            self.assertIn(needle, notice)


if __name__ == "__main__":
    unittest.main()
