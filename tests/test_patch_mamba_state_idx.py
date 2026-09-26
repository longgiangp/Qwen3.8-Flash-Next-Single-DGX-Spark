#!/usr/bin/env python3
"""CPU-only check of files/patch_mamba_state_idx.py against synthetic sources.

The fixtures reproduce only the anchor lines; they are NOT the image's files.
Phase-2 of the rollout (grep in the real image) is what proves the anchors match.
"""
import ast
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "files" / "patch_mamba_state_idx.py"

WORKER_ORIG = '''\
class Base:
    pass


class MambaHybridModelState(Base):
    def add_request(self, req_index, new_req_data):
        self._mamba_state_idx_gpu[req_index].fill_(
                (new_req_data.num_computed_tokens - 1) // self.cache_config.block_size
        )
'''

SCHED_ORIG = '''\
class Scheduler:
    def _mamba_block_aligned_split(self, request, num_new_tokens):
        block_size = self.cache_config.block_size
        # The last block-aligned position whose state can be cached.
        last = (request.num_computed_tokens + num_new_tokens) // block_size * block_size
        return last
'''


def run(workdir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(workdir / GENERATOR.name)],
        capture_output=True, text=True,
    )


class PatchMambaStateIdxTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        shutil.copy(GENERATOR, self.tmp / GENERATOR.name)
        (self.tmp / "mamba_hybrid_patched.py.orig").write_text(WORKER_ORIG)
        (self.tmp / "scheduler_patched.py.orig").write_text(SCHED_ORIG)

    def test_patch_applies_and_parses(self):
        result = run(self.tmp)
        self.assertEqual(result.returncode, 0, result.stderr)
        worker = (self.tmp / "mamba_hybrid_patched.py").read_text()
        sched = (self.tmp / "scheduler_patched.py").read_text()
        ast.parse(worker)
        ast.parse(sched)
        self.assertNotIn("// self.cache_config.block_size", worker)
        self.assertIn("_resolve_mamba_block_size(self)", worker)
        self.assertNotIn("self.cache_config.block_size\n        # The last", sched)
        self.assertIn("block_size = self.block_size", sched)

    def test_worker_seed_uses_mamba_block_size(self):
        self.assertEqual(run(self.tmp).returncode, 0)
        ns: dict = {}
        exec(compile((self.tmp / "mamba_hybrid_patched.py").read_text(), "w", "exec"), ns)
        helper = ns["_resolve_mamba_block_size"]

        # Mamba spec wins over the (wrong) min-group scheduler block size.
        state = SimpleNamespace(
            _mamba_spec=SimpleNamespace(block_size=1664),
            cache_config=SimpleNamespace(block_size=16, mamba_block_size=1600),
        )
        self.assertEqual(helper(state), 1664)
        # Without a spec, cache_config.mamba_block_size is used.
        state = SimpleNamespace(
            cache_config=SimpleNamespace(block_size=16, mamba_block_size=1600)
        )
        self.assertEqual(helper(state), 1600)

        # The seeded column for a hit: (1665 - 1) // 1664 == 1, not 104.
        seed = (1665 - 1) // helper(SimpleNamespace(
            _mamba_spec=SimpleNamespace(block_size=1664),
            cache_config=SimpleNamespace(block_size=16),
        ))
        self.assertEqual(seed, 1)

    def test_fails_loud_when_anchor_missing(self):
        (self.tmp / "scheduler_patched.py.orig").write_text(
            SCHED_ORIG.replace("cache_config.block_size", "cache_config.other")
        )
        result = run(self.tmp)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("anchor", result.stderr)

    def test_fails_loud_when_anchor_not_unique(self):
        (self.tmp / "mamba_hybrid_patched.py.orig").write_text(WORKER_ORIG + WORKER_ORIG)
        result = run(self.tmp)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("count=2", result.stderr)

    def test_missing_orig_is_reported(self):
        (self.tmp / "mamba_hybrid_patched.py.orig").unlink()
        result = run(self.tmp)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing", result.stderr)

    def test_chains_onto_block_drop_output_when_present(self):
        """MTP_DISABLE_BLOCK_DROP=1: start.sh runs patch_block_drop.py first, and its
        scheduler.py output (a different edit, same file) must carry our fix too --
        not scheduler_patched.py.orig, which start.sh would then mount unused."""
        block_dropped = SCHED_ORIG.replace(
            "class Scheduler:\n",
            "class Scheduler:\n"
            "    def other_method(self):\n"
            "        if self.use_eagle_block_drop:  # renamed by patch_block_drop.py\n"
            "            pass\n",
        )
        sched_dir = self.tmp / "block_drop" / "v1" / "core" / "sched"
        sched_dir.mkdir(parents=True)
        (sched_dir / "scheduler.py").write_text(block_dropped)
        # A stale/wrong scheduler_patched.py.orig must NOT be the one read from.
        (self.tmp / "scheduler_patched.py.orig").write_text(SCHED_ORIG.replace("cache_config", "BOGUS"))

        result = run(self.tmp)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("chaining onto", result.stdout)
        sched = (self.tmp / "scheduler_patched.py").read_text()
        ast.parse(sched)
        self.assertIn("use_eagle_block_drop", sched)          # block-drop's edit survived
        self.assertIn("block_size = self.block_size", sched)  # our edit was applied on top
        self.assertNotIn("BOGUS", sched)                      # read from block_drop output, not .orig


if __name__ == "__main__":
    unittest.main()
