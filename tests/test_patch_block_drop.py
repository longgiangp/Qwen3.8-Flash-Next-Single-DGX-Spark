#!/usr/bin/env python3
"""CPU-only check of files/patch_block_drop.py against synthetic sources.

The fixtures reproduce only the anchor lines; they are NOT the image's files.
Verified against the real, extracted image sources once by hand (see the
commit message for this file): all three anchors are unique in the pinned
image's config/speculative.py, v1/core/kv_cache_utils.py and
v1/core/sched/scheduler.py, disable_eagle_block_drop is not already known, and
the two block-dropped output files (config/speculative.py,
v1/core/kv_cache_utils.py) plus the chained scheduler_patched.py all parse.
"""
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "files" / "patch_block_drop.py"

SPEC_ORIG = '''\
class SpeculativeConfig:
    method: str = "none"

    def use_eagle(self) -> bool:
        return self.method in ("eagle", "eagle3", "mtp", "dflash", "dspark")

    use_local_argmax_reduction: bool = False
'''

KV_CACHE_UTILS_ORIG = '''\
def get_kv_connector_cache_layout(vllm_config, spec_config):
    if spec_config is None or not spec_config.use_eagle():
        return None
'''

SCHED_ORIG = '''\
class Scheduler:
    def __init__(self, vllm_config, speculative_config=None):
        self.use_eagle = False
        if speculative_config is not None:
            self.use_eagle = speculative_config.use_eagle()
        self.kv_cache_manager = KVCacheManager(
            use_eagle=self.use_eagle,
        )

    def _mamba_block_aligned_split(self, request, num_new_tokens):
        block_size = self.cache_config.block_size
        # The last block-aligned position whose state can be cached.
        last_cache_position = request.num_tokens - request.num_tokens % block_size
        if self.use_eagle:
            last_cache_position = max(last_cache_position - block_size, 0)
        return last_cache_position
'''


def run(orig_dir: Path, out_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(orig_dir.parent / GENERATOR.name), str(orig_dir), str(out_dir)],
        capture_output=True, text=True,
    )


class PatchBlockDropTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        shutil.copy(GENERATOR, self.tmp / GENERATOR.name)
        self.orig = self.tmp / "orig"
        (self.orig / "config").mkdir(parents=True)
        (self.orig / "v1" / "core" / "sched").mkdir(parents=True)
        (self.orig / "config" / "speculative.py").write_text(SPEC_ORIG)
        (self.orig / "v1" / "core" / "kv_cache_utils.py").write_text(KV_CACHE_UTILS_ORIG)
        (self.orig / "v1" / "core" / "sched" / "scheduler.py").write_text(SCHED_ORIG)
        self.out = self.tmp / "out"

    def test_list_gives_exactly_the_three_reachable_files(self):
        result = subprocess.run([sys.executable, str(self.tmp / GENERATOR.name), "--list"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        files = result.stdout.split()
        self.assertEqual(files, ["config/speculative.py", "v1/core/kv_cache_utils.py",
                                  "v1/core/sched/scheduler.py"])
        # No KV-connector file: this deployment never configures a KV connector.
        self.assertFalse(any("kv_connector" in f or "kv_offload" in f for f in files))

    def test_patches_all_three_and_parses(self):
        result = run(self.orig, self.out)
        self.assertEqual(result.returncode, 0, result.stderr)
        spec = (self.out / "config" / "speculative.py").read_text()
        kv = (self.out / "v1" / "core" / "kv_cache_utils.py").read_text()
        sched = (self.out / "v1" / "core" / "sched" / "scheduler.py").read_text()
        import ast
        ast.parse(spec); ast.parse(kv); ast.parse(sched)
        self.assertIn("disable_eagle_block_drop: bool = False", spec)
        self.assertIn("def use_eagle_block_drop(self) -> bool:", spec)
        self.assertIn("spec_config.use_eagle_block_drop()", kv)
        self.assertNotIn("spec_config.use_eagle()\n", kv)
        self.assertIn("self.use_eagle_block_drop = False", sched)
        self.assertIn("use_eagle=self.use_eagle_block_drop,", sched)
        self.assertIn("if self.use_eagle_block_drop:\n            last_cache_position", sched)
        # The mamba fix's own anchor (a different line, same function) is untouched here --
        # files/patch_mamba_state_idx.py chains onto this output separately.
        self.assertIn("block_size = self.cache_config.block_size\n        # The last", sched)

    def test_idempotent(self):
        run(self.orig, self.out)
        first = (self.out / "config" / "speculative.py").read_text()
        result = run(self.orig, self.out)
        self.assertEqual(result.returncode, 0)
        self.assertEqual((self.out / "config" / "speculative.py").read_text(), first)

    def test_image_already_knowing_the_key_writes_nothing(self):
        spec = self.orig / "config" / "speculative.py"
        spec.write_text(spec.read_text().replace(
            "use_local_argmax_reduction: bool = False\n",
            "disable_eagle_block_drop: bool = False\n    use_local_argmax_reduction: bool = False\n",
        ))
        result = run(self.orig, self.out)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("nothing to mount", result.stdout)
        self.assertFalse((self.out / "config" / "speculative.py").exists())

    def test_failed_anchor_writes_no_partial_output(self):
        sched = self.orig / "v1" / "core" / "sched" / "scheduler.py"
        sched.write_text(sched.read_text().replace("self.use_eagle = False\n", "self.eager = False\n"))
        result = run(self.orig, self.out)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("anchor not unique/missing", result.stderr)
        self.assertFalse((self.out / "config" / "speculative.py").exists())
        self.assertFalse((self.out / "v1" / "core" / "kv_cache_utils.py").exists())

    def test_missing_source_is_reported(self):
        (self.orig / "config" / "speculative.py").unlink()
        result = run(self.orig, self.out)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing", result.stderr)


if __name__ == "__main__":
    unittest.main()
