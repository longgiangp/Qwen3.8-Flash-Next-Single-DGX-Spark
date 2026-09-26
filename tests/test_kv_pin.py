#!/usr/bin/env python3
"""CPU-only: the KV pool is pinned, and with the flag name vLLM actually has."""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class KvPinTests(unittest.TestCase):
    def test_start_sh_uses_the_full_flag_name(self):
        source = (ROOT / "start.sh").read_text()
        self.assertIn('"--kv-cache-memory-bytes" "$KV_CACHE_MEMORY"', source)
        self.assertNotIn('"--kv-cache-memory" ', source)   # relied on argparse abbreviation

    def test_release_env_pins_a_sane_pool(self):
        match = re.search(r"^KV_CACHE_MEMORY=(\d+)$", (ROOT / "start.sh").parent.joinpath(".env.sample").read_text(), re.M)
        self.assertIsNotNone(match)
        gib = int(match.group(1)) / 2**30
        self.assertGreaterEqual(gib, 4.5)   # one 262,144-token request needs ~3.92 GiB; keep >1.15x
        self.assertLessEqual(gib, 8.0)      # above this the 10 GiB MemAvailable margin is at risk


if __name__ == "__main__":
    unittest.main()
