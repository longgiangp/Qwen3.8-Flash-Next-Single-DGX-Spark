#!/usr/bin/env python3
import argparse
import unittest
from unittest import mock

import needle


class NeedleTests(unittest.TestCase):
    FILLER = "".join(f"line {i} of filler text\n" for i in range(1000))

    def test_needle_lands_at_depth_and_question_is_last(self):
        for depth in (0, 5, 50, 95, 100):
            prompt = needle.build_prompt(self.FILLER, depth, "salt", "aaaa-bbbb-cccc")
            self.assertTrue(prompt.startswith("Validation salt salt."))
            self.assertTrue(prompt.endswith(needle.QUESTION))
            self.assertEqual(prompt.count("aaaa-bbbb-cccc"), 1)
            body = prompt[len("Validation salt salt.\n"):-len(needle.QUESTION)]
            position = body.index("aaaa-bbbb-cccc") / len(body)
            self.assertAlmostEqual(position, depth / 100, delta=0.02)

    def test_parse_depths_rejects_out_of_range(self):
        self.assertEqual(needle.parse_depths("5,50,95"), [5, 50, 95])
        with self.assertRaises(argparse.ArgumentTypeError):
            needle.parse_depths("101")

    def test_trial_scores_cold_and_warm_separately(self):
        args = argparse.Namespace(base_url="x", model="m", max_tokens=8)
        answers = iter([
            {"text": "nothing", "prompt_tokens": 9, "cached_tokens": 0},
            {"text": "still nothing", "prompt_tokens": 9, "cached_tokens": 4992},
        ])
        with mock.patch.object(needle, "ask", side_effect=lambda *a: next(answers)):
            result = needle.trial(args, self.FILLER, 50)
        self.assertFalse(result["cold_found"])
        self.assertFalse(result["warm_found"])
        self.assertEqual(result["warm_cached_tokens"], 4992)

    def test_summarize_counts_per_depth(self):
        trials = [
            {"depth_pct": 95, "cold_found": True, "warm_found": False, "warm_cached_tokens": 100},
            {"depth_pct": 95, "cold_found": True, "warm_found": True, "warm_cached_tokens": 0},
            {"depth_pct": 5, "cold_found": True, "warm_found": True, "warm_cached_tokens": 100},
        ]
        summary = needle.summarize(trials)
        self.assertEqual(summary["95"], {"trials": 2, "cold_found": 2, "warm_found": 1, "warm_cache_hits": 1})
        self.assertEqual(list(summary), ["5", "95"])


if __name__ == "__main__":
    unittest.main()
