#!/usr/bin/env python3
import argparse
import unittest
from unittest import mock

import prefix_extend as pe


class PrefixExtendTests(unittest.TestCase):
    def test_default_cases_scale_with_block_size(self):
        cases = pe.default_cases(1664)
        labels = [c.label for c in cases]
        self.assertEqual(len(labels), len(set(labels)))  # no duplicate labels
        one_block = next(c for c in cases if c.label == "one-block")
        self.assertEqual(one_block.extend_tokens, 1664)
        boundary = next(c for c in cases if c.label == "boundary-cross")
        self.assertEqual(boundary.extend_tokens, 1664 + 40)
        sub_block = next(c for c in cases if c.label == "sub-block")
        self.assertLess(sub_block.extend_tokens, 1664)

    def test_parse_cases(self):
        cases = pe.parse_cases("100,50; 200,10", 1664)
        self.assertEqual([(c.base_tokens, c.extend_tokens) for c in cases], [(100, 50), (200, 10)])
        with self.assertRaises(argparse.ArgumentTypeError):
            pe.parse_cases("bad", 1664)
        with self.assertRaises(argparse.ArgumentTypeError):
            pe.parse_cases("", 1664)
        with self.assertRaises(argparse.ArgumentTypeError):
            pe.parse_cases("1,2,3", 1664)

    def test_slice_chars_is_monotonic_and_never_empty(self):
        filler = "x" * 1000
        self.assertEqual(pe.slice_chars(filler, 10, 2.0), "x" * 20)
        self.assertGreaterEqual(len(pe.slice_chars(filler, 0, 2.0)), 1)

    def test_trial_flags_a_crash_on_the_base_call(self):
        args = argparse.Namespace(base_url="http://x", model="m", max_tokens=64)
        case = pe.Case("probe", 100, 50)
        with mock.patch.object(pe, "ask", side_effect=TimeoutError("boom")):
            result = pe.trial(args, "f" * 10_000, 4.0, case)
        self.assertTrue(result["crashed"])
        self.assertIn("base call", result["error"])
        self.assertEqual(pe.verdict(result), "CRASH")

    def test_trial_flags_an_unhealthy_server_between_the_two_calls(self):
        args = argparse.Namespace(base_url="http://x", model="m", max_tokens=64)
        case = pe.Case("probe", 100, 50)
        base_reply = {"text": "ok", "cached_tokens": 0, "prompt_tokens": 100, "finish_reason": "stop"}
        with mock.patch.object(pe, "ask", return_value=base_reply), \
             mock.patch.object(pe, "is_healthy", return_value=False):
            result = pe.trial(args, "f" * 10_000, 4.0, case)
        self.assertTrue(result["crashed"])
        self.assertIn("unhealthy", result["error"])
        self.assertEqual(pe.verdict(result), "CRASH")

    def test_trial_flags_a_crash_on_the_extension_call(self):
        args = argparse.Namespace(base_url="http://x", model="m", max_tokens=64)
        case = pe.Case("probe", 100, 50)
        base_reply = {"text": "ok", "cached_tokens": 0, "prompt_tokens": 100, "finish_reason": "stop"}
        with mock.patch.object(pe, "ask", side_effect=[base_reply, ConnectionResetError("boom")]), \
             mock.patch.object(pe, "is_healthy", return_value=True):
            result = pe.trial(args, "f" * 10_000, 4.0, case)
        self.assertTrue(result["crashed"])
        self.assertIn("extension call", result["error"])

    def test_trial_detects_wrong_recall_and_missing_cache_hit(self):
        args = argparse.Namespace(base_url="http://x", model="m", max_tokens=64)
        case = pe.Case("probe", 100, 50)
        calls = {"n": 0}

        def scripted_ask(base, model, content, max_tokens):
            calls["n"] += 1
            if calls["n"] == 1:
                scripted_ask.passphrase = content.split("secret code is ")[1].split(".")[0]
                return {"text": f"ack {scripted_ask.passphrase}", "cached_tokens": 0,
                         "prompt_tokens": 100, "finish_reason": "stop"}
            return {"text": "I don't know", "cached_tokens": 0, "prompt_tokens": 150, "finish_reason": "stop"}

        with mock.patch.object(pe, "ask", side_effect=scripted_ask), \
             mock.patch.object(pe, "is_healthy", return_value=True):
            result = pe.trial(args, "f" * 10_000, 4.0, case)
        self.assertFalse(result["crashed"])
        self.assertTrue(result["base_found"])
        self.assertFalse(result["extend_found"])
        self.assertTrue(result["wrong_recall"])
        self.assertFalse(result["reused_cache"])
        self.assertFalse(result["passed"])
        self.assertEqual(pe.verdict(result), "RECALL-FAIL")

    def test_trial_no_cache_hit_but_recall_ok_is_still_not_passed(self):
        args = argparse.Namespace(base_url="http://x", model="m", max_tokens=64)
        case = pe.Case("probe", 100, 50)
        calls = {"n": 0}

        def scripted_ask(base, model, content, max_tokens):
            calls["n"] += 1
            if calls["n"] == 1:
                scripted_ask.passphrase = content.split("secret code is ")[1].split(".")[0]
                return {"text": "ok", "cached_tokens": 0, "prompt_tokens": 100, "finish_reason": "stop"}
            return {"text": f"the code is {scripted_ask.passphrase}", "cached_tokens": 0,
                     "prompt_tokens": 150, "finish_reason": "stop"}

        with mock.patch.object(pe, "ask", side_effect=scripted_ask), \
             mock.patch.object(pe, "is_healthy", return_value=True):
            result = pe.trial(args, "f" * 10_000, 4.0, case)
        self.assertFalse(result["wrong_recall"])
        self.assertFalse(result["reused_cache"])
        self.assertFalse(result["passed"])
        self.assertEqual(pe.verdict(result), "NO-CACHE-HIT")

    def test_trial_passes_when_cache_reused_and_recalled(self):
        args = argparse.Namespace(base_url="http://x", model="m", max_tokens=64)
        case = pe.Case("probe", 100, 50)
        calls = {"n": 0}

        def scripted_ask(base, model, content, max_tokens):
            calls["n"] += 1
            if calls["n"] == 1:
                scripted_ask.passphrase = content.split("secret code is ")[1].split(".")[0]
                return {"text": "ok", "cached_tokens": 0, "prompt_tokens": 100, "finish_reason": "stop"}
            return {"text": f"the code is {scripted_ask.passphrase}", "cached_tokens": 100,
                     "prompt_tokens": 150, "finish_reason": "stop"}

        with mock.patch.object(pe, "ask", side_effect=scripted_ask), \
             mock.patch.object(pe, "is_healthy", return_value=True):
            result = pe.trial(args, "f" * 10_000, 4.0, case)
        self.assertTrue(result["passed"])
        self.assertEqual(pe.verdict(result), "ok")


if __name__ == "__main__":
    unittest.main()
