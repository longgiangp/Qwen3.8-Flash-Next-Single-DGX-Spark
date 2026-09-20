"""Offline tests for cache_correctness; no model server is contacted."""

import argparse
import pathlib
import tempfile
import unittest
from unittest import mock

import cache_correctness as validation


class CacheCorrectnessTests(unittest.TestCase):
    def test_parse_sizes(self):
        self.assertEqual(validation.parse_sizes("0,8192,32768"), [0, 8192, 32768])
        for invalid in ("", "x", "-1", "8192,8192", "250001"):
            with self.subTest(invalid=invalid), self.assertRaises(Exception):
                validation.parse_sizes(invalid)

    def test_answer_hash_does_not_return_text(self):
        digest = validation.answer_hash("private generated answer")
        self.assertEqual(len(digest), 64)
        self.assertNotIn("private", digest)

    def test_samples_match(self):
        left = {"hash": "a", "scores": {"token": -0.25}}
        same = {"hash": "a", "scores": {"token": -0.2500001}}
        changed = {"hash": "b", "scores": {"other": -0.25}}
        left.update(token_fingerprint="tokens")
        same.update(token_fingerprint="tokens")
        changed.update(token_fingerprint="changed")
        self.assertEqual(validation.samples_match(left, same), (True, True, True))
        self.assertEqual(validation.samples_match(left, changed), (False, False, False))

    def test_cache_case_requires_correctness_and_hit(self):
        args = argparse.Namespace(
            base_url="http://127.0.0.1:8888", model="model",
            corpus=pathlib.Path("unused"), max_tokens=32, min_tokens=8,
            require_prefix_hit=True,
        )
        first = {
            "hash": "same", "chars": 4, "finish_reason": "stop",
            "scores": {"x": -0.1}, "token_count": 8,
            "token_fingerprint": "tokens", "valid": True,
        }
        with mock.patch.object(validation, "make_prompt", return_value=("prompt", 8192)), \
             mock.patch.object(validation, "completion", side_effect=[first, dict(first)]), \
             mock.patch.object(validation.runtime, "prefix_hits", side_effect=[10.0, 10.0, 100.0]):
            result = validation.cache_case(args, 8192)
        self.assertTrue(result["passed"])
        self.assertTrue(result["text_equal"])
        self.assertTrue(result["first_logprobs_equal"])
        self.assertTrue(result["repeated_prefix_hit"])
        self.assertTrue(result["valid_samples"])
        self.assertTrue(result["generated_tokens_equal"])

    def test_cached_tokens_field_is_required_when_asked(self):
        args = argparse.Namespace(
            base_url="http://127.0.0.1:8888", model="model",
            corpus=pathlib.Path("unused"), max_tokens=32, min_tokens=8,
            require_prefix_hit=False, require_cached_tokens=True,
        )
        base = {
            "hash": "same", "chars": 4, "finish_reason": "stop",
            "scores": {"x": -0.1}, "token_count": 8,
            "token_fingerprint": "tokens", "valid": True,
        }

        def run(first_cached, repeat_cached):
            first = dict(base, cached_tokens=first_cached)
            repeat = dict(base, cached_tokens=repeat_cached)
            with mock.patch.object(validation, "make_prompt", return_value=("prompt", 8192)), \
                 mock.patch.object(validation, "completion", side_effect=[first, repeat]), \
                 mock.patch.object(validation.runtime, "prefix_hits", return_value=None):
                return validation.cache_case(args, 8192)

        self.assertTrue(run(0, 4992)["passed"])
        self.assertFalse(run(0, 0)["passed"])      # no reuse
        self.assertFalse(run(None, None)["passed"])  # server flag missing
        self.assertFalse(run(4992, 4992)["passed"])  # first call was not cold
        self.assertEqual(run(0, 4992)["cached_tokens_repeat"], 4992)

    def test_empty_samples_are_never_valid(self):
        args = argparse.Namespace(
            base_url="http://127.0.0.1:8888", model="model",
            corpus=pathlib.Path("unused"), max_tokens=32, min_tokens=8,
            require_prefix_hit=False,
        )
        empty = {
            "hash": validation.answer_hash(""), "chars": 0,
            "finish_reason": "stop", "scores": {"x": -0.1},
            "token_count": 8, "token_fingerprint": "tokens", "valid": False,
        }
        with mock.patch.object(validation, "make_prompt", return_value=("prompt", 8192)), \
             mock.patch.object(validation, "completion", side_effect=[empty, dict(empty)]), \
             mock.patch.object(validation.runtime, "prefix_hits", return_value=None):
            result = validation.cache_case(args, 8192)
        self.assertFalse(result["passed"])
        self.assertFalse(result["valid_samples"])


if __name__ == "__main__":
    unittest.main()
