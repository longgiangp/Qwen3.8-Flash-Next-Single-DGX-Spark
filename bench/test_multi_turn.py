#!/usr/bin/env python3
import argparse
import unittest
from unittest import mock

import multi_turn as mt


class MultiTurnTests(unittest.TestCase):
    def test_run_conversation_appends_real_replies_and_grows_prompt(self):
        args = argparse.Namespace(base_url="http://x", model="m", max_tokens=64, turns=3,
                                  continue_on_crash=False)
        seen_message_counts = []

        def scripted(base, model, messages, max_tokens):
            seen_message_counts.append(len(messages))
            n = len(messages)
            return {"ttft": 0.1 * n, "content": f"reply-{n}", "prompt_tokens": 100 * n, "cached_tokens": 10 * n}

        with mock.patch.object(mt, "stream_chat", side_effect=scripted), \
             mock.patch.object(mt, "is_healthy", return_value=True):
            turns = mt.run_conversation(args, "system prompt", trial=0)
        self.assertEqual(len(turns), 3)
        self.assertEqual([t["turn"] for t in turns], [0, 1, 2])
        self.assertFalse(any(t["crashed"] for t in turns))
        # Each turn's message count grows by 2 (one user, one assistant) over the last.
        self.assertEqual(seen_message_counts, [2, 4, 6])
        self.assertEqual(turns[0]["cached_tokens"], 20)

    def test_stops_on_crash_by_default(self):
        args = argparse.Namespace(base_url="http://x", model="m", max_tokens=64, turns=5,
                                  continue_on_crash=False)
        with mock.patch.object(mt, "stream_chat", side_effect=[
                {"ttft": 0.1, "content": "ok", "prompt_tokens": 100, "cached_tokens": 0},
                TimeoutError("boom"),
             ]), \
             mock.patch.object(mt, "is_healthy", return_value=True):
            turns = mt.run_conversation(args, "system prompt", trial=0)
        self.assertEqual(len(turns), 2)
        self.assertFalse(turns[0]["crashed"])
        self.assertTrue(turns[1]["crashed"])
        self.assertIn("TimeoutError", turns[1]["error"])

    def test_unhealthy_after_a_turn_is_recorded_and_stops(self):
        args = argparse.Namespace(base_url="http://x", model="m", max_tokens=64, turns=5,
                                  continue_on_crash=False)
        with mock.patch.object(mt, "stream_chat",
                               return_value={"ttft": 0.1, "content": "ok", "prompt_tokens": 100, "cached_tokens": 0}), \
             mock.patch.object(mt, "is_healthy", return_value=False):
            turns = mt.run_conversation(args, "system prompt", trial=0)
        self.assertEqual(len(turns), 2)
        self.assertTrue(turns[1]["crashed"])
        self.assertIn("unhealthy", turns[1]["error"])

    def test_trials_use_independent_salts(self):
        args = argparse.Namespace(base_url="http://x", model="m", max_tokens=64, turns=1,
                                  continue_on_crash=False)
        seen_systems = []

        def scripted(base, model, messages, max_tokens):
            seen_systems.append(messages[0]["content"])
            return {"ttft": 0.1, "content": "ok", "prompt_tokens": 100, "cached_tokens": 0}

        with mock.patch.object(mt, "stream_chat", side_effect=scripted), \
             mock.patch.object(mt, "is_healthy", return_value=True):
            mt.run_conversation(args, "shared corpus text", trial=0)
            mt.run_conversation(args, "shared corpus text", trial=1)
        self.assertNotEqual(seen_systems[0], seen_systems[1])
        self.assertTrue(seen_systems[0].endswith("shared corpus text"))

    def test_summarize_groups_by_turn_and_skips_crashed_rows(self):
        trials = [
            [{"turn": 0, "crashed": False, "ttft": 1.0, "cached_tokens": 0, "prompt_tokens": 100},
             {"turn": 1, "crashed": False, "ttft": 2.0, "cached_tokens": 500, "prompt_tokens": 700}],
            [{"turn": 0, "crashed": False, "ttft": 1.5, "cached_tokens": 0, "prompt_tokens": 100},
             {"turn": 1, "crashed": True, "error": "boom"}],
        ]
        summary = mt.summarize(trials)
        self.assertEqual(summary["0"]["n"], 2)
        self.assertEqual(summary["0"]["ttft_median_s"], 1.25)
        self.assertEqual(summary["1"]["n"], 1)  # the crashed row is excluded
        self.assertEqual(summary["1"]["cached_tokens_median"], 500)

    def test_parser_rejects_out_of_range_turns_and_trials(self):
        p = mt.parser()
        args = p.parse_args(["--turns", "4", "--trials", "3"])
        self.assertEqual((args.turns, args.trials), (4, 3))


if __name__ == "__main__":
    unittest.main()
