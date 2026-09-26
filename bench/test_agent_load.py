#!/usr/bin/env python3
import unittest
import zlib

import agent_load as load


class AgentLoadTests(unittest.TestCase):
    def test_solid_png_is_wellformed(self):
        png = load.solid_png(4, 3, (10, 20, 30))
        self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))
        idat = png[png.index(b"IDAT") + 4:png.index(b"IEND") - 8]
        raw = zlib.decompress(idat)
        self.assertEqual(len(raw), 3 * (1 + 4 * 3))          # 3 rows: filter byte + 4 px * RGB
        self.assertEqual(raw[1:4], bytes((10, 20, 30)))

    def test_parser_defaults_and_modes(self):
        args = load.parser().parse_args(["load", "--levels", "1", "8"])
        self.assertEqual(args.levels, [1, 8])
        self.assertEqual(args.mode, "load")
        with self.assertRaises(SystemExit):
            load.parser().parse_args(["bogus"])

    def test_tool_schema_requires_path(self):
        self.assertIn("filePath", load.TOOL["function"]["parameters"]["required"])


if __name__ == "__main__":
    unittest.main()
