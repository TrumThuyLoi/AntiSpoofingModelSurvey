"""Tests cho scripts/run_hairymax_onnx_drivers_bpcer.py."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_hairymax_onnx_drivers_bpcer import is_live_accepted  # noqa: E402


class TestIsLiveAccepted(unittest.TestCase):
    def test_live_pass(self) -> None:
        self.assertTrue(is_live_accepted([0.9, 0.1], threshold=0.5))

    def test_live_fail_low_score(self) -> None:
        self.assertFalse(is_live_accepted([0.4, 0.6], threshold=0.5))

    def test_spoof_class(self) -> None:
        self.assertFalse(is_live_accepted([0.2, 0.8], threshold=0.5))


if __name__ == "__main__":
    unittest.main()
