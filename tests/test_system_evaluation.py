"""
Unit tests for Module 18 — System Performance Evaluation.
"""

import tempfile
import unittest
from pathlib import Path

from src.system_evaluation import (
    SystemPerformanceEvaluator,
    ScenarioTestResult,
    SystemPerformanceSummary,
    SyntheticTestFrameGenerator
)


class TestSystemEvaluation(unittest.TestCase):
    """Test suite verifying SystemPerformanceEvaluator execution and report generation."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)

    def tearDown(self) -> None:
        import logging
        for name in logging.root.manager.loggerDict:
            lg = logging.getLogger(name)
            for h in list(lg.handlers):
                h.close()
                lg.removeHandler(h)
        try:
            self.tmp_dir.cleanup()
        except Exception:
            pass

    def test_synthetic_frame_generator(self) -> None:
        """Verifies synthetic test frame generation with geometric parameters."""
        frame, gt_boxes = SyntheticTestFrameGenerator.create_synthetic_wheel_frame(
            width=640, height=480, tilt_deg=10.0, num_wheels=2
        )

        self.assertEqual(frame.shape, (480, 640, 3))
        self.assertEqual(len(gt_boxes), 2)

    def test_evaluator_runs_all_scenarios(self) -> None:
        """Verifies SystemPerformanceEvaluator runs all 14 scenarios and produces reports."""
        evaluator = SystemPerformanceEvaluator(results_dir=self.tmp_path)
        summary = evaluator.run_all_evaluations()

        self.assertIsInstance(summary, SystemPerformanceSummary)
        self.assertEqual(summary.total_scenarios_tested, 14)
        self.assertGreater(summary.scenarios_passed, 0)
        self.assertGreater(summary.overall_classification_accuracy, 0.0)

        # Check exported report files
        csv_scen = self.tmp_path / "scenario_test_results.csv"
        csv_sum = self.tmp_path / "system_performance_metrics.csv"
        md_rep = self.tmp_path / "system_performance_report.md"

        self.assertTrue(csv_scen.exists())
        self.assertTrue(csv_sum.exists())
        self.assertTrue(md_rep.exists())


if __name__ == "__main__":
    unittest.main()
