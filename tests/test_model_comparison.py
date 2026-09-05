"""
Unit Tests for Machine Learning Model Comparison Subsystem.

Tests:
1. Candidate classifier training, latency profiling, and evaluation metric calculation.
2. Comparative benchmarking across Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting.
3. Tradeoff-based optimal model selection.
4. Export of comparison CSV summary, visualization plot, and markdown report.
"""

from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_comparison import ModelComparer, ModelBenchmarkResult


class TestModelComparisonSubsystem(unittest.TestCase):
    """Unit test suite for ModelComparer component."""

    def setUp(self) -> None:
        """Sets up synthetic preprocessed feature datasets for candidate benchmarking."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.comparer = ModelComparer(random_seed=42)

        np.random.seed(42)
        n_train = 60
        n_test = 30
        n_features = 19

        self.X_train = np.random.randn(n_train, n_features).astype(np.float32)
        self.y_train = pd.Series(["NORMAL"] * 20 + ["POSSIBLE_MISALIGNMENT"] * 20 + ["SEVERE_MISALIGNMENT"] * 20)

        self.X_test = np.random.randn(n_test, n_features).astype(np.float32)
        self.y_test = pd.Series(["NORMAL"] * 10 + ["POSSIBLE_MISALIGNMENT"] * 10 + ["SEVERE_MISALIGNMENT"] * 10)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_evaluate_single_candidate(self) -> None:
        """Tests single classifier evaluation and latency profiling."""
        clf = self.comparer.candidate_models["Decision Tree"]
        res = self.comparer.evaluate_candidate(
            "Decision Tree", clf, self.X_train, self.y_train, self.X_test, self.y_test
        )

        self.assertIsInstance(res, ModelBenchmarkResult)
        self.assertEqual(res.model_name, "Decision Tree")
        self.assertGreaterEqual(res.accuracy, 0.0)
        self.assertLessEqual(res.accuracy, 1.0)
        self.assertGreater(res.inference_time_ms, 0.0)
        self.assertGreater(res.model_size_kb, 0.0)

    def test_run_comparison_selection(self) -> None:
        """Tests model comparison across all candidates and optimal model selection."""
        results, best_res, best_clf = self.comparer.run_comparison(
            self.X_train, self.y_train, self.X_test, self.y_test
        )

        self.assertEqual(len(results), 4)
        selected_count = sum(1 for r in results if r.is_selected)
        self.assertEqual(selected_count, 1)
        self.assertEqual(best_res.model_name, next(r.model_name for r in results if r.is_selected))
        self.assertIsNotNone(best_clf)

    def test_export_comparison_results(self) -> None:
        """Tests CSV, plot, and Markdown report creation."""
        results, _, _ = self.comparer.run_comparison(
            self.X_train, self.y_train, self.X_test, self.y_test
        )

        csv_p, plot_p, report_p = self.comparer.export_comparison_results(results, output_dir=self.temp_dir.name)

        self.assertTrue(csv_p.exists())
        self.assertTrue(plot_p.exists())
        self.assertTrue(report_p.exists())

        df_csv = pd.read_csv(csv_p)
        self.assertEqual(len(df_csv), 4)
        self.assertIn("inference_time_ms", df_csv.columns)
        self.assertIn("composite_score", df_csv.columns)


if __name__ == "__main__":
    unittest.main()
