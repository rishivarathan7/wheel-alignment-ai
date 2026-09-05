"""
CLI Execution Script for Module 18 — System Performance and Reliability Evaluation.

Runs automated empirical benchmarking across 14 operational scenarios and exports
machine-readable CSV metrics and human-readable Markdown evaluation reports.
"""

import argparse
import sys
from pathlib import Path

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RESULTS_DIR, setup_logger
from src.system_evaluation import SystemPerformanceEvaluator

logger = setup_logger("evaluate_system_performance")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Module 18 System Performance and Reliability Evaluation.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(RESULTS_DIR),
        help="Directory to save evaluation CSV and Markdown reports (default: results/)."
    )

    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing SystemPerformanceEvaluator...")
    evaluator = SystemPerformanceEvaluator(results_dir=out_dir)
    summary = evaluator.run_all_evaluations()

    print("\n" + "=" * 80)
    print(" [MODULE 18 - SYSTEM PERFORMANCE EVALUATION COMPLETED]")
    print("=" * 80)
    print(f" Total Scenarios Tested : {summary.total_scenarios_tested}")
    print(f" Scenarios Passed       : {summary.scenarios_passed} / {summary.total_scenarios_tested}")
    print(f" Pass Rate              : {(summary.scenarios_passed / max(1, summary.total_scenarios_tested)) * 100.0:.2f}%")
    print(f" Classification Accuracy: {summary.overall_classification_accuracy * 100.2:.2f}%")
    print(f" Macro F1-Score         : {summary.macro_f1_score:.4f}")
    print(f" Avg Pipeline FPS       : {summary.avg_pipeline_fps:.1f} FPS")
    print(f" Avg Frame Latency      : {summary.avg_frame_latency_ms:.2f} ms")
    print(f" Scenario Results CSV   : {out_dir / 'scenario_test_results.csv'}")
    print(f" Summary Metrics CSV    : {out_dir / 'system_performance_metrics.csv'}")
    print(f" Markdown Report        : {out_dir / 'system_performance_report.md'}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
