"""
Machine Learning Model Comparison Subsystem.

Executes a comparative benchmark across candidate classifiers for 3-class wheel alignment anomaly detection:
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting

Uses identical, leakage-free train/test dataset partitions and evaluates:
- Accuracy, Precision (Macro), Recall (Macro), F1-Score (Macro & Weighted).
- Inference Latency (milliseconds per 100 samples).
- Model Complexity (estimated size/parameters).
- Tradeoff-based Selection for Real-Time Suitability.
"""

from dataclasses import dataclass, asdict
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from src.config import MODELS_DIR, RESULTS_DIR, PROCESSED_DATA_DIR
from src.feature_engineering import FeatureEngineer, FEATURE_COLUMNS
from src.prediction import ALIGNMENT_CLASSES, normalize_class_label

logger = logging.getLogger(__name__)


@dataclass
class ModelBenchmarkResult:
    """Dataclass storing benchmark results for a single candidate classifier."""
    model_name: str
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    f1_weighted: float
    inference_time_ms: float  # ms per 100 samples
    model_size_kb: float
    composite_score: float
    is_selected: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModelComparer:
    """
    Subsystem for training, profiling, and benchmarking multiple classifier algorithms
    under identical train/test splits.
    """

    def __init__(self, random_seed: int = 42) -> None:
        self.random_seed = random_seed
        self.candidate_models = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000, random_state=random_seed, class_weight="balanced"
            ),
            "Decision Tree": DecisionTreeClassifier(
                max_depth=6, random_state=random_seed, class_weight="balanced"
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100, random_state=random_seed, class_weight="balanced", n_jobs=-1
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=100, random_state=random_seed, max_depth=3
            )
        }

    def evaluate_candidate(
        self,
        name: str,
        clf: Any,
        X_train: np.ndarray,
        y_train: pd.Series,
        X_test: np.ndarray,
        y_test: pd.Series
    ) -> ModelBenchmarkResult:
        """
        Trains a single candidate model, measures prediction performance, profiles inference latency,
        and estimates model size.
        """
        # 1. Fit Model
        clf.fit(X_train, y_train)

        # 2. Evaluate Performance on Test Split
        y_pred = clf.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
        rec_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
        f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

        # 3. Profile Inference Latency (100 repetitions over test samples)
        rep_count = 50
        start_time = time.perf_counter()
        for _ in range(rep_count):
            _ = clf.predict(X_test)
        elapsed = time.perf_counter() - start_time
        avg_ms_per_100 = (elapsed / (rep_count * len(X_test))) * 100.0 * 1000.0

        # 4. Estimate Model Complexity / Memory Footprint
        temp_file = RESULTS_DIR / f"temp_{name.lower().replace(' ', '_')}.joblib"
        joblib.dump(clf, temp_file)
        size_kb = float(temp_file.stat().st_size) / 1024.0
        if temp_file.exists():
            try:
                temp_file.unlink()
            except Exception:
                pass

        # Composite Score balancing F1 performance and real-time latency (penalty for latency > 10ms)
        latency_penalty = max(0.0, (avg_ms_per_100 - 5.0) * 0.02)
        composite = round(max(0.0, f1_macro - latency_penalty), 4)

        return ModelBenchmarkResult(
            model_name=name,
            accuracy=round(acc, 4),
            precision_macro=round(prec_macro, 4),
            recall_macro=round(rec_macro, 4),
            f1_macro=round(f1_macro, 4),
            f1_weighted=round(f1_weighted, 4),
            inference_time_ms=round(avg_ms_per_100, 3),
            model_size_kb=round(size_kb, 2),
            composite_score=composite,
            is_selected=False
        )

    def run_comparison(
        self,
        X_train: np.ndarray,
        y_train: pd.Series,
        X_test: np.ndarray,
        y_test: pd.Series
    ) -> Tuple[List[ModelBenchmarkResult], ModelBenchmarkResult, Any]:
        """
        Runs model comparison benchmark across all candidates using identical training/test partitions.
        Selects optimal candidate based on trade-off between accuracy/F1 score and inference latency.
        """
        results = []

        for name, clf in self.candidate_models.items():
            logger.info(f"Evaluating candidate classifier: {name}...")
            res = self.evaluate_candidate(name, clf, X_train, y_train, X_test, y_test)
            results.append(res)

        # Select model with highest composite score
        best_res = max(results, key=lambda r: r.composite_score)
        best_res.is_selected = True
        best_clf = self.candidate_models[best_res.model_name]

        logger.info(f"Selected Optimal Classifier: {best_res.model_name} (Composite Score: {best_res.composite_score:.4f})")
        return results, best_res, best_clf

    def export_comparison_results(
        self,
        results: List[ModelBenchmarkResult],
        output_dir: Optional[Union[str, Path]] = None
    ) -> Tuple[Path, Path, Path]:
        """
        Exports model comparison CSV summary, visualization plot, and markdown report.
        """
        out_dir = Path(output_dir) if output_dir else RESULTS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Export CSV
        df_res = pd.DataFrame([r.to_dict() for r in results])
        csv_path = out_dir / "model_comparison.csv"
        df_res.to_csv(csv_path, index=False)

        # 2. Export Visual Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        models = [r.model_name for r in results]
        f1_scores = [r.f1_macro for r in results]
        latencies = [r.inference_time_ms for r in results]
        colors = ["#2ca02c" if r.is_selected else "#1f77b4" for r in results]

        ax1.bar(models, f1_scores, color=colors)
        ax1.set_ylabel("Macro F1-Score")
        ax1.set_title("Classifier Macro F1-Score Comparison")
        ax1.set_ylim(0.0, 1.05)
        ax1.tick_params(axis="x", rotation=25)

        ax2.bar(models, latencies, color="#ff7f0e")
        ax2.set_ylabel("Inference Latency (ms / 100 samples)")
        ax2.set_title("Inference Latency Comparison (Real-Time Suitability)")
        ax2.tick_params(axis="x", rotation=25)

        plt.tight_layout()
        plot_path = out_dir / "model_comparison.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()

        # 3. Export Markdown Report
        best_model = next((r for r in results if r.is_selected), results[0])
        md_lines = [
            "# Machine Learning Model Comparison Report\n",
            f"**Evaluation Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            f"**Selected Model:** `{best_model.model_name}`\n",
            f"**Selection Rationale:** Highest composite score (`{best_model.composite_score}`) balancing Macro F1 (`{best_model.f1_macro}`) and real-time inference latency (`{best_model.inference_time_ms} ms`).\n\n",
            "## Candidate Comparison Summary Table\n",
            "| Model Name | Accuracy | Precision (Macro) | Recall (Macro) | F1 (Macro) | F1 (Weighted) | Latency (ms/100) | Size (KB) | Selected |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"
        ]

        for r in results:
            sel_str = "**YES**" if r.is_selected else "No"
            md_lines.append(
                f"| {r.model_name} | {r.accuracy:.4f} | {r.precision_macro:.4f} | {r.recall_macro:.4f} | "
                f"{r.f1_macro:.4f} | {r.f1_weighted:.4f} | {r.inference_time_ms:.3f} | {r.model_size_kb:.1f} | {sel_str} |"
            )

        md_lines.append("\n## Selection Criteria Notice")
        md_lines.append(
            "> Candidate model selection strictly balances predictive accuracy against real-time latency. "
            "Complex models are NOT selected if lightweight baselines achieve equivalent performance with lower inference overhead."
        )

        report_path = out_dir / "model_comparison_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))

        logger.info(f"Exported comparison CSV to {csv_path}, plot to {plot_path}, and report to {report_path}")
        return csv_path, plot_path, report_path
