# =============================================================================
# src/models/train_models.py
# Phase 5 — Machine Learning Training Pipeline
# Project: AI-Powered Customer Churn Prediction
# =============================================================================
#
# MODELS TRAINED:
#   1. Logistic Regression      — interpretable linear baseline
#   2. Decision Tree            — visual, rule-based, explainable
#   3. Random Forest            — ensemble, robust, handles non-linearity
#   4. Gradient Boosting        — sequential ensemble, typically best performer
#
# NOTE ON XGBOOST:
#   XGBoost requires a separate install (pip install xgboost).
#   It is included as an optional block — the pipeline runs fully without it.
#   Add it when your environment supports it.
#
# WHAT THIS MODULE DOES:
#   1.  Loads X_train, X_test, y_train, y_test from Phase 4
#   2.  Applies class-weight balancing (handles 78/22 imbalance)
#   3.  Trains all 4 models with cross-validation
#   4.  Evaluates: Accuracy, Precision, Recall, F1, ROC-AUC
#   5.  Plots confusion matrices, ROC curves, feature importances
#   6.  Selects the best model (by ROC-AUC + Recall)
#   7.  Saves all models and a full comparison report
#
# KEY DESIGN DECISIONS:
#   • We optimise for RECALL (catching churners > avoiding false alarms)
#   • class_weight='balanced' on all models — no SMOTE needed for tree models
#   • StratifiedKFold(5) for CV — preserves class balance in each fold
#   • threshold=0.40 (not 0.50) — lowers bar to flag a churner, boosts recall
# =============================================================================

import sys
import warnings
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, auc,
)

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    DATA_PROCESSED, MODELS_DIR, EVAL_DIR, FIGURES_DIR,
    LOG_FILE, RANDOM_STATE, TARGET_COLUMN,
)
from src.utils.logger import get_logger

logger = get_logger(__name__, str(LOG_FILE))

# =============================================================================
# CONSTANTS
# =============================================================================

CHURN_THRESHOLD = 0.40   # Lower than 0.50 → higher recall, fewer missed churners
CV_FOLDS        = 5      # StratifiedKFold folds

CORP_BLUE  = "#2C3E50"
ACC_RED    = "#E74C3C"
ACC_GREEN  = "#2ECC71"
ACC_ORANGE = "#E67E22"
ACC_PURPLE = "#8E44AD"

MODEL_COLORS = {
    "Logistic Regression"   : "#3498DB",
    "Decision Tree"         : "#E67E22",
    "Random Forest"         : "#2ECC71",
    "Gradient Boosting"     : "#E74C3C",
}


# =============================================================================
# STEP 1 — LOAD DATA
# =============================================================================

def load_train_test_data() -> Tuple[pd.DataFrame, pd.DataFrame,
                                    pd.Series, pd.Series]:
    """
    Loads the Phase 4 train/test splits from disk.
    These are already scaled and feature-engineered.
    """
    logger.info("STEP 1 — Loading train/test data")

    paths = {
        "X_train": DATA_PROCESSED / "X_train.csv",
        "X_test" : DATA_PROCESSED / "X_test.csv",
        "y_train": DATA_PROCESSED / "y_train.csv",
        "y_test" : DATA_PROCESSED / "y_test.csv",
    }

    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{name} not found at {path}.\n"
                "Run Phase 4 first: python src/features/feature_engineering.py"
            )

    X_train = pd.read_csv(paths["X_train"])
    X_test  = pd.read_csv(paths["X_test"])
    y_train = pd.read_csv(paths["y_train"]).squeeze()
    y_test  = pd.read_csv(paths["y_test"]).squeeze()

    logger.info(f"X_train: {X_train.shape} | Churn rate: {y_train.mean()*100:.1f}%")
    logger.info(f"X_test : {X_test.shape}  | Churn rate: {y_test.mean()*100:.1f}%")

    return X_train, X_test, y_train, y_test


# =============================================================================
# STEP 2 — DEFINE MODELS
# =============================================================================

def get_model_definitions() -> Dict[str, Any]:
    """
    Returns a dict of model name → unfitted sklearn estimator.

    Hyperparameters are set here as production-quality defaults.
    Full hyperparameter tuning (GridSearchCV) is done in the notebook
    for the selected best model.

    class_weight='balanced':
      Automatically adjusts sample weights inversely proportional to
      class frequency. This is equivalent to oversampling the minority
      class (churners) without actually duplicating samples.
      Formula: weight_class = n_samples / (n_classes * n_samples_class)
    """
    logger.info("STEP 2 — Defining model configurations")

    models = {

        # ── Logistic Regression ───────────────────────────────────────────────
        # Why: Best interpretable baseline. Coefficients show feature direction.
        # C=1.0: moderate regularisation (prevents overfitting on 30 features)
        # max_iter=1000: enough for convergence on scaled data
        "Logistic Regression": LogisticRegression(
            C            = 1.0,
            class_weight = "balanced",
            solver       = "lbfgs",
            max_iter     = 1000,
            random_state = RANDOM_STATE,
        ),

        # ── Decision Tree ─────────────────────────────────────────────────────
        # Why: Fully interpretable — produces human-readable rules.
        # max_depth=6: prevents overfitting while capturing key splits
        # min_samples_split=50: avoids splits on tiny leaf nodes (noise)
        "Decision Tree": DecisionTreeClassifier(
            max_depth         = 6,
            min_samples_split = 50,
            min_samples_leaf  = 20,
            class_weight      = "balanced",
            random_state      = RANDOM_STATE,
        ),

        # ── Random Forest ─────────────────────────────────────────────────────
        # Why: Robust ensemble. Averages 200 trees → low variance, high stability.
        # n_estimators=200: more trees → lower variance (diminishing returns after ~200)
        # max_features='sqrt': each tree sees sqrt(30)≈5 features → diversity
        "Random Forest": RandomForestClassifier(
            n_estimators  = 200,
            max_depth     = 10,
            max_features  = "sqrt",
            min_samples_leaf = 10,
            class_weight  = "balanced",
            n_jobs        = -1,
            random_state  = RANDOM_STATE,
        ),

        # ── Gradient Boosting ─────────────────────────────────────────────────
        # Why: Sequentially corrects errors of prior trees → typically best AUC.
        # learning_rate=0.05: small steps → more stable convergence
        # n_estimators=200: more boosting rounds with small learning rate
        # subsample=0.8: stochastic gradient boosting — reduces overfitting
        # Note: no class_weight param; use scale_pos_weight equivalent via
        #       sample_weight in fit() — handled in train_single_model()
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators  = 200,
            learning_rate = 0.05,
            max_depth     = 4,
            subsample     = 0.8,
            max_features  = "sqrt",
            random_state  = RANDOM_STATE,
        ),
    }

    logger.info(f"Models defined: {list(models.keys())}")
    return models


# =============================================================================
# STEP 3 — COMPUTE SAMPLE WEIGHTS (for Gradient Boosting)
# =============================================================================

def get_sample_weights(y: pd.Series) -> np.ndarray:
    """
    Computes per-sample weights to handle class imbalance for models
    that do not support class_weight='balanced' directly (GradientBoosting).

    Logic:
      Weight for each sample = total_samples / (n_classes * class_count)
      This is exactly what class_weight='balanced' does internally.

    Result: churners (~22%) get weight ≈ 3.6×  non-churners (78%).
    """
    n_samples = len(y)
    n_classes = 2
    class_counts = y.value_counts()

    weights = y.map({
        cls: n_samples / (n_classes * count)
        for cls, count in class_counts.items()
    }).values

    logger.info(
        f"Sample weights: No-Churn={weights[y == 0][0]:.3f}, "
        f"Churn={weights[y == 1][0]:.3f}"
    )
    return weights


# =============================================================================
# STEP 4 — TRAIN A SINGLE MODEL WITH CROSS-VALIDATION
# =============================================================================

def train_single_model(
    name    : str,
    model   : Any,
    X_train : pd.DataFrame,
    y_train : pd.Series,
) -> Tuple[Any, dict]:
    """
    Trains one model with StratifiedKFold cross-validation.

    Cross-validation gives a more reliable performance estimate than a
    single train/val split by averaging over 5 different splits.

    Returns:
        model    : fitted on full training set
        cv_scores: dict of mean CV scores
    """
    logger.info(f"Training: {name}")

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # Compute sample weights for GradientBoosting
    sample_weights = (
        get_sample_weights(y_train)
        if name == "Gradient Boosting"
        else None
    )

    # ── Cross-validation ──────────────────────────────────────────────────────
    fit_params = {}
    if sample_weights is not None:
        fit_params["sample_weight"] = sample_weights

    cv_results = cross_validate(
        model,
        X_train, y_train,
        cv            = skf,
        scoring       = ["accuracy", "f1", "recall", "precision", "roc_auc"],
        params        = fit_params,
        return_train_score = False,
        n_jobs        = -1,
    )

    cv_scores = {
        metric: round(float(scores.mean()), 4)
        for metric, scores in cv_results.items()
        if metric.startswith("test_")
    }

    logger.info(f"  CV ROC-AUC : {cv_scores['test_roc_auc']:.4f}")
    logger.info(f"  CV Recall  : {cv_scores['test_recall']:.4f}")
    logger.info(f"  CV F1      : {cv_scores['test_f1']:.4f}")

    # ── Final fit on full training set ────────────────────────────────────────
    if sample_weights is not None:
        model.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        model.fit(X_train, y_train)

    return model, cv_scores


# =============================================================================
# STEP 5 — EVALUATE ON TEST SET
# =============================================================================

def evaluate_on_test(
    name      : str,
    model     : Any,
    X_test    : pd.DataFrame,
    y_test    : pd.Series,
    threshold : float = CHURN_THRESHOLD,
) -> dict:
    """
    Evaluates a fitted model on the held-out test set.

    Uses a custom threshold (0.40) instead of the default 0.50:
      At threshold=0.50: the model only flags a customer as churner if
        it's 50%+ confident → misses borderline churners.
      At threshold=0.40: flags customer as churner if 40%+ confident
        → catches more real churners (higher recall) at the cost of
        slightly more false alarms (lower precision).

    Business logic: a false alarm costs one unnecessary retention offer
    (low cost). A missed churner costs the full CLV of that customer
    (high cost). So we deliberately accept more false alarms.

    Returns:
        dict with all metrics + predictions + probabilities
    """
    y_prob = model.predict_proba(X_test)[:, 1]     # Probability of class 1 (churn)
    y_pred = (y_prob >= threshold).astype(int)      # Apply custom threshold

    metrics = {
        "model"      : name,
        "threshold"  : threshold,
        "accuracy"   : round(accuracy_score(y_test, y_pred), 4),
        "precision"  : round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall"     : round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1"         : round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc"    : round(roc_auc_score(y_test, y_prob), 4),
        "y_prob"     : y_prob,
        "y_pred"     : y_pred,
    }

    logger.info(
        f"{name}: Acc={metrics['accuracy']:.3f} | "
        f"Prec={metrics['precision']:.3f} | "
        f"Rec={metrics['recall']:.3f} | "
        f"F1={metrics['f1']:.3f} | "
        f"AUC={metrics['roc_auc']:.3f}"
    )

    return metrics


# =============================================================================
# STEP 6 — VISUALISATIONS
# =============================================================================

def plot_confusion_matrices(
    results  : list,
    y_test   : pd.Series,
) -> None:
    """
    Plots a 2×2 grid of confusion matrices — one per model.

    Confusion Matrix layout:
                   Predicted No  Predicted Yes
    Actual No   [  TN          |  FP           ]
    Actual Yes  [  FN          |  TP           ]

    For churn: TP = correctly caught churners (most important!)
               FN = missed churners (worst outcome — "false negatives")
               FP = false alarms (unnecessary retention call — low cost)
               TN = correctly identified loyal customers
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("Confusion Matrices — All Models (Test Set)",
                 fontsize=14, fontweight="bold", color=CORP_BLUE)
    axes = axes.flatten()

    for i, result in enumerate(results):
        ax   = axes[i]
        cm   = confusion_matrix(y_test, result["y_pred"])
        name = result["model"]

        # Normalised confusion matrix (row-wise percentages)
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

        sns.heatmap(
            cm_pct,
            annot      = False,
            cmap       = "Blues",
            ax         = ax,
            cbar       = False,
            linewidths = 1,
            linecolor  = "white",
        )

        # Manual annotations: show both count and percentage
        labels = [["TN", "FP"], ["FN", "TP"]]
        label_colors = [["white", "white"], ["#E74C3C", "#2ECC71"]]
        for row in range(2):
            for col in range(2):
                count = cm[row, col]
                pct   = cm_pct[row, col]
                lbl   = labels[row][col]
                color = label_colors[row][col]
                ax.text(
                    col + 0.5, row + 0.38,
                    f"{lbl}\n{count:,}",
                    ha="center", va="center",
                    fontsize=13, fontweight="bold",
                    color=color if lbl in ["FN", "TP"] else "white"
                )
                ax.text(
                    col + 0.5, row + 0.68,
                    f"({pct:.1f}%)",
                    ha="center", va="center",
                    fontsize=10, color="white"
                )

        ax.set_title(
            f"{name}\nAUC={result['roc_auc']:.3f} | "
            f"Recall={result['recall']:.3f} | "
            f"F1={result['f1']:.3f}",
            fontsize=10, fontweight="bold"
        )
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)
        ax.set_xticklabels(["No Churn", "Churned"])
        ax.set_yticklabels(["No Churn", "Churned"], rotation=0)

    plt.tight_layout()
    out = FIGURES_DIR / "confusion_matrices_all_models.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"Confusion matrices saved → {out}")


def plot_roc_curves(
    results : list,
    y_test  : pd.Series,
) -> None:
    """
    Plots ROC curves for all models on one chart.

    ROC Curve:
      X-axis: False Positive Rate (FPR) = FP / (FP + TN)
      Y-axis: True Positive Rate (TPR) = TP / (TP + FN) = Recall

    AUC (Area Under the Curve):
      1.0 = perfect model
      0.5 = random guessing (diagonal line)
      > 0.85 = good for churn prediction

    Each point on the curve represents a different threshold.
    The curve shows the full recall-FPR tradeoff across all thresholds.
    """
    fig, ax = plt.subplots(figsize=(9, 7))

    for result in results:
        fpr, tpr, _ = roc_curve(y_test, result["y_prob"])
        roc_auc      = auc(fpr, tpr)
        name         = result["model"]
        color        = MODEL_COLORS.get(name, "#7F8C8D")

        ax.plot(fpr, tpr, linewidth=2.5, color=color,
                label=f"{name}  (AUC = {roc_auc:.3f})")

    # Reference lines
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Random Classifier (AUC = 0.500)")
    ax.axhline(0.80, color="grey", linestyle=":", linewidth=1,
               label="Recall target (0.80)")

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
    ax.set_title("ROC Curves — All Models\n(Test Set)",
                 fontsize=13, fontweight="bold", color=CORP_BLUE)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = FIGURES_DIR / "roc_curves_all_models.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"ROC curves saved → {out}")


def plot_metrics_comparison(results: list, cv_results: dict) -> None:
    """
    Side-by-side bar charts comparing all models across 5 metrics.
    Shows both CV scores and test-set scores for transparency.
    """
    metric_keys   = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]

    model_names  = [r["model"] for r in results]
    test_scores  = {m: [r[m] for r in results] for m in metric_keys}
    cv_score_map = {
        m: [cv_results[n].get(f"test_{m}", 0) for n in model_names]
        for m in metric_keys
    }

    fig, axes = plt.subplots(1, 5, figsize=(20, 6))
    fig.suptitle("Model Performance Comparison — Test Set vs Cross-Validation",
                 fontsize=13, fontweight="bold", color=CORP_BLUE)

    bar_width = 0.35
    x         = np.arange(len(model_names))
    colors    = [MODEL_COLORS.get(n, "#7F8C8D") for n in model_names]

    for ax, metric, label in zip(axes, metric_keys, metric_labels):
        # Test bars
        bars1 = ax.bar(x - bar_width / 2, test_scores[metric],
                       bar_width, color=colors,
                       alpha=0.9, edgecolor="white", label="Test")
        # CV bars (hatched)
        bars2 = ax.bar(x + bar_width / 2, cv_score_map[metric],
                       bar_width, color=colors,
                       alpha=0.45, edgecolor="white", hatch="//", label="CV")

        ax.set_title(label, fontweight="bold", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [n.replace(" ", "\n") for n in model_names],
            fontsize=8
        )
        ax.set_ylim(0, 1.12)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v:.2f}")
        )
        ax.axhline(0.85 if metric == "roc_auc" else 0.80,
                   color="black", linestyle="--",
                   linewidth=0.8, alpha=0.5)

        # Value labels on test bars
        for bar in bars1:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2,
                    h + 0.01, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=7, fontweight="bold")

        if ax == axes[0]:
            ax.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    out = FIGURES_DIR / "model_comparison_metrics.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"Metrics comparison saved → {out}")


def plot_feature_importance(
    models  : dict,
    feature_names : list,
    top_n   : int = 20,
) -> None:
    """
    Plots feature importances for tree-based models.

    Random Forest / GradientBoosting: .feature_importances_ attribute
    Logistic Regression: abs(coefficients) as proxy for importance

    Why feature importance matters:
      1. Validates that the model is learning business-meaningful patterns
      2. Guides future feature engineering efforts
      3. Required for model explanation to stakeholders
      4. Identifies features that could be dropped (low importance)
    """
    tree_models = {
        name: m for name, m in models.items()
        if hasattr(m, "feature_importances_")
    }

    n_models = len(tree_models)
    if n_models == 0:
        return

    fig, axes = plt.subplots(1, n_models, figsize=(10 * n_models, 8))
    if n_models == 1:
        axes = [axes]

    fig.suptitle("Feature Importances — Tree-Based Models",
                 fontsize=14, fontweight="bold", color=CORP_BLUE)

    for ax, (name, model) in zip(axes, tree_models.items()):
        importances = pd.Series(
            model.feature_importances_, index=feature_names
        ).sort_values(ascending=True).tail(top_n)

        color = MODEL_COLORS.get(name, "#3498DB")
        bars  = ax.barh(importances.index, importances.values,
                        color=color, alpha=0.85, edgecolor="white")

        # Annotate values
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.001, bar.get_y() + bar.get_height() / 2,
                    f"{w:.4f}", va="center", fontsize=8)

        ax.set_title(f"{name}\n(Top {top_n} Features)",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Feature Importance (Gini / Impurity Reduction)")

    plt.tight_layout()
    out = FIGURES_DIR / "feature_importances.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"Feature importances saved → {out}")


def plot_threshold_analysis(
    best_model  : Any,
    X_test      : pd.DataFrame,
    y_test      : pd.Series,
    model_name  : str,
) -> None:
    """
    Shows how Precision, Recall, and F1 change across different
    classification thresholds for the best model.

    This is critical for business decisions:
      - At 0.30: Very high recall (catch most churners) but many false alarms
      - At 0.50: Balanced, but may miss too many real churners
      - Optimal: The threshold where F1 is maximised (or where business
                 decides the recall/precision tradeoff is acceptable)
    """
    y_prob = best_model.predict_proba(X_test)[:, 1]

    thresholds  = np.linspace(0.10, 0.90, 81)
    precisions  = []
    recalls     = []
    f1_scores   = []

    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        precisions.append(precision_score(y_test, y_pred, zero_division=0))
        recalls.append(recall_score(y_test, y_pred, zero_division=0))
        f1_scores.append(f1_score(y_test, y_pred, zero_division=0))

    best_f1_idx   = int(np.argmax(f1_scores))
    best_threshold = thresholds[best_f1_idx]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(thresholds, precisions, color="#3498DB", linewidth=2.2, label="Precision")
    ax.plot(thresholds, recalls,    color=ACC_RED,   linewidth=2.2, label="Recall")
    ax.plot(thresholds, f1_scores,  color=ACC_GREEN, linewidth=2.2, label="F1 Score")

    ax.axvline(0.40, color="purple", linestyle="--",
               linewidth=1.5, label="Our threshold (0.40)")
    ax.axvline(best_threshold, color="black", linestyle=":",
               linewidth=1.5, label=f"Best F1 threshold ({best_threshold:.2f})")
    ax.axhline(0.80, color="grey", linestyle=":", linewidth=1, alpha=0.7)

    ax.set_xlabel("Classification Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(
        f"Threshold Analysis — {model_name}\n"
        f"Best F1 at threshold={best_threshold:.2f}",
        fontsize=13, fontweight="bold", color=CORP_BLUE
    )
    ax.legend(loc="center right", fontsize=10)
    ax.set_xlim(0.10, 0.90)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = FIGURES_DIR / "threshold_analysis.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"Threshold analysis saved → {out} | Best F1 threshold: {best_threshold:.2f}")


# =============================================================================
# STEP 7 — SELECT BEST MODEL
# =============================================================================

def select_best_model(
    results    : list,
    cv_results : dict,
) -> Tuple[str, dict]:
    """
    Selects the best model using a weighted scoring system:

      Score = 0.40 × ROC-AUC + 0.40 × Recall + 0.20 × F1

    WHY THIS WEIGHTING:
      ROC-AUC (40%): Measures overall discrimination — how well the model
        separates churners from non-churners across ALL thresholds.
      Recall (40%): Our primary business metric — we must catch churners.
        Missing a churner costs the company the full CLV of that customer.
      F1 (20%): Harmonic mean of precision and recall — ensures we're not
        so aggressive that we flag 90% of customers as churners.

    Returns:
        (best_model_name, best_result_dict)
    """
    logger.info("STEP 7 — Selecting best model")

    scored = []
    for r in results:
        score = (
            0.40 * r["roc_auc"] +
            0.40 * r["recall"]  +
            0.20 * r["f1"]
        )
        scored.append((score, r["model"], r))
        logger.info(f"  {r['model']}: selection_score={score:.4f}")

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_name, best_result = scored[0]

    logger.info(f"\n★ BEST MODEL: {best_name} (selection_score={best_score:.4f})")
    return best_name, best_result


# =============================================================================
# STEP 8 — SAVE MODELS AND REPORT
# =============================================================================

def save_models_and_report(
    models     : dict,
    results    : list,
    cv_results : dict,
    best_name  : str,
) -> pd.DataFrame:
    """
    Saves all trained models and a comparison CSV report.

    Files:
      models/saved/<model_name>.pkl        — each fitted model
      models/saved/best_model.pkl          — the selected best model
      models/evaluation/model_comparison.csv — all metrics in one table
    """
    logger.info("STEP 8 — Saving models and evaluation report")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # ── Save each model ───────────────────────────────────────────────────────
    name_to_file = {
        "Logistic Regression" : "logistic_regression.pkl",
        "Decision Tree"       : "decision_tree.pkl",
        "Random Forest"       : "random_forest.pkl",
        "Gradient Boosting"   : "gradient_boosting.pkl",
    }

    for name, model in models.items():
        path = MODELS_DIR / name_to_file.get(name, f"{name.lower().replace(' ', '_')}.pkl")
        joblib.dump(model, path)
        logger.info(f"Saved → {path}")

    # Save best model separately for easy loading in Phase 6
    best_model = models[best_name]
    joblib.dump(best_model, MODELS_DIR / "best_model.pkl")
    joblib.dump(best_name,  MODELS_DIR / "best_model_name.pkl")
    logger.info(f"Best model ({best_name}) saved → models/saved/best_model.pkl")

    # ── Build comparison DataFrame ────────────────────────────────────────────
    rows = []
    for r in results:
        name = r["model"]
        cv   = cv_results.get(name, {})
        rows.append({
            "Model"          : name,
            "Test Accuracy"  : r["accuracy"],
            "Test Precision" : r["precision"],
            "Test Recall"    : r["recall"],
            "Test F1"        : r["f1"],
            "Test ROC-AUC"   : r["roc_auc"],
            "CV Accuracy"    : cv.get("test_accuracy", None),
            "CV F1"          : cv.get("test_f1", None),
            "CV Recall"      : cv.get("test_recall", None),
            "CV ROC-AUC"     : cv.get("test_roc_auc", None),
            "Threshold"      : r["threshold"],
            "Best Model"     : "★" if name == best_name else "",
        })

    comparison_df = pd.DataFrame(rows).set_index("Model")
    csv_path      = EVAL_DIR / "model_comparison.csv"
    comparison_df.to_csv(csv_path)
    logger.info(f"Comparison report saved → {csv_path}")

    return comparison_df


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_training_pipeline() -> dict:
    """
    Orchestrates the full ML training pipeline.

    Returns a dict with all trained models, metrics, and best model info.
    """
    logger.info("=" * 60)
    logger.info("  ML TRAINING PIPELINE — START")
    logger.info("=" * 60)

    # Step 1: Load data
    X_train, X_test, y_train, y_test = load_train_test_data()
    feature_names = list(X_train.columns)

    # Step 2: Define models
    model_defs = get_model_definitions()

    # Steps 3-4: Train all models with CV
    trained_models = {}
    cv_results     = {}
    test_results   = []

    for name, model in model_defs.items():
        fitted_model, cv_scores = train_single_model(
            name, model, X_train, y_train
        )
        trained_models[name] = fitted_model
        cv_results[name]     = cv_scores

        # Step 5: Evaluate on test set
        test_result = evaluate_on_test(name, fitted_model, X_test, y_test)
        test_results.append(test_result)

    # Step 6: Visualisations
    plot_confusion_matrices(test_results, y_test)
    plot_roc_curves(test_results, y_test)
    plot_metrics_comparison(test_results, cv_results)
    plot_feature_importance(trained_models, feature_names)

    # Step 7: Select best model
    best_name, best_result = select_best_model(test_results, cv_results)
    plot_threshold_analysis(trained_models[best_name], X_test, y_test, best_name)

    # Step 8: Save
    comparison_df = save_models_and_report(
        trained_models, test_results, cv_results, best_name
    )

    # ── Print final summary ───────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("  MODEL TRAINING COMPLETE — RESULTS SUMMARY")
    print("═" * 65)
    display_cols = ["Test Accuracy", "Test Precision",
                    "Test Recall", "Test F1", "Test ROC-AUC", "Best Model"]
    print(comparison_df[display_cols].to_string())
    print("═" * 65)
    print(f"\n  ★  Best Model : {best_name}")
    print(f"     ROC-AUC    : {best_result['roc_auc']:.4f}")
    print(f"     Recall     : {best_result['recall']:.4f}")
    print(f"     F1 Score   : {best_result['f1']:.4f}")
    print(f"     Threshold  : {best_result['threshold']}")
    print("═" * 65 + "\n")

    logger.info("=" * 60)
    logger.info("  ML TRAINING PIPELINE — COMPLETE")
    logger.info(f"  Best model: {best_name}")
    logger.info("=" * 60)

    return {
        "trained_models"  : trained_models,
        "cv_results"      : cv_results,
        "test_results"    : test_results,
        "comparison_df"   : comparison_df,
        "best_name"       : best_name,
        "best_result"     : best_result,
        "feature_names"   : feature_names,
        "X_test"          : X_test,
        "y_test"          : y_test,
    }


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    pipeline_output = run_training_pipeline()
