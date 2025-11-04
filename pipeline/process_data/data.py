
"""
Tornado Q baseline: data loading, exploration, preprocessing, modeling, and evaluation
refactored from a Jupyter notebook into importable functions.

Usage example (see bottom of file or README-style docstring below).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

try:
    from xgboost import XGBClassifier  # type: ignore
except Exception:  # pragma: no cover
    XGBClassifier = None  # allows importing without xgboost installed

try:
    from imblearn.over_sampling import SMOTE  # type: ignore
except Exception:  # pragma: no cover
    SMOTE = None  # allows importing without imblearn installed

# wrapper for pandas.load_csv
def load_csv(filename):
    df = pd.load_csv(filename)
    return df
# wrapper for df.to_csv
def save_csv(df, filename, index = False):
    df.to_csv(filename, index = index)

# =============================================================================
# Config
# =============================================================================
@dataclass
class ScalingConfig:
    kind: str = "minmax"  # "minmax" or "standard"
    feature_range: Tuple[float, float] = (0.0, 1.0)  # only used for minmax


@dataclass
class PCAConfig:
    enabled: bool = True
    n_components: Optional[int] = None  # None = full PCA


@dataclass
class CVConfig:
    folds: int = 5
    shuffle: bool = True
    random_state: int = 42


# =============================================================================
# Data IO & Exploration
# =============================================================================

def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_train = pd.read_excel(train_path)
    df_test = pd.read_excel(test_path)
    print(f"✓ Training data loaded: {df_train.shape[0]} rows, {df_train.shape[1]} columns")
    print(f"✓ Test data loaded: {df_test.shape[0]} rows, {df_test.shape[1]} columns")
    return df_train, df_test


def print_basic_info(df_train: pd.DataFrame, df_test: pd.DataFrame) -> None:
    print(f"\nColumn names: {df_train.columns.tolist()}")

    print("\n" + "-" * 80)
    print("TRAINING DATA INFO:")
    print("-" * 80)
    print(df_train.info())

    print("\n" + "-" * 80)
    print("STATISTICAL SUMMARY (TRAINING):")
    print("-" * 80)
    print(df_train.describe())

    print("\n" + "-" * 80)
    print("MISSING VALUES:")
    print("-" * 80)
    missing_train = df_train.isnull().sum()
    missing_train = missing_train[missing_train > 0]
    if len(missing_train) > 0:
        print("Training set:")
        print(missing_train)
    else:
        print("Training set: No missing values")

    missing_test = df_test.isnull().sum()
    missing_test = missing_test[missing_test > 0]
    if len(missing_test) > 0:
        print("\nTest set:")
        print(missing_test)
    else:
        print("\nTest set: No missing values")


def plot_target_distributions(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    savepath: Optional[str] = None,
) -> None:
    """Visualize binary & multi-class distributions for train/test/combined."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Binary - Training
    axes[0, 0].bar([0, 1], df_train["ef_binary"].value_counts().sort_index(),
                   color=["steelblue", "coral"])  # type: ignore
    axes[0, 0].set_title("Binary Target - TRAINING\n(PRIMARY)", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("EF Binary")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_xticks([0, 1])
    axes[0, 0].set_xticklabels(["Weak", "Strong"])  # type: ignore

    # Binary - Test
    axes[0, 1].bar([0, 1], df_test["ef_binary"].value_counts().sort_index(),
                   color=["steelblue", "coral"])  # type: ignore
    axes[0, 1].set_title("Binary Target - TEST", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("EF Binary")
    axes[0, 1].set_xticks([0, 1])
    axes[0, 1].set_xticklabels(["Weak", "Strong"])  # type: ignore

    # Binary - Combined pie
    combined_binary = pd.concat([df_train["ef_binary"], df_test["ef_binary"]])
    axes[0, 2].pie(
        combined_binary.value_counts().sort_index(),
        labels=["Weak (EF0-1)", "Strong (EF2+)"],
        autopct="%1.1f%%",
        colors=["steelblue", "coral"],
    )
    axes[0, 2].set_title("Binary - COMBINED", fontsize=12, fontweight="bold")

    # Multi-class - Training
    axes[1, 0].bar(range(4), df_train["ef_class"].value_counts().sort_index(), color="steelblue")  # type: ignore
    axes[1, 0].set_title("Multi-Class - TRAINING\n(ADVANCED)", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("EF Class")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_xticks(range(4))
    axes[1, 0].set_xticklabels([f"EF-{i}" for i in range(4)])

    # Multi-class - Test
    axes[1, 1].bar(range(4), df_test["ef_class"].value_counts().sort_index(), color="steelblue")  # type: ignore
    axes[1, 1].set_title("Multi-Class - TEST", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("EF Class")
    axes[1, 1].set_xticks(range(4))
    axes[1, 1].set_xticklabels([f"EF-{i}" for i in range(4)])

    # Multi-class - Combined pie
    combined_class = pd.concat([df_train["ef_class"], df_test["ef_class"]])
    axes[1, 2].pie(
        combined_class.value_counts().sort_index(),
        labels=[f"EF-{i}" for i in range(4)],
        autopct="%1.1f%%",
    )
    axes[1, 2].set_title("Multi-Class - COMBINED", fontsize=12, fontweight="bold")

    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    plt.show()


# =============================================================================
# Preprocessing
# =============================================================================

def separate_features_targets(df_train: pd.DataFrame, df_test: pd.DataFrame):
    X_train = df_train.drop(["ef_class", "ef_binary"], axis=1, errors="ignore")
    X_test = df_test.drop(["ef_class", "ef_binary"], axis=1, errors="ignore")
    y_train_binary = df_train["ef_binary"]
    y_test_binary = df_test["ef_binary"]
    y_train_class = df_train["ef_class"]
    y_test_class = df_test["ef_class"]
    return X_train, X_test, y_train_binary, y_test_binary, y_train_class, y_test_class


def impute_and_scale(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    scaling: ScalingConfig = ScalingConfig(),
) -> Tuple[pd.DataFrame, pd.DataFrame, SimpleImputer, object]:
    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

    if scaling.kind == "standard":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler(feature_range=scaling.feature_range)

    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imp), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_imp), columns=X_test.columns)
    return X_train_scaled, X_test_scaled, imputer, scaler


def export_scaled_data(
    X_train_scaled: pd.DataFrame,
    X_test_scaled: pd.DataFrame,
    train_out: str = "X_train_scaled.csv",
    test_out: str = "X_test_scaled.csv",
) -> None:
    print("\n" + "=" * 80)
    print("STEP 5A: EXPORTING NORMALIZED DATASET")
    print("=" * 80)
    try:
        X_train_scaled.to_csv(train_out, index=False)
        print(f"✓ Normalized training data exported to: {train_out}")
    except Exception as e:  # pragma: no cover
        print(f"❌ Error exporting training data: {e}")

    try:
        X_test_scaled.to_csv(test_out, index=False)
        print(f"✓ Normalized test data exported to: {test_out}")
    except Exception as e:  # pragma: no cover
        print(f"❌ Error exporting test data: {e}")


# =============================================================================
# Correlation & PCA
# =============================================================================

def save_correlation_matrix(X_train_scaled: pd.DataFrame, out_csv: str = "correlation_matrix.csv") -> pd.DataFrame:
    print("\n" + "=" * 80)
    print("STEP 4: CORRELATION ANALYSIS")
    print("=" * 80)
    corr = X_train_scaled.corr()
    # corr.to_csv(out_csv)
    return corr


def run_pca(
    X_train_scaled: pd.DataFrame,
    y_train_binary: pd.Series,
    pca_cfg: PCAConfig = PCAConfig(),
    scree_out: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    print("\n" + "=" * 80)
    print("STEP 5: PRINCIPAL COMPONENT ANALYSIS (PCA)")
    print("=" * 80)

    pca = PCA(n_components=pca_cfg.n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    variance_explained = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(variance_explained)

    print("\nVariance Explained by Components:")
    for i, (var, cum_var) in enumerate(zip(variance_explained, cumulative_variance)):
        print(f"PC{i+1}: {var*100:.2f}% (Cumulative: {cum_var*100:.2f}%)")

    # Plots: scree + 2D projection
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].bar(range(1, len(variance_explained) + 1), variance_explained, alpha=0.7, label="Individual")
    axes[0].plot(range(1, len(variance_explained) + 1), cumulative_variance, "ro-", linewidth=2, label="Cumulative")
    axes[0].set_xlabel("Principal Component", fontsize=12)
    axes[0].set_ylabel("Variance Explained", fontsize=12)
    axes[0].set_title("Scree Plot", fontsize=14, fontweight="bold")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].axhline(y=0.95, color="red", linestyle="--", alpha=0.5)

    scatter = axes[1].scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train_binary, cmap="viridis", alpha=0.6, s=50)
    axes[1].set_xlabel(f"PC1 ({variance_explained[0]*100:.1f}%)", fontsize=12)
    axes[1].set_ylabel(f"PC2 ({variance_explained[1]*100:.1f}%)", fontsize=12)
    axes[1].set_title("2D PCA Projection", fontsize=14, fontweight="bold")
    axes[1].grid(alpha=0.3)
    plt.colorbar(scatter, ax=axes[1], label="EF Binary")

    plt.tight_layout()
    if scree_out:
        fig.savefig(scree_out, bbox_inches="tight")
    plt.show()

    return X_train_pca, variance_explained, cumulative_variance


# =============================================================================
# Modeling: Binary
# =============================================================================

def _get_binary_models(random_state: int = 42) -> Dict[str, object]:
    models: Dict[str, object] = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=random_state, class_weight="balanced", solver="lbfgs"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1, min_samples_split=10, min_samples_leaf=4, random_state=random_state
        ),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=random_state,
            eval_metric="logloss",
            use_label_encoder=False,
        )
    return models


def smote_resample_binary(X_train_scaled: pd.DataFrame, y_train_binary: pd.Series, random_state: int = 42):
    if SMOTE is None:
        raise ImportError("imblearn is required for SMOTE; install with `pip install imbalanced-learn`. ")
    print("\nApplying SMOTE for binary classification...")
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train_scaled, y_train_binary)
    print(f"✓ Training samples: {len(y_train_binary)} → {len(y_res)}")
    print(f"  Class 0: {(y_res == 0).sum()}")
    print(f"  Class 1: {(y_res == 1).sum()}")
    return X_res, y_res


def train_binary_models(
    X_train_balanced: pd.DataFrame,
    y_train_balanced: pd.Series,
    X_train_scaled: pd.DataFrame,
    X_test_scaled: pd.DataFrame,
    y_train_binary: pd.Series,
    y_test_binary: pd.Series,
    cv_cfg: CVConfig = CVConfig(),
) -> Tuple[Dict[str, dict], Dict[str, object]]:
    print("\n" + "-" * 80)
    print("TRAINING BINARY CLASSIFICATION MODELS:")
    print("-" * 80)

    models = _get_binary_models(random_state=cv_cfg.random_state)
    results_binary: Dict[str, dict] = {}
    trained_models: Dict[str, object] = {}

    for name, model in models.items():
        print(f"\nTraining: {name}")
        cv = StratifiedKFold(n_splits=cv_cfg.folds, shuffle=cv_cfg.shuffle, random_state=cv_cfg.random_state)
        cv_scores = cross_val_score(model, X_train_balanced, y_train_balanced, cv=cv, scoring="roc_auc", n_jobs=-1)

        # Fit on resampled data
        model.fit(X_train_balanced, y_train_balanced)

        # Predictions on original scaled splits
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)

        # Probabilities/decision
        if hasattr(model, "predict_proba"):
            y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
            y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_train_proba = model.decision_function(X_train_scaled)
            y_test_proba = model.decision_function(X_test_scaled)

        train_accuracy = accuracy_score(y_train_binary, y_train_pred)
        test_accuracy = accuracy_score(y_test_binary, y_test_pred)
        train_roc_auc = roc_auc_score(y_train_binary, y_train_proba)
        test_roc_auc = roc_auc_score(y_test_binary, y_test_proba)

        results_binary[name] = {
            "cv_mean": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "train_accuracy": float(train_accuracy),
            "test_accuracy": float(test_accuracy),
            "train_roc_auc": float(train_roc_auc),
            "test_roc_auc": float(test_roc_auc),
            "y_pred": y_test_pred,
            "y_proba": y_test_proba,
        }
        trained_models[name] = model

        print(f"  CV ROC AUC: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        print(f"  Test ROC AUC: {test_roc_auc:.4f} | Accuracy: {test_accuracy:.4f}")

    return results_binary, trained_models


def plot_binary_roc_curves(y_test_binary: pd.Series, results_binary: Dict[str, dict], savepath: Optional[str] = None) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    for idx, (name, result) in enumerate(results_binary.items()):
        fpr, tpr, _ = roc_curve(y_test_binary, result["y_proba"])  # type: ignore
        roc_auc = result["test_roc_auc"]
        axes[idx].plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
        axes[idx].plot([0, 1], [0, 1], lw=2, linestyle="--", label="Random")
        axes[idx].set_xlim([0.0, 1.0])
        axes[idx].set_ylim([0.0, 1.05])
        axes[idx].set_xlabel("False Positive Rate", fontsize=11)
        axes[idx].set_ylabel("True Positive Rate", fontsize=11)
        axes[idx].set_title(f"{name}\nROC Curve", fontsize=13, fontweight="bold")
        axes[idx].legend(loc="lower right")
        axes[idx].grid(alpha=0.3)
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    plt.show()


def plot_binary_confusion_matrices(y_test_binary: pd.Series, results_binary: Dict[str, dict], savepath: Optional[str] = None) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()
    for idx, (name, result) in enumerate(results_binary.items()):
        cm = confusion_matrix(y_test_binary, result["y_pred"])  # type: ignore
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx], xticklabels=["Weak", "Strong"], yticklabels=["Weak", "Strong"])  # type: ignore
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        axes[idx].set_title(
            f"{name}\nAUC: {result['test_roc_auc']:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f}",
            fontsize=11,
            fontweight="bold",
        )
        axes[idx].set_xlabel("Predicted", fontsize=10)
        axes[idx].set_ylabel("True", fontsize=10)
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    plt.show()


def binary_classification_report(best_model_name: str, y_test_binary: pd.Series, results_binary: Dict[str, dict]) -> str:
    y_pred_best = results_binary[best_model_name]["y_pred"]
    report = classification_report(y_test_binary, y_pred_best, target_names=["Weak (EF0-1)", "Strong (EF2+)"], digits=4)
    print("\n" + "-" * 80)
    print(f"CLASSIFICATION REPORT - {best_model_name}:")
    print("-" * 80)
    print(report)
    return report


# =============================================================================
# Modeling: Multi-class
# =============================================================================

def _get_multiclass_models(random_state: int = 42) -> Dict[str, object]:
    models: Dict[str, object] = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=random_state, class_weight="balanced", multi_class="multinomial", solver="lbfgs"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1, min_samples_split=10, min_samples_leaf=4, random_state=random_state
        ),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=random_state,
            eval_metric="mlogloss",
            use_label_encoder=False,
        )
    return models


def smote_resample_multiclass(X_train_scaled: pd.DataFrame, y_train_class: pd.Series, random_state: int = 42):
    if SMOTE is None:
        raise ImportError("imblearn is required for SMOTE; install with `pip install imbalanced-learn`. ")
    print("\nApplying SMOTE for multi-class classification...")
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train_scaled, y_train_class)
    print(f"✓ Training samples: {len(y_train_class)} → {len(y_res)}")
    for cls in sorted(pd.Series(y_res).unique()):
        print(f"  Class {cls}: {(y_res == cls).sum()}")
    return X_res, y_res


def train_multiclass_models(
    X_train_balanced: pd.DataFrame,
    y_train_balanced: pd.Series,
    X_test_scaled: pd.DataFrame,
    y_test_class: pd.Series,
    cv_cfg: CVConfig = CVConfig(),
) -> Tuple[Dict[str, dict], Dict[str, object]]:
    print("\n" + "-" * 80)
    print("TRAINING MULTI-CLASS CLASSIFICATION MODELS:")
    print("-" * 80)

    models = _get_multiclass_models(random_state=cv_cfg.random_state)
    results_class: Dict[str, dict] = {}
    trained_models: Dict[str, object] = {}

    for name, model in models.items():
        print(f"\nTraining: {name}")
        cv = StratifiedKFold(n_splits=cv_cfg.folds, shuffle=cv_cfg.shuffle, random_state=cv_cfg.random_state)
        cv_scores = cross_val_score(model, X_train_balanced, y_train_balanced, cv=cv, scoring="accuracy", n_jobs=-1)

        model.fit(X_train_balanced, y_train_balanced)
        y_test_pred = model.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test_class, y_test_pred)

        results_class[name] = {
            "cv_mean": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "test_accuracy": float(test_accuracy),
            "y_pred": y_test_pred,
        }
        trained_models[name] = model

        print(f"  CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        print(f"  Test Accuracy: {test_accuracy:.4f}")

    return results_class, trained_models


def plot_multiclass_confusions(y_test_class: pd.Series, results_class: Dict[str, dict], savepath: Optional[str] = None) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.ravel()
    for idx, (name, result) in enumerate(results_class.items()):
        cm = confusion_matrix(y_test_class, result["y_pred"])  # type: ignore
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes[idx],
            xticklabels=[f"EF-{i}" for i in range(4)],
            yticklabels=[f"EF-{i}" for i in range(4)],
        )
        axes[idx].set_title(f"{name}\nAccuracy: {result['test_accuracy']:.3f}", fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("Predicted", fontsize=10)
        axes[idx].set_ylabel("True", fontsize=10)
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    plt.show()


def multiclass_classification_report(best_model_name: str, y_test_class: pd.Series, results_class: Dict[str, dict]) -> str:
    y_pred_best = results_class[best_model_name]["y_pred"]
    report = classification_report(y_test_class, y_pred_best, target_names=[f"EF-{i}" for i in range(4)], digits=4)
    print("\n" + "-" * 80)
    print(f"CLASSIFICATION REPORT - {best_model_name}:")
    print("-" * 80)
    print(report)
    return report


# =============================================================================
# Feature Importance (Tree-based)
# =============================================================================

def compute_feature_importances(trained_models: Dict[str, object], feature_names: List[str]) -> Dict[str, pd.DataFrame]:
    importance_data: Dict[str, pd.DataFrame] = {}
    for name, model in trained_models.items():
        if hasattr(model, "feature_importances_"):
            importance_data[name] = (
                pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
                .sort_values("importance", ascending=False)
            )
    if importance_data:
        print("\n" + "-" * 80)
        print("FEATURE IMPORTANCE:")
        print("-" * 80)
    return importance_data


def plot_top_feature_importances(importance_data: Dict[str, pd.DataFrame], top_k: int = 15, savepath: Optional[str] = None) -> None:
    if not importance_data:
        print("No tree-based feature importances available.")
        return

    keys = list(importance_data.keys())[: min(len(importance_data), 3)]
    fig, axes = plt.subplots(1, len(keys), figsize=(6 * len(keys) + 6, 6))
    if len(keys) == 1:
        axes = [axes]  # type: ignore

    for idx, name in enumerate(keys):
        imp_df = importance_data[name].head(top_k)
        print(f"\n{name} - Top {min(top_k, len(imp_df))} Features:")
        print(imp_df.head(min(top_k, len(imp_df))).to_string(index=False))
        axes[idx].barh(imp_df["feature"], imp_df["importance"])  # type: ignore
        axes[idx].set_xlabel("Importance")
        axes[idx].set_title(f"{name}\nFeature Importance", fontsize=12, fontweight="bold")
        axes[idx].invert_yaxis()
        axes[idx].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    plt.show()


# =============================================================================
# Final Summary
# =============================================================================

def print_final_summary(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    best_model_name_binary: str,
    best_roc_auc_binary: float,
    best_model_name_class: str,
    best_accuracy_class: float,
    X_train_balanced_binary: pd.DataFrame,
    y_train_balanced_binary: pd.Series,
    X_train_balanced_class: pd.DataFrame,
    y_train_balanced_class: pd.Series,
    X_test_scaled: pd.DataFrame,
    y_test_binary: pd.Series,
    y_test_class: pd.Series,
) -> None:
    print("\n" + "=" * 80)
    print("STEP 8: FINAL SUMMARY & QUANTUM CHALLENGE TARGETS")
    print("=" * 80)

    print(
        f"""
✅ CLASSICAL BASELINE ANALYSIS COMPLETE!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATASET SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Training Samples: {X_train.shape[0]}
• Test Samples: {X_test.shape[0]}
• Features: {X_train.shape[1]} meteorological variables
• Targets: Binary (ef_binary) + Multi-class (ef_class)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 YOUR QUANTUM CHALLENGE TARGETS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PRIMARY CHALLENGE (Binary Classification):
Best Model: {best_model_name_binary}
Target ROC AUC to Beat: {best_roc_auc_binary:.4f}

ADVANCED CHALLENGE (Multi-class Classification):
Best Model: {best_model_name_class}
Target Accuracy to Beat: {best_accuracy_class:.4f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATA AVAILABLE FOR QUANTUM MODELS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Binary Classification:
- X_train_balanced_binary: {X_train_balanced_binary.shape}
- y_train_balanced_binary: {y_train_balanced_binary.shape}
- X_test_scaled: {X_test_scaled.shape}
- y_test_binary: {y_test_binary.shape}

Multi-class Classification:
- X_train_balanced_class: {X_train_balanced_class.shape}
- y_train_balanced_class: {y_train_balanced_class.shape}
- X_test_scaled: {X_test_scaled.shape}
- y_test_class: {y_test_class.shape}
"""
    )


# =============================================================================
# Example scriptable flow (call these from your driver .py or notebook)
# =============================================================================

def full_pipeline(
    train_path: str,
    test_path: str,
    scaling: ScalingConfig = ScalingConfig(),
    cv_cfg: CVConfig = CVConfig(),
    run_pca_plots: bool = True,
) -> Dict[str, object]:
    """Run a full baseline pipeline and return the key artifacts.

    Returns a dict containing data splits, models, metrics, etc.
    """
    # Load + info
    df_train, df_test = load_data(train_path, test_path)
    print_basic_info(df_train, df_test)
    plot_target_distributions(df_train, df_test)

    # XY split
    X_train, X_test, y_train_binary, y_test_binary, y_train_class, y_test_class = separate_features_targets(df_train, df_test)

    # Impute + scale
    X_train_scaled, X_test_scaled, imputer, scaler = impute_and_scale(X_train, X_test, scaling=scaling)

    # Export scaled (optional): caller can do this; default filenames if desired
    # export_scaled_data(X_train_scaled, X_test_scaled)

    # Correlation
    corr = save_correlation_matrix(X_train_scaled)

    # PCA (optional)
    pca_artifacts = None
    if run_pca_plots:
        pca_artifacts = run_pca(X_train_scaled, y_train_binary)

    # Binary modeling
    X_train_balanced_binary, y_train_balanced_binary = smote_resample_binary(X_train_scaled, y_train_binary, random_state=cv_cfg.random_state)
    results_binary, trained_models_binary = train_binary_models(
        X_train_balanced_binary,
        y_train_balanced_binary,
        X_train_scaled,
        X_test_scaled,
        y_train_binary,
        y_test_binary,
        cv_cfg=cv_cfg,
    )

    results_df_binary = pd.DataFrame(results_binary).T
    best_model_name_binary = results_df_binary["test_roc_auc"].idxmax()
    best_roc_auc_binary = float(results_df_binary.loc[best_model_name_binary, "test_roc_auc"])  # type: ignore

    # Reports & plots
    _ = binary_classification_report(best_model_name_binary, y_test_binary, results_binary)
    plot_binary_roc_curves(y_test_binary, results_binary)
    plot_binary_confusion_matrices(y_test_binary, results_binary)

    # Importances
    imp_binary = compute_feature_importances(trained_models_binary, X_train.columns.tolist())
    if imp_binary:
        plot_top_feature_importances(imp_binary)

    # Multiclass modeling
    X_train_balanced_class, y_train_balanced_class = smote_resample_multiclass(X_train_scaled, y_train_class, random_state=cv_cfg.random_state)
    results_class, trained_models_class = train_multiclass_models(
        X_train_balanced_class,
        y_train_balanced_class,
        X_test_scaled,
        y_test_class,
        cv_cfg=cv_cfg,
    )

    results_df_class = pd.DataFrame(results_class).T
    best_model_name_class = results_df_class["test_accuracy"].idxmax()
    best_accuracy_class = float(results_df_class.loc[best_model_name_class, "test_accuracy"])  # type: ignore

    _ = multiclass_classification_report(best_model_name_class, y_test_class, results_class)
    plot_multiclass_confusions(y_test_class, results_class)

    # Final summary
    print_final_summary(
        X_train,
        X_test,
        best_model_name_binary,
        best_roc_auc_binary,
        best_model_name_class,
        best_accuracy_class,
        X_train_balanced_binary,
        y_train_balanced_binary,
        X_train_balanced_class,
        y_train_balanced_class,
        X_test_scaled,
        y_test_binary,
        y_test_class,
    )

    return {
        "df_train": df_train,
        "df_test": df_test,
        "X_train": X_train,
        "X_test": X_test,
        "y_train_binary": y_train_binary,
        "y_test_binary": y_test_binary,
        "y_train_class": y_train_class,
        "y_test_class": y_test_class,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "imputer": imputer,
        "scaler": scaler,
        "correlation": corr,
        "pca": pca_artifacts,
        "results_binary": results_binary,
        "trained_models_binary": trained_models_binary,
        "best_model_name_binary": best_model_name_binary,
        "best_roc_auc_binary": best_roc_auc_binary,
        "feature_importances_binary": imp_binary,
        "results_class": results_class,
        "trained_models_class": trained_models_class,
        "best_model_name_class": best_model_name_class,
        "best_accuracy_class": best_accuracy_class,
    }


if __name__ == "__main__":
    # Minimal CLI-like example (edit paths):
    TRAIN_FILE = "2025-Quantathon-Tornado-Q-training_data-640-examples.xlsx"
    TEST_FILE = "2025-Quantum-Tornado-Q-test_data-200-examples.xlsx"

    artifacts = full_pipeline(TRAIN_FILE, TEST_FILE)
    # Save scaled CSVs if desired
    export_scaled_data(artifacts["X_train_scaled"], artifacts["X_test_scaled"])  # type: ignore
