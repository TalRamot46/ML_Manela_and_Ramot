"""
ex2.py — Standalone end-to-end pipeline (no imports from part_A / part_B / partD).

This file mirrors the functionality of the project under ``code/`` when data live under
``data/`` at the repository root:

  1) Data loading: iterate ``.pgm`` from a directory or from a zip (same contract as part_A).
  2) Preprocessing: optimized pipeline — resize 64×64, CLAHE, HOG, z-score, PCA (Part C).
  3) Persist / load Part C cache: ``data/part_c_splits_optimized.npz`` (train/val split).
  4) Nested stratified K-fold CV (outer K_out, inner K_in) with hyperparameter grids.
  5) Optional multi-model comparison (SVM, custom L1-softmax, custom decision tree).
  6) Final model fit on the full labeled pool (stacked train+val PCA features).
  7) Unlabeled prediction: raw HOG → same StandardScaler+PCA fitted on labeled raw HOG
     (matches ``final_model.py``), then predict; write ``results_manela_and_ramot.csv``
     as a single row of length 392 with identity indices 1..28.

Environment (optional):
  EX2_FULL_PIPELINE=1  — run nested CV for SVM, linear_reg, and tree, then pick the
                        lowest mean outer error model for the final fit (very slow).
  EX2_FULL_PIPELINE=0 — default: nested CV for SVM only; final model is SVM with
                        hyperparameters chosen by majority vote over outer folds.
"""

from __future__ import annotations

import os
import pickle
import zipfile
from collections import Counter
from itertools import product
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import cv2
import numpy as np
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **_kwargs):
        return iterable


# ---------------------------------------------------------------------------
# Paths (repo layout: <repo>/code/ex2.py, <repo>/data/...)
# ---------------------------------------------------------------------------

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CODE_DIR)
TRAIN_DIR = os.path.join(REPO_ROOT, "data", "Train Set (Labeled)")
TEST_DIR = os.path.join(REPO_ROOT, "data", "Test Set (Unlabeled)")
PARTC_CACHE = os.path.join(REPO_ROOT, "data", "part_c_splits_optimized.npz")
RESULTS_CSV_PATH = os.path.join(REPO_ROOT, "results_manela_and_ramot.csv")
MODEL_BUNDLE_PATH = os.path.join(REPO_ROOT, "data", "ex2_final_bundle.pkl")
NESTED_CV_LOG_PATH = os.path.join(REPO_ROOT, "ex2_nested_cv_log.txt")

N_COMPONENTS = 50
PARTC_TEST_SIZE = 0.25
PARTC_RANDOM_STATE = 42
K_OUT = 5
K_IN = 3
NESTED_CV_RANDOM_STATE = 42

EX2_FULL_PIPELINE = os.environ.get("EX2_FULL_PIPELINE", "0") == "1"


# =============================================================================
# Part A–equivalent: PGM iteration, HOG, PCA cache
# =============================================================================


def iter_pgm_files(data_path: str):
    """Yield (filename, path_or_none, bytes_or_none) like part_A._iter_pgm_files."""
    if os.path.isdir(data_path):
        for filename in sorted(os.listdir(data_path)):
            if filename.lower().endswith(".pgm"):
                yield filename, os.path.join(data_path, filename), None
    else:
        with zipfile.ZipFile(data_path, "r") as archive:
            for filename in sorted(name for name in archive.namelist() if name.lower().endswith(".pgm")):
                yield filename, None, archive.read(filename)


def extract_optimized_hog_features(data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Raw HOG feature matrix (optimized pipeline): 64×64 resize, CLAHE, HOG (9 ori, 8×8, 2×2).
    Returns (X_raw, y_str) where y_str[i] is the person id token from the filename prefix.
    """
    features_list: List[np.ndarray] = []
    labels: List[str] = []
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    for filename, file_path, file_bytes in iter_pgm_files(data_path):
        if file_bytes is not None:
            img_data = file_bytes
        else:
            with open(file_path, "rb") as file_handle:
                img_data = file_handle.read()

        img_array = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img_small = cv2.resize(img, (64, 64))
        img_clahe = clahe.apply(img_small)
        fd = hog(
            img_clahe,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=False,
        )
        features_list.append(fd)
        labels.append(os.path.splitext(os.path.basename(filename))[0].split("_")[0])

    return np.asarray(features_list), np.asarray(labels)


def process_optimized(train_dir: str, n_components: int = N_COMPONENTS) -> Tuple[np.ndarray, np.ndarray]:
    """HOG → StandardScaler → PCA (same as part_A.process_optimized)."""
    X, labels_str = extract_optimized_hog_features(train_dir)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=min(n_components, X_scaled.shape[0], X_scaled.shape[1]))
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, labels_str


def partc_build_splits(
    train_dir: str,
    *,
    n_components: int = N_COMPONENTS,
    test_size: float = PARTC_TEST_SIZE,
    random_state: int = PARTC_RANDOM_STATE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
    """Part C: PCA features, label encode, stratified train/val split (part_A._partc_build_splits)."""
    X_full, y_full_str = process_optimized(train_dir, n_components=n_components)
    le = LabelEncoder()
    y_full = le.fit_transform(y_full_str)
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=test_size, random_state=random_state
    )
    num_classes = int(len(le.classes_))
    return X_train, X_val, y_train, y_val, num_classes, le.classes_


def save_partc_train_val_npz(
    train_dir: str,
    npz_path: str,
    *,
    n_components: int = N_COMPONENTS,
    test_size: float = PARTC_TEST_SIZE,
    random_state: int = PARTC_RANDOM_STATE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    X_train, X_val, y_train, y_val, num_classes, label_classes = partc_build_splits(
        train_dir,
        n_components=n_components,
        test_size=test_size,
        random_state=random_state,
    )
    out_dir = os.path.dirname(os.path.abspath(npz_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        npz_path,
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        label_classes=label_classes,
    )
    return X_train, X_val, y_train, y_val, num_classes


def load_partc_train_val_npz(npz_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    data = np.load(npz_path, allow_pickle=True)
    label_classes = data["label_classes"]
    num_classes = int(len(label_classes))
    return data["X_train"], data["X_val"], data["y_train"], data["y_val"], num_classes


def load_full_labeled_pca_xy(
    train_dir: str = TRAIN_DIR,
    npz_path: str = PARTC_CACHE,
) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """Stack train+val from Part C cache (same as partD.load_full_partc_features + label_classes)."""
    if not os.path.isfile(npz_path):
        save_partc_train_val_npz(train_dir, npz_path)
    data = np.load(npz_path, allow_pickle=True)
    X_tr = data["X_train"]
    X_va = data["X_val"]
    y_tr = data["y_train"]
    y_va = data["y_val"]
    label_classes = np.asarray(data["label_classes"])
    num_classes = int(len(label_classes))
    X = np.vstack([X_tr, X_va])
    y = np.concatenate([y_tr, y_va])
    return X, y, num_classes, label_classes


# =============================================================================
# Part B–equivalent: L1 softmax regression + decision tree
# =============================================================================


class MySoftmaxRegressionL1:
    """From part_B: multiclass softmax with L1-style gradient (sign(weights))."""

    def __init__(self, lr: float = 0.01, epochs: int = 1000, reg: float = 0.1):
        self.lr = lr
        self.epochs = epochs
        self.reg = reg
        self.weights: Optional[np.ndarray] = None

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X: np.ndarray, y_encoded: np.ndarray, num_classes: int) -> None:
        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features, num_classes))
        for _ in range(self.epochs):
            scores = np.dot(X, self.weights)
            probs = self._softmax(scores)
            gradient = (1 / n_samples) * np.dot(X.T, (probs - y_encoded))
            gradient += self.reg * np.sign(self.weights)
            self.weights -= self.lr * gradient

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.weights is not None
        scores = np.dot(X, self.weights)
        return np.argmax(self._softmax(scores), axis=1)


class MyDecisionTree:
    """From part_B: axis-aligned splits on PCA/HOG-derived numeric features."""

    def __init__(self, max_depth: int = 10, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root: Optional[Dict[str, Any]] = None

    def _entropy(self, y: np.ndarray) -> float:
        proportions = np.bincount(y) / len(y)
        return -np.sum([p * np.log2(p) for p in proportions if p > 0])

    def _create_split(self, X: np.ndarray, thresh: float) -> Tuple[np.ndarray, np.ndarray]:
        left_idx = np.argwhere(X <= thresh).flatten()
        right_idx = np.argwhere(X > thresh).flatten()
        return left_idx, right_idx

    def _information_gain(self, X_column: np.ndarray, y: np.ndarray, thresh: float) -> float:
        parent_entropy = self._entropy(y)
        left_idx, right_idx = self._create_split(X_column, thresh)
        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0.0
        n = len(y)
        n_l, n_r = len(left_idx), len(right_idx)
        e_l, e_r = self._entropy(y[left_idx]), self._entropy(y[right_idx])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r
        return parent_entropy - child_entropy

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float]]:
        best_gain = -1.0
        split_idx, split_thresh = None, None
        n_features = X.shape[1]
        feature_indices = np.random.choice(n_features, min(30, n_features), replace=False)
        for i in feature_indices:
            X_column = X[:, i]
            thresholds = np.unique(X_column)
            for thresh in thresholds:
                gain = self._information_gain(X_column, y, float(thresh))
                if gain > best_gain:
                    best_gain = gain
                    split_idx = int(i)
                    split_thresh = float(thresh)
        return split_idx, split_thresh

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Dict[str, Any]:
        n_samples, _n_features = X.shape
        n_labels = len(np.unique(y))
        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            most_common = int(np.bincount(y).argmax())
            return {"leaf": True, "class": most_common}
        idx, thresh = self._best_split(X, y)
        if idx is None or thresh is None:
            most_common = int(np.bincount(y).argmax())
            return {"leaf": True, "class": most_common}
        left_idx, right_idx = self._create_split(X[:, idx], thresh)
        left_node = self._build_tree(X[left_idx, :], y[left_idx], depth + 1)
        right_node = self._build_tree(X[right_idx, :], y[right_idx], depth + 1)
        return {"leaf": False, "index": idx, "threshold": thresh, "left": left_node, "right": right_node}

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.root = self._build_tree(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.root is not None
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x: np.ndarray, node: Dict[str, Any]) -> int:
        if node["leaf"]:
            return int(node["class"])
        if x[node["index"]] <= node["threshold"]:
            return self._traverse_tree(x, node["left"])
        return self._traverse_tree(x, node["right"])


# =============================================================================
# Part D–equivalent: nested CV, model comparison
# =============================================================================


def param_grid_product(param_grid: Mapping[str, Iterable[Any]]) -> List[Dict[str, Any]]:
    keys = list(param_grid.keys())
    return [dict(zip(keys, vals)) for vals in product(*[param_grid[k] for k in keys])]


def misclassification_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(1.0 - accuracy_score(y_true, y_pred))


def fit_predict_nested(
    model_name: str,
    params: Mapping[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    num_classes: int,
    tree_random_seed: Optional[int],
) -> np.ndarray:
    """Train one hypothesis and predict on X_test (partD._fit_predict)."""
    name = model_name.strip().lower()
    if name == "svm":
        model = SVC(kernel="rbf", **dict(params))
        model.fit(X_train, y_train)
        return model.predict(X_test)
    if name in ("linear_reg", "softmax_l1", "logreg_l1"):
        lr = params.get("lr", 0.05)
        epochs = int(params.get("epochs", 100))
        reg = params.get("reg", 0.1)
        model = MySoftmaxRegressionL1(lr=lr, epochs=epochs, reg=reg)
        y_enc = np.eye(num_classes)[y_train]
        model.fit(X_train, y_enc, num_classes)
        return model.predict(X_test)
    if name in ("tree", "decision_tree"):
        if tree_random_seed is not None:
            np.random.seed(tree_random_seed)
        max_depth = int(params.get("max_depth", 10))
        min_samples_split = int(params.get("min_samples_split", 2))
        model = MyDecisionTree(max_depth=max_depth, min_samples_split=min_samples_split)
        model.fit(X_train, y_train)
        return model.predict(X_test)
    raise ValueError(f"Unknown model_name {model_name!r}.")


def nested_cross_validate(
    model_name: str,
    param_grid: Mapping[str, Iterable[Any]],
    X: np.ndarray,
    y: np.ndarray,
    K_out: int,
    K_in: int,
    num_classes: int,
    random_state: int = NESTED_CV_RANDOM_STATE,
) -> Dict[str, Any]:
    """
    Nested stratified CV (partD.nested_cross_validate): outer folds hold out test;
    inner CV scores each hyperparameter tuple by mean validation error; best tuple
    is refit on all outer-train data and scored on the outer test fold.
    """
    if K_out < 2 or K_in < 2:
        raise ValueError("K_out and K_in must be at least 2.")
    combos = param_grid_product(param_grid)
    if not combos:
        raise ValueError("param_grid must contain at least one hyperparameter with values.")

    outer_errors: List[float] = []
    chosen_hyperparameters: List[Dict[str, Any]] = []
    outer_test_indices: List[np.ndarray] = []

    print(f"\nNested CV: {model_name} | {len(combos)} hyperparameter tuples | K_out={K_out} K_in={K_in}")
    skf_out = StratifiedKFold(n_splits=K_out, shuffle=True, random_state=random_state)
    for fold_out, (idx_outer_train, idx_outer_test) in tqdm(
        enumerate(skf_out.split(X, y)), total=K_out, desc=f"{model_name} outer"
    ):
        X_ot, y_ot = X[idx_outer_train], y[idx_outer_train]
        X_oe, y_oe = X[idx_outer_test], y[idx_outer_test]

        best_mean_inner = np.inf
        best_params: Optional[Dict[str, Any]] = None
        skf_in = StratifiedKFold(
            n_splits=K_in,
            shuffle=True,
            random_state=random_state + 1000 * (fold_out + 1),
        )

        for params in combos:
            inner_errors: List[float] = []
            for idx_in_train, idx_in_val in skf_in.split(X_ot, y_ot):
                X_in_tr, y_in_tr = X_ot[idx_in_train], y_ot[idx_in_train]
                X_in_va, y_in_va = X_ot[idx_in_val], y_ot[idx_in_val]
                y_pred = fit_predict_nested(
                    model_name,
                    params,
                    X_in_tr,
                    y_in_tr,
                    X_in_va,
                    num_classes,
                    tree_random_seed=random_state + fold_out * 7919 + len(inner_errors),
                )
                inner_errors.append(misclassification_error(y_in_va, y_pred))
            mean_inner = float(np.mean(inner_errors))
            if mean_inner < best_mean_inner:
                best_mean_inner = mean_inner
                best_params = dict(params)

        assert best_params is not None
        y_hat_outer = fit_predict_nested(
            model_name,
            best_params,
            X_ot,
            y_ot,
            X_oe,
            num_classes,
            tree_random_seed=random_state + fold_out * 4999,
        )
        outer_errors.append(misclassification_error(y_oe, y_hat_outer))
        chosen_hyperparameters.append(best_params)
        outer_test_indices.append(np.asarray(idx_outer_test))

    mean_outer_error = float(np.mean(outer_errors))
    std_outer_error = float(np.std(outer_errors, ddof=1)) if K_out > 1 else 0.0
    return {
        "model_name": model_name,
        "K_out": K_out,
        "K_in": K_in,
        "outer_fold_errors": outer_errors,
        "chosen_hyperparameters": chosen_hyperparameters,
        "outer_fold_test_indices": outer_test_indices,
        "mean_outer_error": mean_outer_error,
        "std_outer_error": std_outer_error,
    }


def compare_models_nested_cv(
    model_specs: Mapping[str, Tuple[str, Mapping[str, Iterable[Any]]]],
    X: np.ndarray,
    y: np.ndarray,
    K_out: int,
    K_in: int,
    num_classes: int,
    random_state: int = NESTED_CV_RANDOM_STATE,
    log_path: str = NESTED_CV_LOG_PATH,
) -> Dict[str, Any]:
    """Run nested CV for each model family and rank by mean outer error (partD.compare_models_nested_cv)."""
    results: Dict[str, Any] = {}
    for label, (mname, grid) in model_specs.items():
        results[label] = nested_cross_validate(
            mname, grid, X, y, K_out, K_in, num_classes, random_state=random_state
        )
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"=== {label} ===\n")
            for key, value in results[label].items():
                f.write(f"{key}: {value}\n")
            f.write("\n")

    ranking = sorted(results.items(), key=lambda kv: kv[1]["mean_outer_error"])
    labels = list(results.keys())
    best_idx = int(np.argmin([results[l]["mean_outer_error"] for l in labels]))
    best_label = labels[best_idx]
    comparison: Dict[str, Any] = {
        "per_model": results,
        "ranking_by_mean_outer_error": [name for name, _ in ranking],
        "overall_recommendation": (
            f"Lowest mean nested-CV error: {best_label} "
            f"(mean={results[best_label]['mean_outer_error']:.4f}, "
            f"std={results[best_label]['std_outer_error']:.4f})."
        ),
    }
    return comparison


def print_nested_cv_summary(ncv: Mapping[str, Any]) -> None:
    print(f"\n=== Nested CV: {ncv['model_name']} (K_out={ncv['K_out']}, K_in={ncv['K_in']}) ===")
    print(f"Mean outer error: {ncv['mean_outer_error']:.4f} +/- {ncv['std_outer_error']:.4f}")
    for i, (err, params) in enumerate(zip(ncv["outer_fold_errors"], ncv["chosen_hyperparameters"]), 1):
        print(f"  Outer fold {i}: error={err:.4f} | best params={params}")


def majority_svm_params(chosen: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate outer-fold SVM winners by majority vote on (C, gamma) — stable default."""
    counts: Dict[Tuple[Any, Any], int] = {}
    for p in chosen:
        key = (p["C"], p["gamma"])
        counts[key] = counts.get(key, 0) + 1
    best_key = max(counts.items(), key=lambda kv: kv[1])[0]
    return {"C": best_key[0], "gamma": best_key[1]}


def majority_linear_params(chosen: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts: Dict[Tuple[Any, Any, Any], int] = {}
    for p in chosen:
        key = (p["lr"], p["epochs"], p["reg"])
        counts[key] = counts.get(key, 0) + 1
    best_key = max(counts.items(), key=lambda kv: kv[1])[0]
    return {"lr": best_key[0], "epochs": best_key[1], "reg": best_key[2]}


def majority_tree_params(chosen: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts: Dict[Tuple[int, int], int] = {}
    for p in chosen:
        key = (int(p["max_depth"]), int(p["min_samples_split"]))
        counts[key] = counts.get(key, 0) + 1
    best_key = max(counts.items(), key=lambda kv: kv[1])[0]
    return {"max_depth": best_key[0], "min_samples_split": best_key[1]}


# =============================================================================
# Final-model–equivalent: scaler/PCA on raw HOG for unlabeled + bundle I/O
# =============================================================================


def fit_scaler_pca_on_labeled_hog(
    train_dir: str = TRAIN_DIR,
    n_components: int = N_COMPONENTS,
) -> Tuple[StandardScaler, PCA]:
    X_raw, _ = extract_optimized_hog_features(train_dir)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    pca = PCA(n_components=min(n_components, X_scaled.shape[0], X_scaled.shape[1]))
    pca.fit(X_scaled)
    return scaler, pca


def load_or_fit_preprocessing_bundle(
    train_dir: str,
    bundle_path: str,
) -> Tuple[StandardScaler, PCA]:
    if os.path.isfile(bundle_path):
        try:
            with open(bundle_path, "rb") as f:
                bundle = pickle.load(f)
            scaler, pca = bundle.get("scaler"), bundle.get("pca")
            if isinstance(scaler, StandardScaler) and isinstance(pca, PCA):
                return scaler, pca
        except (pickle.UnpicklingError, EOFError, KeyError, TypeError):
            pass
    return fit_scaler_pca_on_labeled_hog(train_dir, N_COMPONENTS)


def transform_dir_hog_to_pca(data_dir: str, scaler: StandardScaler, pca: PCA) -> np.ndarray:
    X_raw, _ = extract_optimized_hog_features(data_dir)
    return pca.transform(scaler.transform(X_raw))


def label_strings_to_identity_ids(label_strings: np.ndarray) -> np.ndarray:
    return np.array([int(s.lstrip("p")) + 1 for s in label_strings], dtype=int)


def fit_final_classifier(
    winner: str,
    params: Dict[str, Any],
    X_full: np.ndarray,
    y_full: np.ndarray,
    num_classes: int,
) -> Tuple[Any, str]:
    """Fit the production classifier on the full labeled PCA matrix."""
    name = winner.strip().lower()
    if name == "svm":
        model = SVC(kernel="rbf", **params)
        model.fit(X_full, y_full)
        return model, "svm"
    if name in ("linear_reg", "softmax_l1", "logreg_l1"):
        m = MySoftmaxRegressionL1(
            lr=float(params["lr"]),
            epochs=int(params["epochs"]),
            reg=float(params["reg"]),
        )
        y_enc = np.eye(num_classes)[y_full]
        m.fit(X_full, y_enc, num_classes)
        return m, "linear_reg"
    if name in ("tree", "decision_tree"):
        np.random.seed(NESTED_CV_RANDOM_STATE)
        m = MyDecisionTree(
            max_depth=int(params["max_depth"]),
            min_samples_split=int(params["min_samples_split"]),
        )
        m.fit(X_full, y_full)
        return m, "tree"
    raise ValueError(winner)


def predict_final(
    model_kind: str,
    model: Any,
    X_unlabeled: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    if model_kind == "svm":
        return model.predict(X_unlabeled)
    if model_kind == "linear_reg":
        return model.predict(X_unlabeled)
    if model_kind == "tree":
        return model.predict(X_unlabeled)
    raise ValueError(model_kind)


# =============================================================================
# Main
# =============================================================================


def default_param_grids() -> Dict[str, Tuple[str, Dict[str, Iterable[Any]]]]:
    """Same grids as partD_main.py ``specs``."""
    svm_grid = {
        "C": [1.0, 10.0, 100.0, 1000.0, 10000.0],
        "gamma": [1e-4, 1e-2, 1, 100, "scale"],
    }
    linear_grid = {
        "lr": [1e-4, 0.001, 0.01, 0.05],
        "epochs": [100, 200, 500],
        "reg": [0.01, 0.5, 1, 10],
    }
    tree_grid = {
        "max_depth": [2, 4],
        "min_samples_split": [2, 4],
    }
    return {
        "SVM": ("SVM", svm_grid),
        "linear_reg": ("linear_reg", linear_grid),
        "tree": ("tree", tree_grid),
    }


def main() -> None:
    print("ex2: loading / building Part C cache …")
    X_full, y_full, num_classes, label_classes = load_full_labeled_pca_xy(TRAIN_DIR, PARTC_CACHE)
    print(f"  Labeled PCA pool: X.shape={X_full.shape}, num_classes={num_classes}")

    specs = default_param_grids()

    if EX2_FULL_PIPELINE:
        print("EX2_FULL_PIPELINE=1: nested CV for SVM, linear_reg, tree (this takes a long time).")
        comparison = compare_models_nested_cv(
            specs, X_full, y_full, K_OUT, K_IN, num_classes, random_state=NESTED_CV_RANDOM_STATE
        )
        winner = comparison["ranking_by_mean_outer_error"][0]
        print(comparison["overall_recommendation"])
        per = comparison["per_model"][winner]
        if winner == "SVM":
            final_params = majority_svm_params(per["chosen_hyperparameters"])
        elif winner == "linear_reg":
            final_params = majority_linear_params(per["chosen_hyperparameters"])
        else:
            final_params = majority_tree_params(per["chosen_hyperparameters"])
        nested_for_bundle: Dict[str, Any] = {"mode": "full_compare", "comparison": comparison}
    else:
        print("EX2_FULL_PIPELINE=0: SVM nested CV only (set EX2_FULL_PIPELINE=1 for all models).")
        svm_only = nested_cross_validate(
            "SVM",
            specs["SVM"][1],
            X_full,
            y_full,
            K_OUT,
            K_IN,
            num_classes,
            random_state=NESTED_CV_RANDOM_STATE,
        )
        print_nested_cv_summary(svm_only)
        winner = "SVM"
        final_params = majority_svm_params(svm_only["chosen_hyperparameters"])
        nested_for_bundle = {"mode": "svm_only", "svm_nested_cv": svm_only}

    print(f"\nFinal model family: {winner}")
    print(f"Final hyperparameters (aggregated): {final_params}")

    model, model_kind = fit_final_classifier(winner, final_params, X_full, y_full, num_classes)

    scaler, pca = load_or_fit_preprocessing_bundle(TRAIN_DIR, MODEL_BUNDLE_PATH)
    X_unlabeled = transform_dir_hog_to_pca(TEST_DIR, scaler, pca)
    y_pred_idx = predict_final(model_kind, model, X_unlabeled, num_classes)
    pred_label_str = label_classes[np.asarray(y_pred_idx, dtype=int)]
    y_pred_identity = label_strings_to_identity_ids(pred_label_str)

    if y_pred_identity.size != 392:
        print(f"Warning: expected 392 predictions, got {y_pred_identity.size}.")

    bundle: Dict[str, Any] = {
        "winner": winner,
        "model_kind": model_kind,
        "final_params": final_params,
        "model": model,
        "scaler": scaler,
        "pca": pca,
        "label_classes": label_classes,
        "partc_cache": PARTC_CACHE,
        "nested_cv": nested_for_bundle,
    }
    os.makedirs(os.path.dirname(MODEL_BUNDLE_PATH), exist_ok=True)
    with open(MODEL_BUNDLE_PATH, "wb") as f:
        pickle.dump(bundle, f)

    np.savetxt(RESULTS_CSV_PATH, y_pred_identity.reshape(1, -1), delimiter=",", fmt="%d")
    print(f"\nSaved bundle: {MODEL_BUNDLE_PATH}")
    print(f"Saved predictions (1×{y_pred_identity.size}): {RESULTS_CSV_PATH}")


if __name__ == "__main__":
    main()
