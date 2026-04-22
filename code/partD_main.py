"""
Part D: nested cross-validation (outer K_out, inner K_in) for model / hyperparameter selection.
"""
from __future__ import annotations

import os
from itertools import product
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from tqdm import tqdm
import time
import part_A
import part_B

# ---------------------------------------------------------------------------
# Data loading (full labeled pool = train + val from Part C cache)
# ---------------------------------------------------------------------------

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_CODE_DIR)
TRAIN_DIR = os.path.join(_REPO_ROOT, "data", "Train Set (Labeled)")
PARTC_CACHE = os.path.join(_REPO_ROOT, "data", "part_c_splits_optimized.npz")


def load_full_partc_features(
    npz_path: str = PARTC_CACHE,
    train_dir: str = TRAIN_DIR,) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Load Part C features and return the full matrix X, labels y, and num_classes
    (concatenation of the saved train and validation splits).
    """
    if os.path.isfile(npz_path):
        X_tr, X_va, y_tr, y_va, num_classes = part_A.load_partc_train_val_npz(npz_path)
    else:
        X_tr, X_va, y_tr, y_va, num_classes = part_A.save_partc_train_val_npz(
            train_dir, npz_path, pipeline="optimized"
        )
    X = np.vstack([X_tr, X_va])
    y = np.concatenate([y_tr, y_va])
    return X, y, num_classes

def _param_grid_product(param_grid: Mapping[str, Iterable[Any]]) -> List[Dict[str, Any]]:
    # input - output example of this helper mapping function that the AI gave us:
    # {'C': [1.0, 10.0], 'gamma': ['scale', 0.1]} -> 
    # [{'C': 1.0, 'gamma': 'scale'}, {'C': 1.0, 'gamma': 0.1}, {'C': 10.0, 'gamma': 'scale'}, {'C': 10.0, 'gamma': 0.1}]

    keys = list(param_grid.keys())
    return [dict(zip(keys, vals)) for vals in product(*[param_grid[k] for k in keys])]


def _misclassification_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # a simple formula for calculating the error of a model with the labels.
    return float(1.0 - accuracy_score(y_true, y_pred))


def _fit_predict(
    model_name: str,
    params: Mapping[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    num_classes: int,
    tree_random_seed: Optional[int],
) -> np.ndarray:
    name = model_name.strip().lower()
    if name == "svm":
        model = SVC(kernel="rbf", **dict(params))
        model.fit(X_train, y_train)
        return model.predict(X_test)

    if name in ("linear_reg", "softmax_l1", "logreg_l1"):
        lr = params.get("lr", 0.05)
        epochs = int(params.get("epochs", 100))
        reg = params.get("reg", 0.1)
        model = part_B.MySoftmaxRegressionL1(lr=lr, epochs=epochs, reg=reg)
        y_enc = np.eye(num_classes)[y_train]
        model.fit(X_train, y_enc, num_classes)
        return model.predict(X_test)

    if name in ("tree", "decision_tree"):
        if tree_random_seed is not None:
            np.random.seed(tree_random_seed)
        max_depth = int(params.get("max_depth", 10))
        min_samples_split = int(params.get("min_samples_split", 2))
        model = part_B.MyDecisionTree(max_depth=max_depth, min_samples_split=min_samples_split)
        model.fit(X_train, y_train)
        return model.predict(X_test)

    raise ValueError(
        f"Unknown model_name {model_name!r}. Use 'SVM', 'linear_reg', or 'tree'."
    )


def nested_cross_validate(
    model_name: str,
    param_grid: Mapping[str, Iterable[Any]],
    X: np.ndarray,
    y: np.ndarray,
    K_out: int,
    K_in: int,
    num_classes: int,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Nested stratified cross-validation on the full dataset.

    Outer loop: hold out one fold as the outer test set.
    Inner loop: on the remaining data, K_in-fold CV; for each hyperparameter
    combination, average validation error across inner folds; pick the best
    combination; refit on all outer-train data; evaluate on the outer test fold.

    Parameters
    ----------
    model_name : str
        One of ``'SVM'``, ``'linear_reg'`` (``MySoftmaxRegressionL1``), ``'tree'``.
    param_grid : dict
        Maps hyperparameter names to iterables of candidate values, e.g.
        ``{'C': [0.1, 1], 'gamma': ['scale', 0.01]}`` for SVM.
    X, y : arrays
        Full feature matrix and integer class labels.
    K_out, K_in : int
        Number of outer and inner stratified folds.
    num_classes : int
        Required for linear softmax training (one-hot size).
    random_state : int
        Base seed for fold shuffling; inner folds use derived seeds per outer fold.

    Returns
    -------
    dict with keys:
        ``outer_fold_errors`` : list of length K_out (misclassification rate)
        ``chosen_hyperparameters`` : list of dicts, best params per outer fold
        ``outer_fold_indices`` : list of (test_idx,) arrays for traceability
        ``mean_outer_error``, ``std_outer_error``
    """
    if K_out < 2 or K_in < 2:
        raise ValueError("K_out and K_in must be at least 2.")

    combos = _param_grid_product(param_grid)
    if not combos:
        raise ValueError("param_grid must contain at least one hyperparameter with values.")

    outer_errors: List[float] = []
    chosen_hyperparameters: List[Dict[str, Any]] = []
    outer_test_indices: List[np.ndarray] = []

    print(f"\nRunning nested cross-validation for {model_name} with {len(combos)} combinations...")

    # loop 2 in our word document
    # The following loop enables to run over all partitions of the data into K_out outerfolds,
    # using the sklearn.model_selection class which splits the data in a clever way (uniform label distribution, invariance for data labeling etc.)
    skf_out = StratifiedKFold(n_splits=K_out, shuffle=True, random_state=random_state)
    for fold_out, (idx_outer_train, idx_outer_test) in tqdm(
        enumerate(skf_out.split(X, y)), total=K_out
    ):
        X_ot, y_ot = X[idx_outer_train], y[idx_outer_train]
        X_oe, y_oe = X[idx_outer_test], y[idx_outer_test]

        best_mean_inner = np.inf
        best_params: Optional[Dict[str, Any]] = None

        print(f"Running inner cross-validation for outer fold {fold_out+1}/{K_out} with {len(combos)} models M_i...")
        
        # Another use of StratifiedKFold for automatic and clever partition of the train fold (X_ot, y_ot) to K_in inner folds.
        skf_in = StratifiedKFold(n_splits=K_in, shuffle=True, random_state=random_state + 1000 * (fold_out + 1))

        # Iterating over "combos" - the models M_i (each with a different combo of hyperparameters) 
        for counter, params in enumerate(combos):
            print(f"Running M_{counter+1} with params {params}...", end="\t")
            start_time = time.time()
            inner_errors: List[float] = []

            for idx_in_train, idx_in_val in skf_in.split(X_ot, y_ot):
                X_in_tr, y_in_tr = X_ot[idx_in_train], y_ot[idx_in_train]
                X_in_va, y_in_va = X_ot[idx_in_val], y_ot[idx_in_val]

                # Training M_i over the inner training fold.
                y_pred = _fit_predict(
                    model_name,
                    params,
                    X_in_tr,
                    y_in_tr,
                    X_in_va,
                    num_classes,
                    tree_random_seed=random_state + fold_out * 7919 + len(inner_errors),
                )
                
                # getting \varepsilon_{ij} for hypothesis h_{ij} trained over the inner 
                # training fold (Ramot wrote it).
                inner_errors.append(_misclassification_error(y_in_va, y_pred))

            # Estimating generalization error - this is exactly 2(c)(ii) in our word document.
            mean_inner = float(np.mean(inner_errors))

            # looking for the model with the best generalization error for the current outer training fold, used later for the model selection in 2(d) in our word document.
            if mean_inner < best_mean_inner:
                best_mean_inner = mean_inner
                best_params = dict(params)

            print(f"finished in {time.time() - start_time:.2f} seconds | mean inner error = {mean_inner:.4f}")

        # Raising error of best_params was None (no best model was found for the current outer training fold).
        assert best_params is not None

        # 2(e) in the word document - fitting the best hypothesis H_i := M_i over all outer training fold (2e).
        y_hat_outer = _fit_predict(
            model_name,
            best_params,
            X_ot,
            y_ot,
            X_oe,
            num_classes,
            tree_random_seed=random_state + fold_out * 4999,
        )
    
        # 2(f) Testing H_i over the outer test fold and receive the outer error.
        outer_errors.append(_misclassification_error(y_oe, y_hat_outer))
        mean_outer_error = float(np.mean(outer_errors))
        std_outer_error = float(np.std(outer_errors, ddof=1)) if K_out > 1 else 0.0
        chosen_hyperparameters.append(best_params)
        outer_test_indices.append(np.asarray(idx_outer_test))
        print(f"finished in {time.time() - start_time:.2f} seconds | mean outer error = {mean_outer_error:.4f}")

    result = {
        "model_name": model_name,
        "K_out": K_out,
        "K_in": K_in,
        "outer_fold_errors": outer_errors,
        "chosen_hyperparameters": chosen_hyperparameters,
        "outer_fold_test_indices": outer_test_indices,
        "mean_outer_error": mean_outer_error,
        "std_outer_error": std_outer_error,
    }
    print(result)
    return result


def compare_models_nested_cv(
    model_specs: Mapping[str, Tuple[str, Mapping[str, Iterable[Any]]]],
    X: np.ndarray,
    y: np.ndarray,
    K_out: int,
    K_in: int,
    num_classes: int,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Run :func:`nested_cross_validate` for several models and summarize.

    ``model_specs`` maps a label (e.g. ``'SVM'``) to
    ``(model_name_for_nested_cv, param_grid)``.

    Returns per-model nested CV results plus a simple overall ranking by mean
    outer error and, if SciPy is available, paired tests on outer-fold errors.
    """
    results: Dict[str, Any] = {}
    for label, (mname, grid) in model_specs.items():
        results[label] = nested_cross_validate(
            mname, grid, X, y, K_out, K_in, num_classes, random_state=random_state
        )
        # append results to a txt file
        with open(f"results_partD.txt", "a") as f:
            for key, value in results[label].items():
                f.write(f"{key}: {value}\n")
            f.write("\n")

    ranking = sorted(
        results.items(),
        key=lambda kv: kv[1]["mean_outer_error"],
    )

    comparison: Dict[str, Any] = {
        "per_model": results,
        "ranking_by_mean_outer_error": [name for name, _ in ranking],
    }

    labels = list(results.keys())
    errors_matrix = np.array([results[l]["outer_fold_errors"] for l in labels])
    best_idx = int(np.argmin([results[l]["mean_outer_error"] for l in labels]))
    best_label = labels[best_idx]

    comparison["overall_recommendation"] = (
        f"Lowest mean nested-CV error: {best_label} "
        f"(mean={results[best_label]['mean_outer_error']:.4f}, "
        f"std={results[best_label]['std_outer_error']:.4f}). "
        "Use paired_tests to see whether differences across outer folds are significant."
    )
    return comparison


def print_nested_cv_summary(ncv: Mapping[str, Any]) -> None:
    print(f"\n=== Nested CV: {ncv['model_name']} (K_out={ncv['K_out']}, K_in={ncv['K_in']}) ===")
    print(f"Mean outer error: {ncv['mean_outer_error']:.4f} +/- {ncv['std_outer_error']:.4f}")
    for i, (err, params) in enumerate(zip(ncv["outer_fold_errors"], ncv["chosen_hyperparameters"]), 1):
        print(f"  Outer fold {i}: error={err:.4f} | best params={params}")



if __name__ == "__main__":
    X_full, y_full, n_cls = load_full_partc_features()

    # calculate the value of gamma='scale' for the SVM model
    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
    svm_model.fit(X_full, y_full)
    print(f"Gamma value for gamma='scale': {svm_model.gamma}")

    n_cls = 50
    print(X_full.shape)
    gamma_theoretical = 1 / (n_cls * np.var(X_full))
    print(f"Theoretical gamma value: {gamma_theoretical}")

    svm_grid = {
        "C": [1.0, 10.0, 100.0, 1000.0, 10000.0],
        "gamma": [1e-4, 1e-2, 1, 100, 'scale'],
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
    specs = {"SVM": ("SVM", svm_grid), "linear_reg": ("linear_reg", linear_grid), "tree": ("tree", tree_grid)}
    K_OUT, K_IN = 5, 3
    selected_label = "SVM"  # Change to: "SVM", "linear_reg", or "tree"
    selected_model_name, selected_grid = specs[selected_label]

    result = nested_cross_validate(
        model_name=selected_model_name,
        param_grid=selected_grid,
        X=X_full,
        y=y_full,
        K_out=K_OUT,
        K_in=K_IN,
        num_classes=n_cls,
        random_state=42,
    )
    print_nested_cv_summary(result)