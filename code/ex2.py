from __future__ import annotations

import os
import pickle
from itertools import product
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import cv2
import numpy as np
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


CODE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CODE_DIR)
TRAIN_DIR = os.path.join(REPO_ROOT, "data", "Train Set (Labeled)")
TEST_DIR = os.path.join(REPO_ROOT, "data", "Test Set (Unlabeled)")
RESULTS_CSV_PATH = os.path.join(REPO_ROOT, "results_manela_and_ramot.csv")
MODEL_BUNDLE_PATH = os.path.join(REPO_ROOT, "data", "ex2_final_svm_bundle.pkl")
N_COMPONENTS = 50


def iter_pgm_files(data_path: str):
    for filename in sorted(os.listdir(data_path)):
        if filename.lower().endswith(".pgm"):
            yield filename, os.path.join(data_path, filename)


def extract_hog_features(data_path: str, *, resize_shape: Tuple[int, int] = (64, 64)):
    features: List[np.ndarray] = []
    filenames: List[str] = []
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    for filename, file_path in iter_pgm_files(data_path):
        with open(file_path, "rb") as f:
            img_data = f.read()
        img_array = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img_small = cv2.resize(img, resize_shape)
        img_clahe = clahe.apply(img_small)
        fd = hog(
            img_clahe,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=False,
        )
        features.append(fd)
        filenames.append(filename)

    return np.asarray(features), filenames


def fit_preprocessing_labeled(train_dir: str, n_components: int = N_COMPONENTS):
    X_raw, filenames = extract_hog_features(train_dir)
    y_str = np.array([os.path.splitext(name)[0].split("_")[0] for name in filenames])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    pca = PCA(n_components=min(n_components, X_scaled.shape[0], X_scaled.shape[1]))
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, y_str, scaler, pca


def transform_with_fitted_preprocessing(data_dir: str, scaler: StandardScaler, pca: PCA):
    X_raw, filenames = extract_hog_features(data_dir)
    X_scaled = scaler.transform(X_raw)
    X_pca = pca.transform(X_scaled)
    return X_pca, filenames


def misclassification_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return 1.0 - accuracy_score(y_true, y_pred)


def param_grid_product(param_grid: Mapping[str, Iterable[Any]]) -> List[Dict[str, Any]]:
    keys = list(param_grid.keys())
    return [dict(zip(keys, vals)) for vals in product(*[param_grid[k] for k in keys])]


def nested_cv_svm(
    X: np.ndarray,
    y: np.ndarray,
    param_grid: Mapping[str, Iterable[Any]],
    *,
    k_out: int = 5,
    k_in: int = 3,
    random_state: int = 42,
):
    combos = param_grid_product(param_grid)
    outer = StratifiedKFold(n_splits=k_out, shuffle=True, random_state=random_state)

    outer_errors: List[float] = []
    chosen_params: List[Dict[str, Any]] = []

    for fold_out, (idx_train_outer, idx_test_outer) in enumerate(outer.split(X, y), start=1):
        X_train_outer, y_train_outer = X[idx_train_outer], y[idx_train_outer]
        X_test_outer, y_test_outer = X[idx_test_outer], y[idx_test_outer]

        inner = StratifiedKFold(
            n_splits=k_in,
            shuffle=True,
            random_state=random_state + 1000 * fold_out,
        )

        best_params = None
        best_inner_mean = np.inf
        for params in combos:
            inner_errors: List[float] = []
            for idx_train_inner, idx_val_inner in inner.split(X_train_outer, y_train_outer):
                X_train_inner, y_train_inner = X_train_outer[idx_train_inner], y_train_outer[idx_train_inner]
                X_val_inner, y_val_inner = X_train_outer[idx_val_inner], y_train_outer[idx_val_inner]

                model = SVC(kernel="rbf", **params)
                model.fit(X_train_inner, y_train_inner)
                y_pred_inner = model.predict(X_val_inner)
                inner_errors.append(misclassification_error(y_val_inner, y_pred_inner))

            mean_inner = float(np.mean(inner_errors))
            if mean_inner < best_inner_mean:
                best_inner_mean = mean_inner
                best_params = dict(params)

        assert best_params is not None
        final_outer_model = SVC(kernel="rbf", **best_params)
        final_outer_model.fit(X_train_outer, y_train_outer)
        y_pred_outer = final_outer_model.predict(X_test_outer)
        outer_errors.append(misclassification_error(y_test_outer, y_pred_outer))
        chosen_params.append(best_params)

    mean_outer_error = float(np.mean(outer_errors))
    return {
        "outer_fold_errors": outer_errors,
        "chosen_hyperparameters": chosen_params,
        "mean_outer_error": mean_outer_error,
    }


def pick_final_params_from_nested_cv(result: Mapping[str, Any]) -> Dict[str, Any]:
    params_list = result["chosen_hyperparameters"]
    if not params_list:
        raise ValueError("No parameters found in nested CV result.")
    counts: Dict[Tuple[Any, Any], int] = {}
    for p in params_list:
        key = (p["C"], p["gamma"])
        counts[key] = counts.get(key, 0) + 1
    best_key = max(counts.items(), key=lambda kv: kv[1])[0]
    return {"C": best_key[0], "gamma": best_key[1]}


def label_to_identity_number(label: str) -> int:
    return int(label.lstrip("p")) + 1


def main():
    # 1) Load + preprocess labeled data
    X_labeled, y_labeled_str, scaler, pca = fit_preprocessing_labeled(TRAIN_DIR, n_components=N_COMPONENTS)
    label_encoder = LabelEncoder()
    y_labeled = label_encoder.fit_transform(y_labeled_str)

    # 2) Nested CV + hyperparameter tuning (SVM)
    svm_grid = {
        "C": [1.0, 10.0, 100.0, 1000.0, 10000.0],
        "gamma": [1e-4, 1e-2, 1, 100, "scale"],
    }
    nested_result = nested_cv_svm(X_labeled, y_labeled, svm_grid, k_out=5, k_in=3, random_state=42)
    final_params = pick_final_params_from_nested_cv(nested_result)
    print(f"Nested-CV mean outer error: {nested_result['mean_outer_error']:.6f}")
    print(f"Chosen final SVM params: {final_params}")

    # 3) Train final model on all labeled data
    final_model = SVC(kernel="rbf", **final_params)
    final_model.fit(X_labeled, y_labeled)

    # 4) Save final model + preprocessing artifacts
    bundle = {
        "model_name": "SVM",
        "svm_params": final_params,
        "model": final_model,
        "scaler": scaler,
        "pca": pca,
        "label_encoder": label_encoder,
        "nested_cv_result": nested_result,
    }
    os.makedirs(os.path.dirname(MODEL_BUNDLE_PATH), exist_ok=True)
    with open(MODEL_BUNDLE_PATH, "wb") as f:
        pickle.dump(bundle, f)

    # 5) Final prediction on unlabeled set
    X_unlabeled, _ = transform_with_fitted_preprocessing(TEST_DIR, scaler, pca)
    y_pred_encoded = final_model.predict(X_unlabeled)
    y_pred_labels = label_encoder.inverse_transform(y_pred_encoded)
    y_pred_identity = np.array([label_to_identity_number(lbl) for lbl in y_pred_labels], dtype=int)

    if y_pred_identity.size != 392:
        print(f"Warning: expected 392 predictions, got {y_pred_identity.size}.")

    # single-row CSV: 1x392
    np.savetxt(RESULTS_CSV_PATH, y_pred_identity.reshape(1, -1), delimiter=",", fmt="%d")
    print(f"Saved final predictions CSV to: {RESULTS_CSV_PATH}")
    print(f"Saved model bundle to: {MODEL_BUNDLE_PATH}")


if __name__ == "__main__":
    main()
