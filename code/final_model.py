from __future__ import annotations

import os
import pickle
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import part_A

svm_best_params = {"C": 10.0, "gamma": "scale"}

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_CODE_DIR)
TRAIN_DIR = os.path.join(_REPO_ROOT, "data", "Train Set (Labeled)")
TEST_DIR = os.path.join(_REPO_ROOT, "data", "Test Set (Unlabeled)")
PARTC_CACHE = os.path.join(_REPO_ROOT, "data", "part_c_splits_optimized.npz")
MODEL_BUNDLE_PATH = os.path.join(_REPO_ROOT, "data", "final_svm_bundle.pkl")
RESULTS_CSV_PATH = os.path.join(_REPO_ROOT, "results_manela_and_ramot.csv")
N_COMPONENTS = 50


def _load_labeled_xy_from_partc_cache(
    train_dir: str = TRAIN_DIR,
    npz_path: str = PARTC_CACHE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full labeled pool in Part C feature space (PCA on HOG), with integer labels
    and label_classes for decoding predictions.
    """
    if os.path.isfile(npz_path):
        data = np.load(npz_path, allow_pickle=True)
        X_tr = data["X_train"]
        X_va = data["X_val"]
        y_tr = data["y_train"]
        y_va = data["y_val"]
        label_classes = data["label_classes"]
    else:
        part_A.save_partc_train_val_npz(train_dir, npz_path, pipeline="optimized", n_components=N_COMPONENTS)
        data = np.load(npz_path, allow_pickle=True)
        X_tr = data["X_train"]
        X_va = data["X_val"]
        y_tr = data["y_train"]
        y_va = data["y_val"]
        label_classes = data["label_classes"]

    X = np.vstack([X_tr, X_va])
    y = np.concatenate([y_tr, y_va])
    return X, y, np.asarray(label_classes)


def _fit_scaler_pca_on_train_hog(
    train_dir: str = TRAIN_DIR,
    n_components: int = N_COMPONENTS,
) -> Tuple[StandardScaler, PCA]:
    """Fit StandardScaler + PCA on raw HOG from labeled set (same pipeline as part_c cache)."""
    X_raw, _ = part_A.extract_optimized_hog_features(train_dir)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    pca = PCA(n_components=min(n_components, X_scaled.shape[0], X_scaled.shape[1]))
    pca.fit(X_scaled)
    return scaler, pca


def _load_or_fit_preprocessing(
    train_dir: str = TRAIN_DIR,
    bundle_path: str = MODEL_BUNDLE_PATH,
) -> Tuple[StandardScaler, PCA]:
    """
    Reuse scaler+pca from a previous bundle when present; otherwise fit once via part_A HOG.
    """
    if os.path.isfile(bundle_path):
        try:
            with open(bundle_path, "rb") as f:
                bundle = pickle.load(f)
            scaler = bundle.get("scaler")
            pca = bundle.get("pca")
            if isinstance(scaler, StandardScaler) and isinstance(pca, PCA):
                return scaler, pca
        except (pickle.UnpicklingError, EOFError, KeyError):
            pass

    scaler, pca = _fit_scaler_pca_on_train_hog(train_dir, n_components=N_COMPONENTS)
    return scaler, pca


def _transform_dir_hog_to_pca(
    data_dir: str,
    scaler: StandardScaler,
    pca: PCA,
) -> np.ndarray:
    """Raw HOG (part_A) then apply fitted scaler + PCA."""
    X_raw, _ = part_A.extract_optimized_hog_features(data_dir)
    return pca.transform(scaler.transform(X_raw))


def _label_strings_to_identity_ids(label_strings: np.ndarray) -> np.ndarray:
    return np.array([int(s.lstrip("p")) + 1 for s in label_strings], dtype=int)


def main() -> None:
    # Labeled features: precomputed Part C cache (no HOG/PCA recompute on this matrix).
    X_train, y_train, label_classes = _load_labeled_xy_from_partc_cache(TRAIN_DIR, PARTC_CACHE)

    # Scaler + PCA for unlabeled images only (reuse from bundle when available).
    scaler, pca = _load_or_fit_preprocessing(TRAIN_DIR, MODEL_BUNDLE_PATH)

    svm_model = SVC(kernel="rbf", C=svm_best_params["C"], gamma=svm_best_params["gamma"])
    svm_model.fit(X_train, y_train)

    y_pred_train = svm_model.predict(X_train)
    accuracy = float(np.mean(y_train == y_pred_train))
    print(f"Accuracy of the model over the training data: {accuracy:.6f}")

    X_unlabeled = _transform_dir_hog_to_pca(TEST_DIR, scaler, pca)
    y_pred_idx = svm_model.predict(X_unlabeled)
    pred_label_str = label_classes[np.asarray(y_pred_idx, dtype=int)]
    y_pred_numeric = _label_strings_to_identity_ids(pred_label_str)

    if y_pred_numeric.size != 392:
        print(f"Warning: expected 392 predictions, got {y_pred_numeric.size}.")

    bundle: Dict[str, Any] = {
        "model_name": "SVM",
        "svm_params": dict(svm_best_params),
        "model": svm_model,
        "scaler": scaler,
        "pca": pca,
        "label_classes": label_classes,
        "partc_cache": PARTC_CACHE,
    }
    os.makedirs(os.path.dirname(MODEL_BUNDLE_PATH), exist_ok=True)
    with open(MODEL_BUNDLE_PATH, "wb") as file_handle:
        pickle.dump(bundle, file_handle)

    np.savetxt(
        RESULTS_CSV_PATH,
        y_pred_numeric.reshape(1, -1),
        delimiter=",",
        fmt="%d",
    )

    print(f"Saved model bundle to: {MODEL_BUNDLE_PATH}")
    print(f"Saved predictions to: {RESULTS_CSV_PATH}")


if __name__ == "__main__":
    main()
