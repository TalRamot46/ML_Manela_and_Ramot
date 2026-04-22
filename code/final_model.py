from __future__ import annotations

import os
import pickle
from typing import List, Tuple

import cv2
import numpy as np
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

import part_A

svm_best_params = {"C": 100.0, "gamma": "scale"}

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_CODE_DIR)
TRAIN_DIR = os.path.join(_REPO_ROOT, "data", "Train Set (Labeled)")
TEST_DIR = os.path.join(_REPO_ROOT, "data", "Test Set (Unlabeled)")
MODEL_BUNDLE_PATH = os.path.join(_REPO_ROOT, "data", "final_svm_bundle.pkl")
RESULTS_CSV_PATH = os.path.join(_REPO_ROOT, "results_manela_and_ramot.csv")


def _extract_hog(data_dir: str) -> Tuple[np.ndarray, List[str]]:
    features = []
    filenames: List[str] = []
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    for filename, file_path, file_bytes in part_A._iter_pgm_files(data_dir):
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

        features.append(fd)
        filenames.append(os.path.basename(filename))

    return np.asarray(features), filenames


def _fit_preprocessing(train_dir: str, n_components: int = 50):
    X_raw, train_filenames = _extract_hog(train_dir)
    y_labels = np.array([os.path.splitext(name)[0].split("_")[0] for name in train_filenames])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    pca = PCA(n_components=min(n_components, X_scaled.shape[0], X_scaled.shape[1]))
    X_train = pca.fit_transform(X_scaled)

    return X_train, y_labels, scaler, pca


def _transform_unlabeled(test_dir: str, scaler: StandardScaler, pca: PCA):
    X_raw, test_filenames = _extract_hog(test_dir)
    X_scaled = scaler.transform(X_raw)
    X_test = pca.transform(X_scaled)
    return X_test, test_filenames


def main():
    X_train, y_labels, scaler, pca = _fit_preprocessing(TRAIN_DIR, n_components=50)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_labels)

    svm_model = SVC(kernel="rbf", C=svm_best_params["C"], gamma=svm_best_params["gamma"])
    svm_model.fit(X_train, y_encoded)

    # print the accuracy of the model over the training data
    y_pred = svm_model.predict(X_train)
    accuracy = np.mean(y_encoded == y_pred)
    print(f"Accuracy of the model over the training data: {accuracy}")

    bundle = {
        "model_name": "SVM",
        "svm_params": dict(svm_best_params),
        "model": svm_model,
        "scaler": scaler,
        "pca": pca,
        "label_encoder": label_encoder,
    }
    os.makedirs(os.path.dirname(MODEL_BUNDLE_PATH), exist_ok=True)
    with open(MODEL_BUNDLE_PATH, "wb") as file_handle:
        pickle.dump(bundle, file_handle)

    X_unlabeled, _ = _transform_unlabeled(TEST_DIR, scaler, pca)

    y_pred_encoded = svm_model.predict(X_unlabeled)
    y_pred_labels = label_encoder.inverse_transform(y_pred_encoded)
    y_pred_numeric = np.array([int(lbl.lstrip("p")) + 1 for lbl in y_pred_labels], dtype=int)

    if y_pred_numeric.size != 392:
        print(f"Warning: expected 392 predictions, got {y_pred_numeric.size}.")

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
