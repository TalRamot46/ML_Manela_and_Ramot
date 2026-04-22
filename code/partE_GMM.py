import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mode
from sklearn.mixture import GaussianMixture

import part_A


def get_pca(X, dims=2):
    X_centered = X - np.mean(X, axis=0)
    cov_matrix = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    top_vectors = eigenvectors[:, idx[:dims]]
    return X_centered @ top_vectors


PARTC_CACHE = os.path.join("data", "part_c_splits_optimized.npz")


def perform_gmm():
    X_train, X_val, y_train, y_val, num_classes = part_A.load_partc_train_val_npz(PARTC_CACHE)

    X = np.concatenate([X_train, X_val])
    y_true = np.concatenate([y_train, y_val])

    n_components = int(num_classes)
    model = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=42,
        n_init=3,
    )
    y_GMM = model.fit_predict(X)

    mapped_labels = np.zeros_like(y_GMM)
    for i in range(n_components):
        mask = y_GMM == i
        if np.any(mask):
            mapped_labels[mask] = mode(y_true[mask], keepdims=True)[0][0]

    accuracy = np.mean(mapped_labels == y_true)
    print(f"GMM Accuracy (mapped): {accuracy * 100:.2f}%")

    X_pca = get_pca(X, dims=2)
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap="rainbow", alpha=0.6)
    plt.title("True Labels (PCA Projection)")

    plt.subplot(1, 2, 2)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_GMM, cmap="rainbow", alpha=0.6)
    plt.title(f"GMM Components (Acc: {accuracy * 100:.1f}%)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    perform_gmm()
