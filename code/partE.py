import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode

import part_A

class SimpleKMeans:
    def __init__(self, k=28, max_iters=1000):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None

    def fit(self, X):
        # 1. Initialize centroids randomly from the data points
        idx = np.random.choice(len(X), self.k, replace=False)
        self.centroids = X[idx]

        for _ in range(self.max_iters):
            # 2. Assign each point to the nearest centroid
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)

            # 3. Update centroids to be the mean of assigned points
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.k)])
            
            # Check for convergence
            if np.allclose(self.centroids, new_centroids, atol=1e-6):
                break
            self.centroids = new_centroids
        
        return labels

def get_pca(X, dims=2):
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    # Covariance matrix
    cov_matrix = np.cov(X_centered, rowvar=False)
    # Eigen-decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    # Sort and pick top vectors
    idx = np.argsort(eigenvalues)[::-1]
    top_vectors = eigenvectors[:, idx[:dims]]
    return X_centered @ top_vectors

import os
PARTC_CACHE = os.path.join('data', 'part_c_splits_optimized.npz')
def perform_k_means():
    # loading image data from part A
    X_train, X_val, y_train, y_val, num_classes = part_A.load_partc_train_val_npz(PARTC_CACHE)

    # gathering all the data into one array
    X = np.concatenate([X_train, X_val])
    y_true = np.concatenate([y_train, y_val])

    # performing k-means
    model = SimpleKMeans(k=28, max_iters=1000)
    y_kmeans = model.fit(X)

    mapped_labels = np.zeros_like(y_kmeans)
    for i in range(28):
        # setting mask to be the index of the K-means group
        mask = (y_kmeans == i)
        if np.any(mask):
            # setting the K-means group to the most common (true) label in the group
            mapped_labels[mask] = mode(y_true[mask], keepdims=True)[0][0]

    # Checking the percentage of true labels found by the K-means model
    accuracy = np.mean(mapped_labels == y_true)
    print(f"Analytical Accuracy (mapped): {accuracy * 100:.2f}%")

    # randomly choose a label for each K-means group
    # for i in range(28):
    #     mask = (y_kmeans == i)
    #     if np.any(mask):
    #         mapped_labels[mask] = np.random.choice(y_true[mask])

    # # Checking the percentage of true labels found by the K-means model
    # accuracy = np.mean(mapped_labels == y_true)
    # print(f"Random Accuracy (mapped): {accuracy * 100:.2f}%")

    # perform another PCA to reduce X features to 2 dimensions
    X_pca = get_pca(X, dims=2)
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap='rainbow', alpha=0.6)
    plt.title("True Labels (PCA Projection)")

    plt.subplot(1, 2, 2)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, cmap='rainbow', alpha=0.6)
    plt.title(f"K-Means Clusters (Acc: {accuracy*100:.1f}%)")

    plt.tight_layout()
    plt.show()

perform_k_means()