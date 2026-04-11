import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode

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

# --- Execution & Evaluation ---

# Generate some dummy data (3 clusters)
np.random.seed(42)
cluster1 = np.random.normal(0, 1, (50, 5))
cluster2 = np.random.normal(5, 1, (50, 5))
cluster3 = np.random.normal(10, 1, (50, 5))
X = np.vstack([cluster1, cluster2, cluster3])
y_true = np.array([0]*50 + [1]*50 + [2]*50)

# Run KMeans
model = SimpleKMeans(k=3)
y_kmeans = model.fit(X)

# --- Analytical Correlation (Accuracy) ---
# Since KMeans labels are arbitrary (Cluster 0 might be Label 2), 
# we map them to the most frequent true label in each cluster.
mapped_labels = np.zeros_like(y_kmeans)
for i in range(3):
    mask = (y_kmeans == i)
    if np.any(mask):
        mapped_labels[mask] = mode(y_true[mask], keepdims=True)[0][0]

accuracy = np.mean(mapped_labels == y_true)
print(f"Analytical Accuracy (mapped): {accuracy * 100:.2f}%")

# --- Plotting ---
X_2d = get_pca(X)
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_true, cmap='viridis', alpha=0.6)
plt.title("True Labels (PCA Projection)")

plt.subplot(1, 2, 2)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_kmeans, cmap='plasma', alpha=0.6)
plt.title(f"K-Means Clusters (Acc: {accuracy*100:.1f}%)")

plt.tight_layout()
plt.show()