import os
import zipfile
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import part_A
import part_B


# --- PART C: REGULARIZATION ANALYSIS ---
# --- DATA PATHS ---
TRAIN_DIR = r'data\Train Set (Labeled)'
PARTC_CACHE = os.path.join('data', 'part_c_splits_optimized.npz')

if os.path.isfile(PARTC_CACHE):
    X_train, X_val, y_train, y_val, num_classes = part_A.load_partc_train_val_npz(PARTC_CACHE)
else:
    X_train, X_val, y_train, y_val, num_classes = part_A.save_partc_train_val_npz(
        TRAIN_DIR, PARTC_CACHE, pipeline='optimized'
    )


from sklearn.svm import SVC

# --- SVM SWEEP ---
c_values = [0.001, 0.1, 1, 10, 100]
train_accs_svm, val_accs_svm = [], []

print("\nStarting SVM Sweep...")
# use tqdm progress bar for better visualization of the process
from tqdm import tqdm
for c in tqdm(c_values, desc="SVM Sweep"):
    # Using sklearn's SVC as per your Part B
    svm_model = SVC(kernel='rbf', C=c, gamma='scale')
    svm_model.fit(X_train, y_train)

    train_accs_svm.append(accuracy_score(y_train, svm_model.predict(X_train)))
    val_accs_svm.append(accuracy_score(y_val, svm_model.predict(X_val)))
    tqdm.write(f"C: {c:<5} | Train Acc: {train_accs_svm[-1]:.4f} | Val Acc: {val_accs_svm[-1]:.4f}")

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(c_values, train_accs_svm, 'bo-', label='Train Accuracy')
plt.plot(c_values, val_accs_svm, 'ro-', label='Validation Accuracy')
plt.xscale('log')
plt.xlabel('C (Inverse of Regularization)')
plt.ylabel('Accuracy')
plt.title('Part C: SVM Complexity (C parameter)')
plt.legend()
plt.grid(True)
plt.show()