import zipfile
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import part_A
import part_B


# --- PART C: REGULARIZATION ANALYSIS ---
# --- DATA PATHS ---
TRAIN_DIR = r'data\Train Set (Labeled)'

# 1. Load and Encode Data
X_full, y_full_str = part_A.process_optimized(TRAIN_DIR)
le = LabelEncoder()
y_full = le.fit_transform(y_full_str)
num_classes = len(le.classes_)

# 2. Split for Validation (Crucial for detecting Overfitting)
X_train, X_val, y_train, y_val = train_test_split(X_full, y_full, test_size=0.25, random_state=42)

# One-hot encoding for the fit method
y_train_encoded = np.eye(num_classes)[y_train]


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
    print(f"\nC: {c:<5} | Train Acc: {train_accs_svm[-1]:.4f} | Val Acc: {val_accs_svm[-1]:.4f}")
    tqdm.write(f"\nC: {c:<5} | Train Acc: {train_accs_svm[-1]:.4f} | Val Acc: {val_accs_svm[-1]:.4f}")

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