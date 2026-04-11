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
X_full, y_full_str = part_A.process_and_inspect(TRAIN_DIR)
le = LabelEncoder()
y_full = le.fit_transform(y_full_str)
num_classes = len(le.classes_)

# One-hot encoding for the fit method
y_train_encoded = np.eye(num_classes)[y_train]

# 3. Sweep over 5 values of 'reg'
reg_params = [0.0001, 0.01, 0.5, 5, 50]
train_results = []
val_results = []

print("\nStarting Part C Analysis...")
# using tqdm progress bar for better visualization of the process
from tqdm import tqdm
for r in tqdm(reg_params, desc="Regularization Sweep"):
    model = part_B.MySoftmaxRegressionL1(lr=0.05, epochs=100, reg=r)
    model.fit(X_train, y_train_encoded, num_classes)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    val_acc = accuracy_score(y_val, model.predict(X_val))

    train_results.append(train_acc)
    val_results.append(val_acc)
    tqdm.write(f"Reg: {r:<7} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

# 4. Plotting the Complexity vs Performance graph
plt.figure(figsize=(10, 6))
plt.plot(reg_params, train_results, 'o-', label='Training Accuracy (Complexity High)')
plt.plot(reg_params, val_results, 's-', label='Validation Accuracy (Generalization)')
plt.xscale('log')
plt.xlabel('Regularization Strength (Higher = Simpler Model)')
plt.ylabel('Accuracy')
plt.title('Part C: Effect of Regularization on Performance')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.show()