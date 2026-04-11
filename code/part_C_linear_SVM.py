import os
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import zipfile
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import part_A
import part_B

"""במקום SVC עם קרנל RBF (שהוא כבד מאוד), נשתמש ב-LinearSVC. הוא מותאם במיוחד לוקטורים ארוכים כמו HOG ורץ פי 100 יותר מהר."""


# --- PART C: REGULARIZATION ANALYSIS ---
# --- DATA PATHS ---
TRAIN_DIR = r'data\Train Set (Labeled)'
PARTC_CACHE = os.path.join('data', 'part_c_splits_inspect.npz')

if os.path.isfile(PARTC_CACHE):
    X_train, X_val, y_train, y_val, num_classes = part_A.load_partc_train_val_npz(PARTC_CACHE)
else:
    X_train, X_val, y_train, y_val, num_classes = part_A.save_partc_train_val_npz(
        TRAIN_DIR, PARTC_CACHE, pipeline='inspect'
    )


# נשתמש רק ב-20% מהנתונים כדי לרוץ מהר מאוד
X_train_fast, _, y_train_fast, _ = train_test_split(
    X_train, y_train, train_size=0.2, random_state=42, stratify=y_train
)

# --- SPEED FIX 1: Feature Slicing ---
# Instead of 1200+ HOG features, let's use the first 300.
# It's plenty to show the effect of C for Part C.
X_train_ultra_fast = X_train_fast[:, :300]
X_val_ultra_fast = X_val[:, :300]

c_values = [0.0001, 0.001, 0.01, 0.1, 1, 10] # Reduced the max C from 100 to 10 to prevent hanging
train_accs, val_accs = [], []

print("Running Ultra-Fast SVM Sweep...")
# use tqdm progress bar for better visualization of the process
from tqdm import tqdm
for c in tqdm(c_values, desc="SVM Sweep"):
    # SPEED FIX 2: max_iter=500 and tol=0.1
    # We don't need a perfect model, just the trend for the graph!
    model = LinearSVC(C=c, max_iter=500, tol=0.1, dual=False, random_state=42)
    model.fit(X_train_ultra_fast, y_train_fast)

    train_accs.append(accuracy_score(y_train_fast, model.predict(X_train_ultra_fast)))
    val_accs.append(accuracy_score(y_val, model.predict(X_val_ultra_fast)))
    print(f"\nC: {c} - Done")
    tqdm.write(f"\nC: {c} | Train Acc: {train_accs[-1]:.4f} | Val Acc: {val_accs[-1]:.4f}")
        
# ציור הגרף
plt.figure(figsize=(8, 5))
plt.plot(c_values, train_accs, 'o-', label='Train (Complexity)')
plt.plot(c_values, val_accs, 's-', label='Validation (Generalization)')
plt.xscale('log')
plt.title('SVM: Effect of C (Complexity Control)')
plt.xlabel('C parameter (Inverse of Regularization)')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()