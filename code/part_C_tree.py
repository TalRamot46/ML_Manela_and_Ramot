import os
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


# --- PART C: REGULARIZATION ANALYSIS ---
# --- DATA PATHS ---
TRAIN_ZIP = r'data\Train Set (Labeled)'
PARTC_CACHE = os.path.join('data', 'part_c_splits_inspect.npz')

if os.path.isfile(PARTC_CACHE):
    X_train, X_val, y_train, y_val, num_classes = part_A.load_partc_train_val_npz(PARTC_CACHE)
else:
    X_train, X_val, y_train, y_val, num_classes = part_A.save_partc_train_val_npz(
        TRAIN_ZIP, PARTC_CACHE, pipeline='inspect'
    )

# --- הגדרת Sweep מהיר במיוחד ---
# 1. צמצום נתונים (רק 15% מהמידע)
X_train_ultra, _, y_train_ultra, _ = train_test_split(
    X_train, y_train, train_size=0.15, random_state=42, stratify=y_train
)

# 2. הוספת "קיצור דרך" למחלקה שלך (ללא שינוי המבנה שלה)
# אנחנו נשתמש בגרסה שבודקת רק חלק קטן מהתכונות
depths = [1, 2, 4, 6, 10]
train_accs_tree = []
val_accs_tree = []

print("Running Fast Tree Sweep (Part C)...")

for d in depths:
    # הגדלת min_samples_split ל-50 תגרום לעץ לרוץ הרבה יותר מהר
    model = part_B.MyDecisionTree(max_depth=d, min_samples_split=50)

    # אימון על הנתונים המצומצמים
    model.fit(X_train_ultra, y_train_ultra)

    # חיזוי
    train_accs_tree.append(accuracy_score(y_train_ultra, model.predict(X_train_ultra)))
    val_accs_tree.append(accuracy_score(y_val, model.predict(X_val)))
    print(f"Depth {d} completed.")

# ציור הגרף
plt.figure(figsize=(8, 5))
plt.plot(depths, train_accs_tree, 'o-', label='Train Accuracy')
plt.plot(depths, val_accs_tree, 's-', label='Validation Accuracy')
plt.title('Decision Tree: Complexity vs Generalization')
plt.xlabel('Max Depth (Complexity Control)')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
