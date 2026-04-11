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
TRAIN_ZIP = r'C:\Users\ariel\PycharmProjects\TECHNOTZ DS TARGIL 2\Train Set (Labeled)-20260407T132757Z-3-001.zip'

# 1. Load and Encode Data
X_full, y_full_str = part_A.process_and_inspect(TRAIN_ZIP)
le = LabelEncoder()
y_full = le.fit_transform(y_full_str)
num_classes = len(le.classes_)

# 2. Split for Validation (Crucial for detecting Overfitting)
X_train, X_val, y_train, y_val = train_test_split(X_full, y_full, test_size=0.25, random_state=42)

# One-hot encoding for the fit method
y_train_encoded = np.eye(num_classes)[y_train]

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
