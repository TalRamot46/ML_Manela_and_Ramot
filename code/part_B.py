import numpy as np
from collections import Counter

class MySoftmaxRegression:
    def __init__(self, lr=0.01, epochs=1000, reg=0.1):
        self.lr = lr  # Learning Rate
        self.epochs = epochs
        self.reg = reg  # Regularization strength (L2)
        self.weights = None

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y_encoded, num_classes):
        # y_encoded צריך להיות בפורמט One-Hot
        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features, num_classes))

        for i in range(self.epochs):
            # 1. Forward Pass (חישוב הסתברויות)
            scores = np.dot(X, self.weights)
            probs = self._softmax(scores)

            # 2. חישוב הגרדיאנט (כולל L2 Regularization)
            gradient = (1 / n_samples) * np.dot(X.T, (probs - y_encoded))
            gradient += self.reg * self.weights  # רגולריזציה

            # 3. עדכון משקלים (Gradient Descent)
            self.weights -= self.lr * gradient

    def predict(self, X):
        scores = np.dot(X, self.weights)
        return np.argmax(self._softmax(scores), axis=1)

from sklearn.svm import SVC

# יצירת המודל
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')

# אימון
# svm_model.fit(X_train, y_train)

# חיזוי
# y_pred = svm_model.predict(X_test)

import numpy as np


class MyDecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth  # עומק מקסימלי (למניעת Overfitting)
        self.min_samples_split = min_samples_split  # מינימום דגימות לפיצול
        self.root = None

    def _entropy(self, y):
        # חישוב אנטרופיה (מדד לאי-סדר בקבוצה)
        proportions = np.bincount(y) / len(y)
        return -np.sum([p * np.log2(p) for p in proportions if p > 0])

    def _create_split(self, X, thresh):
        left_idx = np.argwhere(X <= thresh).flatten()
        right_idx = np.argwhere(X > thresh).flatten()
        return left_idx, right_idx

    def _information_gain(self, X_column, y, thresh):
        # חישוב כמה הפיצול הזה "ניקה" את המידע
        parent_entropy = self._entropy(y)
        left_idx, right_idx = self._create_split(X_column, thresh)

        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(left_idx), len(right_idx)
        e_l, e_r = self._entropy(y[left_idx]), self._entropy(y[right_idx])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        return parent_entropy - child_entropy

    def _best_split(self, X, y):
        best_gain = -1
        split_idx, split_thresh = None, None

        # קיצור דרך: נבדוק רק 30 תכונות אקראיות מתוך ה-HOG
        n_features = X.shape[1]
        feature_indices = np.random.choice(n_features, min(30, n_features), replace=False)

        # עוברים על כל התכונות (HOG) ומוצאים את הפיצול הכי טוב
        for i in feature_indices:
            X_column = X[:, i]
            thresholds = np.unique(X_column)
            for thresh in thresholds:
                gain = self._information_gain(X_column, y, thresh)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = i
                    split_thresh = thresh
        return split_idx, split_thresh

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # תנאי עצירה (עומק או מחלקה אחת בלבד)
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            most_common = np.bincount(y).argmax()
            return {'leaf': True, 'class': most_common}

        # מציאת הפיצול האופטימלי
        idx, thresh = self._best_split(X, y)
        left_idx, right_idx = self._create_split(X[:, idx], thresh)

        left_node = self._build_tree(X[left_idx, :], y[left_idx], depth + 1)
        right_node = self._build_tree(X[right_idx, :], y[right_idx], depth + 1)

        return {'leaf': False, 'index': idx, 'threshold': thresh, 'left': left_node, 'right': right_node}

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node['leaf']:
            return node['class']
        if x[node['index']] <= node['threshold']:
            return self._traverse_tree(x, node['left'])
        return self._traverse_tree(x, node['right'])


#softmax with L1
class MySoftmaxRegressionL1:
    def __init__(self, lr=0.01, epochs=1000, reg=0.1):
        self.lr = lr
        self.epochs = epochs
        self.reg = reg
        self.weights = None

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y_encoded, num_classes):
        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features, num_classes))

        for i in range(self.epochs):
            scores = np.dot(X, self.weights)
            probs = self._softmax(scores)

            # חישוב הגרדיאנט הבסיסי
            gradient = (1 / n_samples) * np.dot(X.T, (probs - y_encoded))

            # --- הוספת L1 Regularization ---
            # אנחנו מוסיפים את הסימן של המשקלים כפול מקדם הרגולריזציה
            gradient += self.reg * np.sign(self.weights)

            self.weights -= self.lr * gradient

    def predict(self, X):
        scores = np.dot(X, self.weights)
        return np.argmax(self._softmax(scores), axis=1)