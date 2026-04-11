import os
import zipfile
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler


def _iter_pgm_files(data_path):
    if os.path.isdir(data_path):
        for filename in sorted(os.listdir(data_path)):
            if filename.lower().endswith('.pgm'):
                yield filename, os.path.join(data_path, filename), None
    else:
        with zipfile.ZipFile(data_path, 'r') as archive:
            for filename in sorted(name for name in archive.namelist() if name.lower().endswith('.pgm')):
                yield filename, None, archive.read(filename)


def process_and_inspect(zip_path):
    features_list = []
    labels = []

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    for i, (filename, file_path, file_bytes) in enumerate(_iter_pgm_files(zip_path)):
        # Read and decode
        if file_bytes is not None:
            img_data = file_bytes
        else:
            with open(file_path, 'rb') as file_handle:
                img_data = file_handle.read()

        img_array = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        # --- PREPROCESSING ---
        img_clahe = clahe.apply(img)

        # --- FEATURE EXTRACTION (HOG) ---
        # We turn pixels into 'gradients'
        fd = hog(img_clahe, orientations=9, pixels_per_cell=(8, 8),
                 cells_per_block=(2, 2), visualize=False)

        features_list.append(fd)
        labels.append(os.path.splitext(os.path.basename(filename))[0].split('_')[0])

        # INSPECTION: Print details for the first 2 photos
        if i < 2:
            print(f"--- Inspection for Photo {i + 1} ({filename}) ---")
            print(f"Raw Image Shape: {img.shape} (Total pixels: {img.size})")
            print(f"HOG Feature Vector Shape: {fd.shape}")
            print(f"First 5 HOG values (Gradients): {fd[:5]}")
            print("-" * 40)

    # --- FINAL STEP: Z-SCORE ---
    X = np.array(features_list)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"\nFinal Matrix Shape (X): {X_scaled.shape}")
    if len(X_scaled) > 0:
        print(f"First 5 Z-scored values of Photo 1: {X_scaled[0][:5]}")

    return X_scaled, np.array(labels)

# X, y = process_and_inspect('your_photos.zip')

#בתוספת PCA
from sklearn.decomposition import PCA


def process_optimized(zip_path, n_components=50):
    features_list = []
    labels = []
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    for filename, file_path, file_bytes in _iter_pgm_files(zip_path):
        if file_bytes is not None:
            img_data = file_bytes
        else:
            with open(file_path, 'rb') as file_handle:
                img_data = file_handle.read()

        img_array = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # --- שינוי גודל (מאיץ את הכל משמעותית!) ---
        img_small = cv2.resize(img, (64, 64))
        img_clahe = clahe.apply(img_small)

        # --- HOG ---
        fd = hog(img_clahe, orientations=9, pixels_per_cell=(8, 8),
                 cells_per_block=(2, 2), visualize=False)
        features_list.append(fd)
        labels.append(os.path.splitext(os.path.basename(filename))[0].split('_')[0])

    X = np.array(features_list)

    # --- Z-SCORE ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # נשמור רק את 50 הרכיבים הכי חשובים
    pca = PCA(n_components=min(n_components, X_scaled.shape[0], X_scaled.shape[1]))
    X_pca = pca.fit_transform(X_scaled)

    print(f"Original HOG shape: {X_scaled.shape}")
    print(f"PCA reduced shape: {X_pca.shape}")

    return X_pca, np.array(labels)