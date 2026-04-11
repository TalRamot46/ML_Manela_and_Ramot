import os
import zipfile
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


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


def process_optimized(train_dir, n_components=50):
    features_list = []
    labels = []
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    for filename, file_path, file_bytes in _iter_pgm_files(train_dir):
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


def _partc_build_splits(
    train_dir,
    *,
    pipeline="optimized",
    n_components=50,
    test_size=0.25,
    random_state=42,
):
    if pipeline == "optimized":
        X_full, y_full_str = process_optimized(train_dir, n_components=n_components)
    elif pipeline == "inspect":
        X_full, y_full_str = process_and_inspect(train_dir)
    else:
        raise ValueError("pipeline must be 'optimized' or 'inspect'")

    le = LabelEncoder()
    y_full = le.fit_transform(y_full_str)
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=test_size, random_state=random_state
    )
    num_classes = len(le.classes_)
    return X_train, X_val, y_train, y_val, num_classes, le.classes_


def save_partc_train_val_npz(
    train_dir,
    npz_path,
    *,
    pipeline="optimized",
    n_components=50,
    test_size=0.25,
    random_state=42,
):
    """
    Run preprocessing, label encoding, and train/val split; write compressed .npz.
    Returns (X_train, X_val, y_train, y_val, num_classes) same as load_partc_train_val_npz.
    """
    X_train, X_val, y_train, y_val, num_classes, label_classes = _partc_build_splits(
        train_dir,
        pipeline=pipeline,
        n_components=n_components,
        test_size=test_size,
        random_state=random_state,
    )
    out_dir = os.path.dirname(os.path.abspath(npz_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        npz_path,
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        label_classes=label_classes,
    )
    return X_train, X_val, y_train, y_val, num_classes


def load_partc_train_val_npz(npz_path):
    """Load X_train, X_val, y_train, y_val and num_classes from save_partc_train_val_npz."""
    data = np.load(npz_path, allow_pickle=True)
    label_classes = data["label_classes"]
    num_classes = int(len(label_classes))
    return (
        data["X_train"],
        data["X_val"],
        data["y_train"],
        data["y_val"],
        num_classes,
    )
