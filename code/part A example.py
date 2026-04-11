import zipfile
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
import os


def process_and_inspect_full(zip_path):
    features_list = []
    labels = []
    processed_filenames = []

    # Initialize CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # This normalizes lighting locally to make facial features consistent
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    print(f"{'=' * 60}")
    print(f"STEP 1 & 2: LOADING, CLAHE, AND HOG EXTRACTION")
    print(f"{'=' * 60}")

    with zipfile.ZipFile(zip_path, 'r') as z:
        # Filter for .pgm files and ignore system hidden files (like __MACOSX)
        all_files = [f for f in z.namelist() if f.endswith('.pgm') and '__MACOSX' not in f]
        all_files = sorted(all_files)

        # Select only the first 10 photos
        target_files = all_files[:10]

        for i, filename in enumerate(target_files):
            # Read image data from ZIP
            img_data = z.read(filename)
            img_array = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print(f"Error: Could not decode {filename}")
                continue

            # --- PREPROCESSING: CLAHE ---
            img_clahe = clahe.apply(img)

            # --- FEATURE EXTRACTION: HOG ---
            # orientations: directions of gradients
            # pixels_per_cell: local spatial area
            # cells_per_block: normalization area
            fd = hog(img_clahe,
                     orientations=9,
                     pixels_per_cell=(8, 8),
                     cells_per_block=(2, 2),
                     visualize=False)

            features_list.append(fd)
            processed_filenames.append(filename)

            # Extract label (e.g., 'person01' from 'folder/person01_01.pgm')
            label = os.path.basename(filename).split('_')[0]
            labels.append(label)

            print(f"Photo {i + 1}: {filename}")
            print(f"   -> Raw Pixels: {img.size} ({img.shape[0]}x{img.shape[1]})")
            print(f"   -> HOG Features: {len(fd)}")

    # Convert list to NumPy array for mathematical operations
    X_raw_hog = np.array(features_list)

    # --- STEP 3: GLOBAL Z-SCORE (Standardization) ---
    # This centers data around 0 and scales to unit variance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw_hog)

    print(f"\n{'=' * 60}")
    print(f"STEP 3: FINAL PROCESSED DATA (Z-SCORED)")
    print(f"{'=' * 60}")
    print(f"Final Matrix Shape: {X_scaled.shape} (Samples x Features)")
    print(f"Each row is now a standardized vector ready for ML models.\n")

    # Print the first 5 final values for each of the 10 photos
    for idx in range(len(X_scaled)):
        # Rounding to 3 decimal places for readability in the console
        sample_snippet = X_scaled[idx][:5].round(3)
        print(f"Photo {idx + 1} ({labels[idx]}) Features: {sample_snippet} ...")

    print(f"\n{'=' * 60}")
    print("ANALYSIS COMPLETE")
    print("The data is now ready for Linear, Margin, or Tree-based models.")
    print(f"{'=' * 60}")

    return X_scaled, np.array(labels)

# --- EXECUTION ---
# Change 'my_dataset.zip' to the actual name of your zip file
X, y = process_and_inspect_full('my_dataset.zip')