import zipfile
import random
from PIL import Image
import io

# Path to your zip file
zip_path = 'C:\\Users\\ariel\PycharmProjects\TECHNOTZ DS TARGIL 2\Train Set (Labeled)-20260407T132757Z-3-001.zip'

with zipfile.ZipFile(zip_path, 'r') as archive:
    # Filter for files that end with .pgm
    all_files = [f for f in archive.namelist() if f.endswith('.pgm')]

    # Ensure we don't try to grab more files than exist
    num_to_sample = min(len(all_files), 20)
    selected_files = random.sample(all_files, num_to_sample)

    for file_name in selected_files:
        # Read the file data into memory
        with archive.open(file_name) as file:
            img_data = file.read()
            # Wrap data in BytesIO so PIL can read it like a file
            img = Image.open(io.BytesIO(img_data))

            print(f"Opening: {file_name}")
            img.show()