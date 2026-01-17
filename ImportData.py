import kaggle
kaggle.api.authenticate()
kaggle.api.dataset_download_files('andrewmvd/cancer-instance-segmentation-and-classification-3', path = '.',unzip = True)
kaggle.api.dataset_metadata('andrewmvd/cancer-instance-segmentation-and-classification-3', path = '.')

import shutil
import os

##Make a new folder called Dataset
# Define the name/path of the folder
folder_name = "dataset"

# Check if it already exists to avoid errors, then create it
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    print(f"Directory '{folder_name}' created successfully!")
else:
    print(f"Directory '{folder_name}' already exists.")

##Copy IMAGES files into dataset folder
# 1. Define paths
images_file = r'.\data\Segmentation Project - Github\Images\images.npy'
destination_folder = r'.\data\Segmentation Project - Github\dataset'

# 2. Ensure the destination folder exists
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# 3. Copy the file
# shutil.copy2 preserves metadata like creation/modification dates
shutil.copy2(images_file, destination_folder)

print(f"Copied {images_file} to {destination_folder}")


##Copy MASKS files into dataset folder
# 1. Define paths
masks_file = r'.\data\Segmentation Project - Github\Masks\masks.npy'
destination_folder = r'.\data\Segmentation Project - Github\dataset'

# 2. Ensure the destination folder exists
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# 3. Copy the file
# shutil.copy2 preserves metadata like creation/modification dates
shutil.copy2(masks_file, destination_folder)

print(f"Copied {masks_file} to {destination_folder}")


##Make a new folder called Split
# Define the name/path of the folder
split = "split"

# Check if it already exists to avoid errors, then create it
if not os.path.exists(split):
    os.makedirs(split)
    print(f"Directory '{split}' created successfully!")
else:
    print(f"Directory '{split}' already exists.")