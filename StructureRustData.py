import os
import cv2
import numpy as np
import random
import shutil
from PIL import Image

def save_non_rust_masks():
    # Define source and destination directories
    src_dir = 'Training-2/non_rust'
    dest_dir = 'Training-2/masks_non_rust'

    # Get list of all image files in the source directory
    image_files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]

    # For each image file
    for image_file in image_files:

        if image_file.startswith('._'):
            continue

        # Read the image
        img = cv2.imread(f"{src_dir}/{image_file}")

        print(img.shape)
        
        # Get the shape of the image
        shape = img.shape
        
        # Create a mask of the same shape filled with zeros
        mask = np.zeros(shape, dtype=np.uint8)
        
        # Save the mask to the destination directory with the same name as the image file
        cv2.imwrite(os.path.join(dest_dir, image_file), mask)


def clean_data():
    dir1 = [f'Training/{x}' for x in os.listdir('Training')]
    dir2 = [f'Training-2/{x}' for x in os.listdir('Training-2')]

    all_dirs = dir1 + dir2

    for d in all_dirs:
        for image_files in os.listdir(d):
            if image_files.startswith('._'):
                os.remove(os.path.join(d,image_files))

def print_unique_mask_vals(dir_path):
    # Loop through the files in the directory
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        if os.path.isfile(file_path):  # Check if it's a file (not a subdirectory)
            mask = np.array(Image.open(file_path))
            unique_mask_vals = np.unique(mask)

            # Print the filename and its unique mask values
            print(f"Unique values for {filename}: {unique_mask_vals}")

def threshold(img, threshold_val=0):
    pass

def sample_images_and_masks(src_images_dir, src_masks_dir, dest_images_dir, dest_masks_dir, n_samples):
    """
    Randomly samples images and their corresponding masks from specified directories and copies them to new directories.

    Args:
    src_images_dir (str): Source directory containing the original images.
    src_masks_dir (str): Source directory containing the mask images.
    dest_images_dir (str): Destination directory to store sampled images.
    dest_masks_dir (str): Destination directory to store corresponding masks.
    n_samples (int): Number of images and masks to sample.
    """
    # Create the destination directories if they don't exist
    os.makedirs(dest_images_dir, exist_ok=True)
    os.makedirs(dest_masks_dir, exist_ok=True)

    # Get a list of all files in the source image directory
    all_files = os.listdir(src_images_dir)
    # Ensure that we do not sample more files than exist
    n_samples = min(n_samples, len(all_files))

    # Randomly sample file names
    sampled_files = random.sample(all_files, n_samples)

    # Copy the sampled images and their corresponding masks to the destination directories
    for filename in sampled_files:
        src_image_path = os.path.join(src_images_dir, filename)
        src_mask_path = os.path.join(src_masks_dir, filename)  # assuming mask has the same file name
        dest_image_path = os.path.join(dest_images_dir, filename)
        dest_mask_path = os.path.join(dest_masks_dir, filename)

        # Copy files
        shutil.copy(src_image_path, dest_image_path)
        shutil.copy(src_mask_path, dest_mask_path)


# if __name__=='__main__':
#     # print_unique_mask_vals('data/Training_OG/mask')
#     sample_images_and_masks('data/Training-2/non_rust', 'data/Training-2/masks_non_rust', 'data/Training_OG/image', 'data/Training_OG/mask', 1700)


# for d in os.listdir('Training-2'):
#     print(f"{d}:" + f" {len(os.listdir(f'Training-2/{d}'))}")

# import os

# # Directory paths
# rust_dir = "Training-2/rotated_rust"
# masks_dir = "Training-2/rotated_masks"

# # Get the list of files in each directory
# rust_files = os.listdir(rust_dir)
# masks_files = os.listdir(masks_dir)

# # Find files in 'rotated_rust' but not in 'rotated_masks'
# rust_not_in_masks = [file for file in rust_files if file not in masks_files]

# # Print filenames
# print("Files in 'rotated_rust' but not in 'rotated_masks':")
# for file in rust_not_in_masks:
#     print(file)


