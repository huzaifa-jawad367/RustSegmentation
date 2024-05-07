import os
import cv2
import numpy as np
import random
import shutil

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
            unique_mask_vals = set()  # Initialize an empty set for unique mask values
            with open(file_path, 'r') as file:
                # Read the contents of the file and split it into lines
                lines = file.readlines()
                # Iterate over each line and extract unique values
                for line in lines:
                    unique_mask_vals.update(line.strip().split(','))

            # Print the filename and its unique mask values
            print(f"Unique values for {filename}: {unique_mask_vals}")

def threshold(img, threshold_val=0):
    pass

def sample_images(image_dir, mask_dir, output_image_dir, output_mask_dir, num_samples=1700):
    # Ensure output directories exist
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    # Get the list of images in the directory
    images = os.listdir(image_dir)
    random.shuffle(images)  # Shuffle the list for random sampling

    # Sample images
    sampled_images = images[:num_samples]

    # Copy the sampled images and masks to the output directories
    for image_name in sampled_images:
        image_path = os.path.join(image_dir, image_name)
        mask_path = os.path.join(mask_dir, image_name)

        # Ensure the mask exists
        if os.path.exists(mask_path):
            shutil.copy(image_path, os.path.join(output_image_dir, image_name))
            shutil.copy(mask_path, os.path.join(output_mask_dir, image_name))
        else:
            print(f"Mask not found for image: {image_name}")

    print("Sampling and copying complete.")

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


