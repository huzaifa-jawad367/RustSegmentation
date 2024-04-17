import os
import cv2
import numpy as np

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

for d in os.listdir('Training-2'):
    print(f"{d}:" + f" {len(os.listdir(f'Training-2/{d}'))}")

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
