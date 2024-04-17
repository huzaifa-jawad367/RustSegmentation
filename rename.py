# from datasets import load_dataset
# dataset = load_dataset("hjawad367/ADE_20")

# dataset['train'] = dataset['train'].rename_column('label', 'labels').rename_column('image', 'pixel_values')

# print(dataset)

# # Write changes to new hub
# dataset.push_to_hub("hjawad367/ADE_20")

"""
\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
"""
# from DatasetPrep import get_dataset

# print(get_dataset())

"""
\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
"""
# import os
# import cv2
# import numpy as np

# # For each directory in the list
# for dir in list_of_mask_dirs:
#     # Get list of all image files in the directory
#     image_files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]

#     # For each image file
#     for image_file in image_files:
#         # Read the image
#         img = cv2.imread(os.path.join(dir, image_file))
        
#         # Convert the image to grayscale
#         img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
#         # Set all non-zero pixel values to 255
#         img_gray[np.where(img_gray != 0)] = 255

#         # Save the modified image back to the same location
#         cv2.imwrite(os.path.join(dir, image_file), img_gray)


###

# import os
# import cv2
# import numpy as np

# list_of_mask_dirs = [f'Training-2/{x}' for x in os.listdir('Training-2') if 'mask' in x]

# # For each directory in the list
# for dir in list_of_mask_dirs:
#     # Get list of all image files in the directory
#     image_files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]

#     # For each image file
#     for image_file in image_files:
#         # Read the image
#         img = cv2.imread(os.path.join(dir, image_file))
        
#         # Print the shape of the image
#         print("Shape:", img.shape)

#         # Get unique values and their counts
#         unique_values, counts = np.unique(img, return_counts=True)

#         # Print the number of unique values
#         print("Number of unique values:", len(unique_values))

#         # Print the unique values
#         print("Unique values:", unique_values)

import os

def verify_files(image_dir, mask_dir):
    # Get list of files in both directories
    image_files = set([x.split('.')[0] for x in os.listdir(image_dir)])
    mask_files = set([x.split('.')[0] for x in os.listdir(mask_dir)])

    # Check for missing files
    missing_in_mask = image_files - mask_files
    missing_in_image = mask_files - image_files
    print(len(missing_in_image))
    print(len(missing_in_mask))

    if not missing_in_mask and not missing_in_image:
        print("All files in the 'image' directory have corresponding files in the 'mask' directory.")
    else:
        if missing_in_mask:
            print("The following files are missing in the 'mask' directory:")
            print(missing_in_mask)
        if missing_in_image:
            print("The following files are in the 'mask' directory but not in the 'image' directory:")
            print(missing_in_image)

# Usage
image_dir = 'Training/image'
mask_dir = 'Training/mask'
verify_files(image_dir, mask_dir)

