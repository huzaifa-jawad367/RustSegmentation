from datasets import Dataset, DatasetDict, Image
import os
import pyarrow.parquet as pq


def get_dataset():
    list_of_img_dirs = [f'Training-2/{x}' for x in os.listdir('Training-2') if 'mask' not in x]
    list_of_mask_dirs = [f'Training-2/{x}' for x in os.listdir('Training-2') if 'mask' in x]

    list_of_img = []
    for l in list_of_img_dirs:
        imgs = os.listdir(l)
        paths_to_imgs = [os.path.join(l, x) for x in imgs]
        list_of_img = list_of_img + paths_to_imgs

    list_of_masks = []
    for l in list_of_mask_dirs:
        imgs = os.listdir(l)
        paths_to_imgs = [os.path.join(l, x) for x in imgs]
        list_of_masks = list_of_masks + paths_to_imgs

    image_paths_train = list_of_img
    label_paths_train = list_of_masks

    # image_paths_train = ["path/to/image_1.jpg/jpg", "path/to/image_2.jpg/jpg", ..., "path/to/image_n.jpg/jpg"]
    # label_paths_train = ["path/to/annotation_1.png", "path/to/annotation_2.png", ..., "path/to/annotation_n.png"]

    # image_paths_validation = [...]
    # label_paths_validation = [...]

    def create_dataset(image_paths, label_paths):
        dataset = Dataset.from_dict({"pixel_values": sorted(image_paths),
                                    "label": sorted(label_paths)})
        dataset = dataset.cast_column("pixel_values", Image())
        dataset = dataset.cast_column("label", Image())
        return dataset

    # step 1: create Dataset objects
    train_dataset = create_dataset(image_paths_train, label_paths_train)
    # validation_dataset = create_dataset(image_paths_validation, label_paths_validation)

    # step 2: create DatasetDict
    dataset = DatasetDict({
        "train": train_dataset,
        }
    )

    return dataset

# step 3: push to Hub (assumes you have ran the huggingface-cli login command in a terminal/notebook)
# dataset.push_to_hub("hjawad367/Rust_aug")

get_dataset()