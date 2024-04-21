from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torch
from torchvision import transforms as T
import cv2
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

class RustDataset(Dataset):

    def __init__(self, img_path, mask_path, mean, std, transform=None):
        self.img_path = img_path
        self.mask_path = mask_path
        # self.X = X
        self.transform = transform
        self.mean = mean
        self.std = std
        self.files = sorted(os.listdir(self.img_path))

    def __len__(self):
        return len(os.listdir('Training/image'))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image_name = self.files[idx]
        if image_name.startswith('rust') and image_name.split('.')[0][-1].isdigit():
            mask_name = image_name[:4] + image_name[5:]
        else:
            mask_name = image_name

        img = np.array(Image.open(os.path.join(self.img_path, image_name)))
        mask = np.array(Image.open(os.path.join(self.mask_path, mask_name)).convert('L'))

        sample = {'image': img, 'mask': mask}

        if self.transform:
            img = self.transform(sample['image'])
            mask = torch.from_numpy(mask).long()
            sample = {'image': img, 'mask': mask}

        return sample

if __name__ == '__main__':    
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    transformations = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    rust_dataset = RustDataset(img_path="Training/image", mask_path='Training/mask', mean=mean, std=std, transform=transformations)

    dataloader = DataLoader(rust_dataset, batch_size=4,
                            shuffle=True, num_workers=4)

    # Split the dataset into train and validation sets
    num_samples = len(rust_dataset)
    num_train = int(num_samples * 0.8)
    num_val = num_samples - num_train

    train_dataset, val_dataset = random_split(rust_dataset, [num_train, num_val])

    # Create DataLoaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)

    print(len(train_loader))
    print(len(val_loader))