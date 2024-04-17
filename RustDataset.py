from torch.utils.data import Dataset, DataLoader
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

    face_dataset = RustDataset(img_path="Training/image", mask_path='Training/mask', mean=mean, std=std, transform=transformations)

    dataloader = DataLoader(face_dataset, batch_size=4,
                            shuffle=True, num_workers=4)
    
    print(f"Total samples in loader: {len(dataloader.dataset)}")
    print(f"total_batches_in_loader: {len(dataloader)}")

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(),
            sample_batched['mask'].size())

        if i_batch == 3:
            # Image display
            plt.figure()
            # Adjust image data range or type if necessary
            image_to_show = sample_batched['image'][0].permute(1, 2, 0)
            image_to_show = (image_to_show - image_to_show.min()) / (image_to_show.max() - image_to_show.min())  # Normalize to 0-1
            plt.imshow(image_to_show.numpy())
            plt.title("Image")
            
            # Mask display
            plt.figure()
            plt.imshow(sample_batched['mask'][0], cmap='gray')  # Ensure correct color mapping
            plt.title("Mask")
            
            plt.show()  # This makes sure that the plots are displayed
            break