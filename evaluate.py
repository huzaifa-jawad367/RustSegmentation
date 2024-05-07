import evaluate
from torch.utils.data import Dataset, DataLoader, random_split
import os
import torch
from PIL import Image
from torchvision import transforms as T
import json
from transformers import SegformerForSemanticSegmentation
import numpy as np
from accelerate import Accelerator

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

        img = Image.open(os.path.join(self.img_path, image_name))
        mask = Image.open(os.path.join(self.mask_path, mask_name)).convert('L')

        sample = {'pixel_values': img, 'labels': mask}

        if self.transform:
            img = self.transform(sample['pixel_values'])
            mask = torch.from_numpy(np.array(mask)).long()
            sample = {'pixel_values': img, 'labels': mask}

        return sample

if __name__=='__main__':

    IMAGE_PATH = 'Training\image'
    MASK_PATH = 'Training\mask'

    torch.cuda.empty_cache()

    accelerator = Accelerator()

    device = accelerator.device

    mean = [0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    train_transform = T.Compose([
        # T.Resize(512, 512),
        T.ToTensor(),
        T.Normalize(mean, std)
    ])

    dataset = RustDataset(IMAGE_PATH, MASK_PATH, mean, std, train_transform)

    num_samples = len(dataset)
    num_train = int(num_samples * 0.8)
    num_val = num_samples - num_train

    train_dataset, val_dataset = random_split(dataset, [num_train, num_val])

    eval_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=2)

    filename = "id2label.json"
    id2label = json.load(open("id2label.json", "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label.items()}

    num_labels = len(id2label)

    pretrained_model_name = "nvidia/mit-b0" 
    model = SegformerForSemanticSegmentation.from_pretrained(
        'segformer_results\model_epoch_7_val_loss_4.712333997835134.pth',
        id2label=id2label,
        label2id=label2id
    ).to(device)

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00006)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    epochs = 10

    metric = evaluate.load("accuracy")
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    metric.compute()