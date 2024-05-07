from transformers import AutoImageProcessor, SegformerModel, SegformerForSemanticSegmentation, SegformerImageProcessor
from torchvision.transforms import ColorJitter
from torchvision import transforms as T
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate
import json
from huggingface_hub import hf_hub_download
from torch import nn
# from DatasetPrep import get_dataset
from pynvml import *
import cv2
from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from accelerate import Accelerator

from sklearn.metrics import accuracy_score

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
        return len(os.listdir('data/Training/image'))

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
    
def mean_iou(preds, labels, smooth=1e-6):
    # Assuming preds and labels are (N, H, W) where N is the batch size,
    # and each value is 0 for background and 1 for the object (binary segmentation)
    intersection = torch.logical_and(preds, labels).sum((1, 2))
    union = torch.logical_or(preds, labels).sum((1, 2))
    iou = (intersection + smooth) / (union + smooth)  # Adding smooth to avoid division by zero
    return iou.mean().item()  # Mean over the batch
    
def mIoU(pred_mask, mask, smooth=1e-10, n_classes=5):
    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for clas in range(0, n_classes): #loop per pixel class
            true_class = pred_mask == clas
            true_label = mask == clas

            if true_label.long().sum().item() == 0: #no exist label in this loop
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + smooth) / (union +smooth)
                iou_per_class.append(iou)
        return np.nanmean(iou_per_class)
    
def pixel_accuracy(output, mask):
    with torch.no_grad():
        output = torch.argmax(F.softmax(output, dim=1), dim=1)
        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy

def plot_learning_curves(train_data, val_data, title, ylabel):
    plt.figure(figsize=(10, 5))
    plt.plot(train_data, label='Train')
    plt.plot(val_data, label='Validation')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()
    
if __name__ == "__main__":

    torch.cuda.empty_cache()

    accelerator = Accelerator()

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    IMAGE_PATH = 'data/Training/image'
    MASK_PATH = 'data/Training/mask'

    train_transform = T.Compose([
        # T.Resize(512, 512),
        # T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        T.ToTensor(),
        T.Normalize(mean, std)
    ])

    val_transform = T.Compose([
        # T.Resize(512, 512),
        T.ToTensor(),
        T.Normalize(mean, std)
    ])

    # device = torch.device('cuda')
    device = accelerator.device

    #datasets
    dataset = RustDataset(IMAGE_PATH, MASK_PATH, mean, std, train_transform)
    # val_set = RustDataset(IMAGE_PATH, MASK_PATH, mean, std, val_transform)

    # Split the dataset into train and validation sets
    num_samples = len(dataset)
    num_train = int(num_samples * 0.8)
    num_val = num_samples - num_train

    train_dataset, val_dataset = random_split(dataset, [num_train, num_val])

    #dataloader
    batch_size= 4

    # # Applying different transforms to the validation subset if necessary
    # val_dataset = Subset(dataset, [x.indices for x in val_dataset])
    # for idx in val_dataset.indices:
    #     dataset.files[idx] = val_transform(dataset.files[idx])

    # Create DataLoaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    hf_dataset_identifier = "hjawad367/ADE_20"

    repo_id = f"datasets/{hf_dataset_identifier}"
    filename = "id2label.json"
    id2label = json.load(open("id2label.json", "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label.items()}

    num_labels = len(id2label)

    pretrained_model_name = "nvidia/mit-b0" 
    model = SegformerForSemanticSegmentation.from_pretrained(
        pretrained_model_name,
        id2label=id2label,
        label2id=label2id
    ).to(device)

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00006)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    epochs = 10

    accelerator = Accelerator(gradient_accumulation_steps=2)

    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    train_accuracies = []
    val_accuracies = []

    train_losses = []
    val_losses = []

    train_mIoUs = []
    val_mIoUs = []

    val_mIoU_max = -1

    checkpoint_path = 'segformer_results/checkpoint.pth'
    best_val_checkpoint_path = 'segformer_results/best_val_checkpoint.pth'

    # Start epoch is zer
    # o for new training
    num_training_steps = 100
    start_epoch = 0
    # progress_bar = tqdm(range(num_training_steps))
    current_step = 0

    num_labels = 255
    ignore_index = 0

    for epoch in range(start_epoch, epochs):

        print(f"\nEpoch: {epoch}\n")

        model.train()
        train_epoch_losses = []
        train_epoch_predictions = []
        train_epoch_labels = []

        # Reinitialize progress bar for the actual number of batches in this epoch
        progress_bar = tqdm(total=len(train_loader), desc=f"Epoch {epoch}")
        # train_metric = evaluate.load("mean_iou", num_labels=num_labels, ignore_index=ignore_index)

        for batch in train_loader:

            optimizer.zero_grad()

            with accelerator.accumulate():
                batch = {k: v for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss=loss)

                train_epoch_losses.append(loss.item())
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                train_epoch_predictions.extend(predictions.cpu().numpy())
                train_epoch_labels.extend(batch["labels"].cpu().numpy())

                # train_metric.add_batch(predictions=predictions, references=batch["labels"])

                optimizer.step()
                scheduler.step()
                progress_bar.update(1)
                current_step += 1

        train_loss = np.mean(train_epoch_losses)
        # train_accuracy = accuracy_score(train_epoch_labels, train_epoch_predictions)
        # train_mIoU = mean_iou(train_epoch_labels, train_epoch_predictions)
        train_losses.append(train_loss)
        # train_accuracies.append(train_accuracy)
        # train_mIoUs.append(train_mIoU)
        # train_iou = train_metric.compute(num_labels=num_labels, ignore_index=ignore_index)

        model.eval()
        val_epoch_losses = []
        val_epoch_predictions = []
        val_epoch_labels = []
        # val_metric = evaluate.load("mean_iou")
        # metric_acc = evaluate.load("accuracy")
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)

                val_epoch_losses.append(loss.item())
                val_epoch_predictions.extend(predictions.cpu().numpy())
                val_epoch_labels.extend(batch["labels"].cpu().numpy())
                # val_metric.add_batch(predictions=predictions, references=batch["labels"])

        val_loss = np.mean(val_epoch_losses)
        # val_accuracy = accuracy_score(val_epoch_labels, val_epoch_predictions)
        # val_mIoU = mean_iou(val_epoch_labels, val_epoch_predictions)
        val_losses.append(val_loss)
        # val_accuracies.append(val_accuracy)
        # val_mIoUs.append(val_mIoU)
        # val_iou = val_metric.compute(num_labels=num_labels, ignore_index=ignore_index)

        # print(f"Epoch {epoch}: Train Loss: {train_loss}, Val Loss: {val_loss}, Train Accuracy: {train_accuracy}, Val Accuracy: {val_accuracy}")
        print(f"Epoch {epoch}: Train Loss: {train_loss}, Val Loss: {val_loss}")



        # validation_iou = metric.compute()
        # validation_acc = metric_acc.compute()
        # print(f"Validation IoU: {validation_iou}")
        # # Here you could add a check to save the model only if it improves

        # torch.save(model.state_dict(), f'segformer_results/model_epoch_{epoch}_val_loss_{val_loss}.pth')
        model.save_pretrained(f'segformer_results/model_epoch_{epoch}')

    # Plot learning curves for loss
    plot_learning_curves(train_losses, val_losses, "Learning Curve (Loss)", "Loss")

    # # Plot learning curves for accuracy
    # plot_learning_curves(train_accuracies, val_accuracies, "Learning Curve (Accuracy)", "Accuracy")

    # # Plot learning curves for Mean IoU
    # plot_learning_curves(train_mIoUs, val_mIoUs, "Learning Curve (mIoU)", "Mean IoU")