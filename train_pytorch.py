from transformers import AutoImageProcessor, SegformerModel, SegformerForSemanticSegmentation, SegformerImageProcessor
from torchvision.transforms import ColorJitter
from torchvision import transforms as T
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate
import json
from huggingface_hub import hf_hub_download
from torch import nn
from DatasetPrep import get_dataset
from pynvml import *
import cv2
from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
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
  
def run_1_epoch(model, loss_fn, loader, optimizer = None, train = False):
    if train:
        model.train()
    else:
        model.eval()


    total_correct_preds = 0

    total_loss = 0

    # Number of images we can get by the loader
    total_samples_in_loader = len(loader.dataset)

    # number of batches we can get by the loader
    total_batches_in_loader = len(loader)

    for index_of_batch, sample_batch in enumerate(tqdm(loader)):

        # Transfer image_batch to GPU if available
        image_batch = sample_batch['image'].to('cuda')
        labels = sample_batch['mask'].to('cuda')

        # Zeroing out the gradients for parameters
        if train:
            assert optimizer is not None, "Optimizer must be provided if train=True"
            optimizer.zero_grad()

        # Forward pass on the input batch
        output = model(image_batch)

        # Acquire predicted class indices
        _, predicted = torch.max(output.data, 1) # the dimension 1 corresponds to max along the rows

        # Compute the loss for the minibatch
        loss = loss_fn(output, labels)

        # Backpropagation
        if train:
            loss.backward()

        # Update the parameters using the gradients
        if train:
            optimizer.step()

        # Extra variables for calculating loss and accuracy
        # count total predictions for accuracy calcutuon for this epoch
        #total_correct_preds += (predicted == labels).sum().item()

        total_loss += loss.item()

    loss = total_loss / total_batches_in_loader
    accuracy = pixel_accuracy(output, labels)
    mean_IoU = mIoU(output, labels)

    return loss, accuracy, mean_IoU
    
if __name__ == "__main__":

    torch.cuda.empty_cache()

    accelerator = Accelerator()

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    IMAGE_PATH = 'Training\image'
    MASK_PATH = 'Training\mask'

    train_transform = T.Compose([
        # T.Resize(512, 512),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
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
    train_set = RustDataset(IMAGE_PATH, MASK_PATH, mean, std, train_transform)
    val_set = RustDataset(IMAGE_PATH, MASK_PATH, mean, std, val_transform)

    #dataloader
    batch_size= 4

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=4)

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
    num_training_steps = 1000
    start_epoch = 0
    progress_bar = tqdm(range(num_training_steps))

    model.train()

    for epoch in range(start_epoch, epochs):

        # # with torch.no_grad:
        # model.zero_grad()
        for batch in train_loader:

            with accelerator.accumulate(model):

                batch = {k: v for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                # loss.backward()
                accelerator.backward(loss=loss)

                print()
                print(f"Loss: {loss}")
                train_losses.append(loss)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

                metric = evaluate.load("accuracy")
                model.eval()
                for batch in eval_dataloader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    with torch.no_grad():
                        outputs = model(**batch)

                    logits = outputs.logits
                    predictions = torch.argmax(logits, dim=-1)
                    metric.add_batch(predictions=predictions, references=batch["labels"])

        model.save()

        #     # Train model for one epoch

        #     # Get the current learning rate from the optimizer
        #     current_lr = optimizer.param_groups[0]['lr']

        #     print("Epoch %d: Train \nLearning Rate: %.6f"%(epoch, current_lr))
        #     train_loss, train_accuracy, train_mIoU  = run_1_epoch(model, loss_function, train_loader, optimizer, train= True)

        #     # Update the learning rate scheduler
        #     scheduler.step()

        #     # Lists for train loss and accuracy for plotting
        #     train_losses.append(train_loss)
        #     train_accuracies.append(train_accuracy)
        #     train_mIoUs.append(train_mIoU)

        #     # Validate the model on validation set
        #     print("Epoch %d: Validation"%(epoch))
        #     with torch.no_grad():
        #         val_loss, val_accuracy, val_mIoU  = run_1_epoch(model, loss_function, val_loader, optimizer, train= False)

        #     # Lists for val loss and accuracy for plotting
        #     val_losses.append(val_loss)
        #     val_accuracies.append(val_accuracy)
        #     val_mIoUs.append(val_mIoU)

        #     print('train loss: %.4f'%(train_loss))
        #     print('val loss: %.4f'%(val_loss))
        #     print('train_accuracy %.2f' % (train_accuracy))
        #     print('val_accuracy %.2f' % (val_accuracy))
        #     print('train_IoU %.2f'%(train_mIoU))
        #     print('val_IoU %.2f'%(val_mIoU))

        #     if val_mIoU > val_mIoU_max:
        #         val_mIoU_max = val_mIoU
        #         print("New max val mean IoU Acheived %.2f. Saving model.\n\n"%(val_mIoU_max))

        #         checkpoint = {
        #         'model': model.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #         'scheduler': scheduler.state_dict(),
        #         'trianed_epochs': epoch,
        #         'train_losses': train_losses,
        #         'train_accuracies': train_accuracies,
        #         'train_mIoUs': train_mIoUs,
        #         'val_losses': val_losses,
        #         'val_accuracies': val_accuracies,
        #         'val_accuracy_max': val_mIoU_max,
        #         'val_mIoUs': val_mIoUs,
        #         'lr': optimizer.param_groups[0]['lr']
        #         }
        #         torch.save(checkpoint, best_val_checkpoint_path)

        #     else:
        #         print("val mean IoU did not increase from %.2f\n\n"%(val_mIoU_max))

        #     # Save checkpoint for the last epoch
        #     checkpoint = {
        #         'model': model.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #         'scheduler': scheduler.state_dict(),
        #         'trianed_epochs': epoch,
        #         'train_losses': train_losses,
        #         'train_accuracies': train_accuracies,
        #         'train_mIoUs': train_mIoUs,
        #         'val_losses': val_losses,
        #         'val_accuracies': val_accuracies,
        #         'val_accuracy_max': val_mIoU_max,
        #         'val_mIoUs': val_mIoUs,
        #         'lr': optimizer.param_groups[0]['lr']
        #         }

        #     torch.save(checkpoint, checkpoint_path)

        # plt.figure()
        # plt.plot(train_accuracies, label="train_accuracy")
        # plt.plot(val_accuracies, label="val_accuracy")
        # plt.legend()

        # plt.xlabel('Epoch')
        # plt.ylabel('Accuracy')

        # plt.title('Training and val Accuracy')

        # plt.figure()
        # plt.plot(train_losses, label="train_loss")
        # plt.plot(val_losses, label="val_loss")

        # plt.legend()
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # plt.title('Training and val Loss')

        # plt.figure()
        # plt.plot(train_mIoUs, label="train_mIoU")
        # plt.plot(val_mIoUs, label="val_mIoU")

        # plt.legend()
        # plt.xlabel('Epoch')
        # plt.ylabel('mIoU')
        # plt.title('Training and val mIoU')

