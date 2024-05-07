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
from DatasetPrep import get_NWRD_Dataset
from pynvml import *
import cv2
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()

def train_transforms(example_batch):
    images = [jitter(x) for x in example_batch['pixel_values']]
    labels = [x for x in example_batch['label']]
    inputs = processor(images, labels)
    return inputs


def val_transforms(example_batch):
    images = [x for x in example_batch['pixel_values']]
    labels = [x for x in example_batch['label']]
    inputs = processor(images, labels)
    return inputs

# def compute_metrics(eval_pred):
#   with torch.no_grad():
#     logits, labels = eval_pred
#     logits_tensor = torch.from_numpy(logits)
#     # scale the logits to the size of the label
#     logits_tensor = nn.functional.interpolate(
#         logits_tensor,
#         size=labels.shape[-2:],
#         mode="bilinear",
#         align_corners=False,
#     ).argmax(dim=1)

#     pred_labels = logits_tensor.detach().cpu().numpy()
#     # currently using _compute instead of compute
#     # see this issue for more info: https://github.com/huggingface/evaluate/pull/328#issuecomment-1286866576
#     metrics = metric._compute(
#             predictions=pred_labels,
#             references=labels,
#             num_labels=len(id2label),
#             ignore_index=0,
#             reduce_labels=processor.do_reduce_labels,
#         )
    
#     # add per category metrics as individual key-value pairs
#     per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
#     per_category_iou = metrics.pop("per_category_iou").tolist()

#     metrics.update({f"accuracy_{id2label[i]}": v for i, v in enumerate(per_category_accuracy)})
#     metrics.update({f"iou_{id2label[i]}": v for i, v in enumerate(per_category_iou)})
    
#     return metrics

hf_dataset_identifier = "hjawad367/ADE_20"

processor = SegformerImageProcessor()
jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1) 

ds = get_NWRD_Dataset()

ds = ds.shuffle(seed=1)
ds = ds["train"].train_test_split(test_size=0.2)
train_ds = ds["train"]
test_ds = ds["test"]

# Set transforms
train_ds.set_transform(train_transforms)
test_ds.set_transform(val_transforms)

print(train_ds)

# train_ds = train_ds.rename_column('image', 'pixel_values')
# test_ds = test_ds.rename_column('image', 'pixel_values')

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
).to("cuda")

epochs = 50
lr = 0.00006
batch_size = 2

hub_model_id = "segformer-b0-finetuned-Rust"

training_args = TrainingArguments(
    output_dir="segformer-b0-finetuned-segments-sidewalk-outputs",
    learning_rate=lr,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    save_total_limit=3,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=20,
    eval_steps=20,
    logging_steps=1,
    eval_accumulation_steps=5,
    load_best_model_at_end=True,
    push_to_hub=True,
    hub_model_id=hub_model_id,
    hub_strategy="end",
)

metric = evaluate.load("mean_iou")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
)

trainer.train()

kwargs = {
    "tags": ["vision", "image-segmentation"],
    "finetuned_from": pretrained_model_name,
    "dataset": hf_dataset_identifier,
}

processor.push_to_hub(hub_model_id)
trainer.push_to_hub(**kwargs)