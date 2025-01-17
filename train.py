import os
import torch
import numpy as np
import monai
import vmtk
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Spacingd, ScaleIntensityRanged, CropForegroundd, \
    RandCropByPosNegLabeld, RandFlipd, RandRotate90d, AsDiscrete
from monai.utils import set_determinism
from torch import nn
from torch.optim import lr_scheduler
import yaml
import logging
from sLoGNN import sLoGNN

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration
with open('config_sLoG.yaml', 'r') as config_file:
    config_param = yaml.load(config_file, Loader=yaml.Loader)
batch_num = config_param['batch_num']
target_path = config_param["target_path"]
dataset_path = config_param["dataset_path"]
train_data_path = os.path.join(dataset_path, 'train', 'image')
test_data_path = os.path.join(dataset_path, 'test', 'image')
val_dice_save = config_param["dice_save_path"]
optimum_dice_pth = config_param["dice_value_path"]

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_determinism(seed=0)

# Define transforms
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Spacingd(keys=["image", "label"], pixdim=(0.8, 0.8, 0.3), mode=("bilinear", "nearest")),
    ScaleIntensityRanged(keys=["image"], a_min=200, a_max=500, b_min=0, b_max=1, clip=True),
    CropForegroundd(keys=["image", "label"], source_key="label", margin=2),
    RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", spatial_size=(64, 64, 64), pos=1, neg=1),
    RandFlipd(keys=["image", "label"], spatial_axis=[0, 1, 2], prob=0.1),
    RandRotate90d(keys=["image", "label"], prob=0.1, max_k=3),
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Spacingd(keys=["image", "label"], pixdim=(0.8, 0.8, 0.8), mode=("bilinear", "nearest")),
    ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
])

post_label = AsDiscrete(to_onehot=2)
post_pred = AsDiscrete(argmax=True, to_onehot=2)

# Define datasets and data loaders
train_ds = CacheDataset(data=[{"image": os.path.join(train_data_path, file),
                               "label": os.path.join(dataset_path, 'train', 'label', file)} for file in
                              os.listdir(train_data_path)],
                        transform=train_transforms, cache_rate=1)
val_ds = CacheDataset(data=[{"image": os.path.join(test_data_path, file),
                             "label": os.path.join(dataset_path, 'test', 'label', file)} for file in
                            os.listdir(test_data_path)],
                      transform=val_transforms, cache_rate=1)

train_loader = DataLoader(train_ds, batch_size=batch_num, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1)

# Define model
model = sLoGNN(1, 2).to(device)

# Define optimizer, scheduler, and loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=5e-5)
loss_fn = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)
metric_fn = DiceMetric(include_background=True, reduction="mean")


# Add Bayesian loss (Dice + KL divergence)
class BayesianDiceLoss(nn.Module):
    def __init__(self, dice_loss, kl_weight=1e-4):
        super().__init__()
        self.dice_loss = dice_loss
        self.kl_weight = kl_weight

    def forward(self, output, target, kl_div):
        dice = self.dice_loss(output, target)
        loss = dice + self.kl_weight * kl_div
        return loss


bayesian_loss_fn = BayesianDiceLoss(loss_fn)


# Training loop
def train(epoch):
    model.train()
    total_loss = 0
    for batch_data in train_loader:
        inputs, targets = batch_data["image"].to(device), batch_data["label"].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        kl_div = model.log_module.compute_kl_div()  # Compute KL divergence
        loss = bayesian_loss_fn(outputs, targets, kl_div)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    print(f"Epoch {epoch}, Loss: {total_loss / len(train_loader):.4f}")


# Validation loop
def validate(epoch):
    model.eval()
    total_loss, dice_scores = 0, []
    with torch.no_grad():
        for batch_data in val_loader:
            inputs, targets = batch_data["image"].to(device), batch_data["label"].to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()
            dice_scores.append(metric_fn(outputs, targets).item())
    print(f"Epoch {epoch}, Val Loss: {total_loss / len(val_loader):.4f}, Dice: {np.mean(dice_scores):.4f}")


# Run training
if __name__ == "__main__":
    for epoch in range(1, 201):
        train(epoch)
        validate(epoch)
