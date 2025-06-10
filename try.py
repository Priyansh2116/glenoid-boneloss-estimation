import os
import glob
import torch
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    ScaleIntensityd, RandCropByPosNegLabeld, RandFlipd, ToTensord, Compose
)
from monai.data import Dataset, DataLoader
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import SimpleInferer
from monai.utils import set_determinism
from torch.optim import Adam

set_determinism(seed=42)

# === DATASET PATH ===
root_dir = "./glenoid/dataset/"
patients = sorted(os.listdir(root_dir))

# === Build patient list ===
data_dicts = [
    {
        "image": os.path.join(root_dir, p, "image.nii.gz"),
        "label": os.path.join(root_dir, p, "label.nii.gz")
    } for p in patients
]

# === Split into train/val (80/20) ===
val_split = int(len(data_dicts) * 0.8)
train_files, val_files = data_dicts[:val_split], data_dicts[val_split:]

# === Define transforms ===
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    ScaleIntensityd(keys="image"),
    RandCropByPosNegLabeld(
    keys=["image", "label"],
    label_key="label",
    spatial_size=(96, 96, 96),
    pos=1,
    neg=1,
    num_samples=4,
    ),
    RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=0),
    ToTensord(keys=["image", "label"]),
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    ScaleIntensityd(keys="image"),
    ToTensord(keys=["image", "label"]),
])

# === Dataloaders ===
train_ds = Dataset(data=train_files, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2)

val_ds = Dataset(data=val_files, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=2)

# === Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)

loss_function = DiceLoss(to_onehot_y=True, softmax=True)
optimizer = Adam(model.parameters(), 1e-4)
dice_metric = DiceMetric(include_background=False, reduction="mean")
inferer = SimpleInferer()

# === Training loop ===
max_epochs = 100
val_interval = 5
best_metric = -1
best_model_path = "best_model.pth"

for epoch in range(max_epochs):
    print(f"Epoch {epoch+1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    for batch in train_loader:
        inputs, labels = batch["image"].to(device), batch["label"].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss /= len(train_loader)
    print(f"Train Loss: {epoch_loss:.4f}")

    # === Validation ===
    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            dice_scores = []
            for val_batch in val_loader:
                val_inputs = val_batch["image"].to(device)
                val_labels = val_batch["label"].to(device)
                val_outputs = inferer(inputs=val_inputs, network=model)
                val_outputs = torch.argmax(val_outputs, dim=1, keepdim=True)
                dice_metric(y_pred=val_outputs, y=val_labels)
            metric = dice_metric.aggregate().item()
            dice_metric.reset()
            print(f"Validation Dice: {metric:.4f}")
            if metric > best_metric:
                best_metric = metric
                torch.save(model.state_dict(), best_model_path)
                print("💾 Saved best model!")

print("✅ Training complete.")

