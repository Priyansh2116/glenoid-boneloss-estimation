from monai.transforms import LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd
from monai.data import Dataset, DataLoader
import os

# Your image-label pairs (change paths as needed)
data_dicts = [
    {"image": "./dataset/patient1/image.nrrd", "label": "./dataset/patient1/label.nrrd"},
    {"image": "./dataset/patient2/image.nrrd", "label": "./dataset/patient2/label.nrrd"},
    # Add all patients here
]

# Transforms to load and standardize data
transforms = [
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
]

from monai.transforms import Compose
val_transforms = Compose(transforms)

# Dataset and DataLoader
dataset = Dataset(data=data_dicts, transform=val_transforms)
loader = DataLoader(dataset, batch_size=1)

# Iterate and check shape
for i, batch in enumerate(loader):
    image = batch["image"]
    label = batch["label"]
    print(f"Patient {i+1}: image shape {image.shape}, label shape {label.shape}")
    assert image.shape == label.shape, f"Shape mismatch in patient {i+1}"

