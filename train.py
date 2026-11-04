
import re
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from model import SRCNN


# --------------------------------
# 1. Project paths
# --------------------------------

BASE_DIR = Path(__file__).resolve().parent

TRAIN_HR = BASE_DIR / "dataset" / "train" / "HR"
TRAIN_LR = BASE_DIR / "dataset" / "train" / "LR"

VALID_HR = BASE_DIR / "dataset" / "valid" / "HR"
VALID_LR = BASE_DIR / "dataset" / "valid" / "LR"

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


# --------------------------------
# 2. Match HR and LR images
# --------------------------------

def get_image_pairs(hr_folder, lr_folder):

    hr_images = {
        re.match(r"(\d+)", p.stem).group(1): p
        for p in hr_folder.glob("*.png")
        if re.match(r"(\d+)", p.stem)
    }

    lr_images = {
        re.match(r"(\d+)", p.stem).group(1): p
        for p in lr_folder.glob("*.png")
        if re.match(r"(\d+)", p.stem)
    }

    common_ids = sorted(set(hr_images) & set(lr_images))

    pairs = [
        (lr_images[i], hr_images[i])
        for i in common_ids
    ]

    return pairs


# --------------------------------
# 3. Dataset class
# --------------------------------

class SuperResolutionDataset(Dataset):

    def __init__(self, pairs, patch_size=128, training=True):

        self.pairs = pairs
        self.patch_size = patch_size
        self.training = training

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):

        lr_path, hr_path = self.pairs[index]

        lr_image = Image.open(lr_path).convert("RGB")
        hr_image = Image.open(hr_path).convert("RGB")

        # Upscale LR image to HR dimensions
        lr_image = lr_image.resize(
            hr_image.size,
            Image.Resampling.BICUBIC
        )

        width, height = hr_image.size
        size = self.patch_size

        if width < size or height < size:
            raise ValueError("Image is smaller than patch size")

        # Select matching image patches
        if self.training:

            left = np.random.randint(0, width - size + 1)
            top = np.random.randint(0, height - size + 1)

        else:

            left = (width - size) // 2
            top = (height - size) // 2

        box = (left, top, left + size, top + size)

        lr_image = lr_image.crop(box)
        hr_image = hr_image.crop(box)

        # Convert images to arrays
        lr_array = np.asarray(
            lr_image, dtype=np.float32
        ) / 255.0

        hr_array = np.asarray(
            hr_image, dtype=np.float32
        ) / 255.0

        # Convert HWC to CHW
        lr_array = np.transpose(lr_array, (2, 0, 1)).copy()
        hr_array = np.transpose(hr_array, (2, 0, 1)).copy()

        lr_tensor = torch.from_numpy(lr_array)
        hr_tensor = torch.from_numpy(hr_array)

        return lr_tensor, hr_tensor


# --------------------------------
# 4. Load datasets
# --------------------------------

train_pairs = get_image_pairs(TRAIN_HR, TRAIN_LR)
valid_pairs = get_image_pairs(VALID_HR, VALID_LR)

print("Training image pairs:", len(train_pairs))
print("Validation image pairs:", len(valid_pairs))

if len(train_pairs) == 0 or len(valid_pairs) == 0:
    raise ValueError("Dataset not found. Check dataset folders.")

if len(train_pairs) != 200 or len(valid_pairs) != 50:
    raise ValueError("Expected 200 training and 50 validation pairs.")


train_dataset = SuperResolutionDataset(
    train_pairs,
    patch_size=128,
    training=True
)

valid_dataset = SuperResolutionDataset(
    valid_pairs,
    patch_size=128,
    training=False
)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0
)


# --------------------------------
# 5. Create model
# --------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = SRCNN().to(device)

print("Training device:", device)


# --------------------------------
# 6. Loss and optimizer
# --------------------------------

criterion = nn.MSELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.0001
)


# --------------------------------
# 7. Training
# --------------------------------

EPOCHS = 10

for epoch in range(EPOCHS):

    model.train()

    train_loss = 0.0

    for lr_images, hr_images in train_loader:

        lr_images = lr_images.to(device)
        hr_images = hr_images.to(device)

        optimizer.zero_grad()

        outputs = model(lr_images)

        loss = criterion(outputs, hr_images)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    average_train_loss = train_loss / len(train_loader)


    # --------------------------------
    # 8. Validation
    # --------------------------------

    model.eval()

    valid_loss = 0.0

    with torch.no_grad():

        for lr_images, hr_images in valid_loader:

            lr_images = lr_images.to(device)
            hr_images = hr_images.to(device)

            outputs = model(lr_images)

            loss = criterion(outputs, hr_images)

            valid_loss += loss.item()

    average_valid_loss = valid_loss / len(valid_loader)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {average_train_loss:.6f} "
        f"Validation Loss: {average_valid_loss:.6f}"
    )


# --------------------------------
# 9. Save trained model
# --------------------------------

model_path = MODEL_DIR / "super_resolution.pth"

torch.save(model.state_dict(), model_path)

print("\nTraining completed successfully!")
print("Model saved at:", model_path)
