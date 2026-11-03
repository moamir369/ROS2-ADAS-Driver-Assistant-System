
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# إعدادات التدريب
ROOT = Path.home() / "adas_training" / "data"
OUTPUT = Path.home() / "adas_training" / "best_lane_cnn.pth"

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 360
BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 1e-3

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


class LaneDataset(Dataset):
    def __init__(self, split):
        self.image_dir = ROOT / split / "images"
        self.mask_dir = ROOT / split / "masks"

        self.images = sorted(self.image_dir.glob("*.jpg"))
        self.samples = []

        for image_path in self.images:
            mask_path = self.mask_dir / (image_path.stem + ".png")
            if mask_path.exists():
                self.samples.append((image_path, mask_path))

        if not self.samples:
            raise RuntimeError(f"No image-mask pairs found for {split}")

        print(f"{split}: {len(self.samples)} pairs")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)

        # HWC -> CHW
        image = torch.from_numpy(image.transpose(2, 0, 1).copy())
        mask = torch.from_numpy(mask[None, :, :].copy())

        return image, mask


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class LaneCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = ConvBlock(3, 16)
        self.enc2 = ConvBlock(16, 32)
        self.enc3 = ConvBlock(32, 64)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(64, 96)
        
        
        #Decoder
        self.up3 = nn.ConvTranspose2d(96, 64, 2, stride=2)
        self.dec3 = ConvBlock(128, 64)

        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = ConvBlock(64, 32)

        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = ConvBlock(32, 16)

        self.output = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.output(d1)


def calculate_metrics(logits, masks):
    
    #إذا القيمة أكبر من 0.5، منعتبر البكسل ضمن المسار، وإذا مو أكبر من 0.5، منعتبره خلفية
    predictions = torch.sigmoid(logits) > 0.5
    targets = masks > 0.5

    intersection = (predictions & targets).sum(dim=(1, 2, 3)).float()
    union = (predictions | targets).sum(dim=(1, 2, 3)).float()

    iou = ((intersection + 1e-6) / (union + 1e-6)).mean()
    return iou.item()


def run_epoch(model, loader, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_iou = 0.0
    count = 0

    criterion = nn.BCEWithLogitsLoss()

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                logits = model(images)
                loss = criterion(logits, masks)

            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        total_loss += loss.item() * images.size(0)
        total_iou += calculate_metrics(logits.detach(), masks) * images.size(0)
        count += images.size(0)

    return total_loss / count, total_iou / count


def main():
    train_set = LaneDataset("train")
    val_set = LaneDataset("val")

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=(device.type == "cuda")
    )

    model = LaneCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_iou = -1.0

    print(f"\nStarting training: {EPOCHS} epochs")
    print("Input size: 640x360 | Batch size:", BATCH_SIZE)

    for epoch in range(1, EPOCHS + 1):
        start = time.time()

        train_loss, train_iou = run_epoch(
            model, train_loader, optimizer, scaler
        )
        val_loss, val_iou = run_epoch(model, val_loader)

        elapsed = time.time() - start

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"train_IoU={train_iou:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_IoU={val_iou:.4f} | "
            f"time={elapsed:.1f}s",
            flush=True,
        )

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "image_width": IMAGE_WIDTH,
                    "image_height": IMAGE_HEIGHT,
                    "epoch": epoch,
                    "val_iou": val_iou,
                },
                OUTPUT,
            )
            print(f"  Saved best model: {OUTPUT}", flush=True)

    print(f"\nTraining complete. Best validation IoU: {best_iou:.4f}")
    print("Best model:", OUTPUT)


if __name__ == "__main__":
    main()
