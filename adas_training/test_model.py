
from pathlib import Path
import cv2
import numpy as np
import torch

from train_cnn import LaneCNN

root = Path.home() / "adas_training"
data = root / "data" / "val"
checkpoint_path = root / "best_lane_cnn.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(
    checkpoint_path, map_location=device, weights_only=True
)

model = LaneCNN().to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

images = sorted((data / "images").glob("*.jpg"))
if not images:
    raise RuntimeError("No validation images found")

# اختيار صورة Validation
image_path = images[0]
mask_path = data / "masks" / (image_path.stem + ".png")

bgr = cv2.imread(str(image_path))
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
rgb = rgb.astype(np.float32) / 255.0

x = torch.from_numpy(rgb.transpose(2, 0, 1).copy())
x = x.unsqueeze(0).to(device)

with torch.no_grad():
    logits = model(x)
    probabilities = torch.sigmoid(logits)[0, 0].cpu().numpy()

pred_mask = (probabilities > 0.5).astype(np.uint8) * 255
true_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

overlay = bgr.copy()
overlay[pred_mask > 0] = (0, 255, 0)
overlay = cv2.addWeighted(bgr, 0.7, overlay, 0.3, 0)

comparison = np.hstack([
    bgr,
    cv2.cvtColor(true_mask, cv2.COLOR_GRAY2BGR),
    cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR),
])

cv2.imwrite(str(root / "prediction_overlay.jpg"), overlay)
cv2.imwrite(str(root / "prediction_comparison.jpg"), comparison)

intersection = np.logical_and(pred_mask > 0, true_mask > 127).sum()
union = np.logical_or(pred_mask > 0, true_mask > 127).sum()
iou = intersection / union if union else 1.0

print("Loaded checkpoint epoch:", checkpoint["epoch"])
print("Saved validation IoU:", checkpoint["val_iou"])
print("Test image:", image_path.name)
print("Image IoU:", round(float(iou), 4))
print("Prediction overlay:", root / "prediction_overlay.jpg")
print("Comparison:", root / "prediction_comparison.jpg")

