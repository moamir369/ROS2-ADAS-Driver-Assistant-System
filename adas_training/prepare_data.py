
import zipfile
import json
import random
from pathlib import Path

import cv2
import numpy as np

ARCHIVE = Path("/home/emir/lane_dataset/archive.zip")
OUT = Path.home() / "adas_training" / "data"
W, H = 640, 360
SEED = 42

TRAIN_LABEL = "TUSimple/train_set/label_data_0313.json"
VAL_LABELS = [
    "TUSimple/train_set/label_data_0531.json",
    "TUSimple/train_set/label_data_0601.json",
]

random.seed(SEED)

def read_records(z, label_path):
    with z.open(label_path) as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]

def make_mask(record):
    mask = np.zeros((H, W), dtype=np.uint8)

    for lane in record["lanes"]:
        points = []
        for x, y in zip(lane, record["h_samples"]):
            if x >= 0 and y >= 0:
                # تحويل الإحداثيات من 1280x720 إلى 640x360
                px = int(round(x * W / 1280))
                py = int(round(y * H / 720))

                if 0 <= px < W and 0 <= py < H:
                    points.append((px, py))
                else:
                    if len(points) >= 2:
                        cv2.polylines(
                            mask, [np.array(points, np.int32)],
                            False, 255, thickness=4
                        )
                    points = []
            else:
                if len(points) >= 2:
                    cv2.polylines(
                        mask, [np.array(points, np.int32)],
                        False, 255, thickness=4
                    )
                points = []

        if len(points) >= 2:
            cv2.polylines(
                mask, [np.array(points, np.int32)],
                False, 255, thickness=4
            )

    return mask

def save_split(z, records, split, count):
    image_dir = OUT / split / "images"
    mask_dir = OUT / split / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    random.shuffle(records)
    selected = records[:count]
    saved = 0

    for i, record in enumerate(selected):
        internal_path = "TUSimple/train_set/" + record["raw_file"]

        try:
            data = np.frombuffer(z.read(internal_path), np.uint8)
        except KeyError:
            print("Missing image:", internal_path)
            continue

        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            print("Unreadable image:", internal_path)
            continue

        image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)
        mask = make_mask(record)

        filename = f"{split}_{i:04d}"
        if not cv2.imwrite(str(image_dir / f"{filename}.jpg"), image):
            raise RuntimeError(f"Failed to save image {filename}")
        if not cv2.imwrite(str(mask_dir / f"{filename}.png"), mask):
            raise RuntimeError(f"Failed to save mask {filename}")

        saved += 1

    print(f"{split}: saved {saved}/{count} image-mask pairs")

with zipfile.ZipFile(ARCHIVE, "r") as z:
    train_records = read_records(z, TRAIN_LABEL)
    val_records = []

    for label_path in VAL_LABELS:
        val_records.extend(read_records(z, label_path))

    save_split(z, train_records, "train", 400)
    save_split(z, val_records, "val", 100)

print("\nOutput directory:", OUT)
print("Done.")
