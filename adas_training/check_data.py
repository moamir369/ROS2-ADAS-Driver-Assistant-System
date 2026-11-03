from pathlib import Path
import cv2
import numpy as np

root = Path.home() / "adas_training" / "data"

for split in ["train", "val"]:
    images = sorted((root / split / "images").glob("*.jpg"))
    masks = sorted((root / split / "masks").glob("*.png"))

    print(f"\n{split}: images={len(images)}, masks={len(masks)}")

    assert len(images) == len(masks), f"Count mismatch in {split}"

    for image_path in images[:5]:
        mask_path = root / split / "masks" / (image_path.stem + ".png")
        assert mask_path.exists(), f"Missing mask: {mask_path}"

        image = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        assert image is not None and mask is not None
        assert image.shape[:2] == (360, 640), image.shape
        assert mask.shape == (360, 640), mask.shape

        print(
            image_path.name,
            "image:", image.shape,
            "mask values:", np.unique(mask).tolist(),
            "positive pixels:", int(np.count_nonzero(mask))
        )

# Save a preview of one validation pair
image_path = sorted((root / "val" / "images").glob("*.jpg"))[0]
mask_path = root / "val" / "masks" / (image_path.stem + ".png")

image = cv2.imread(str(image_path))
mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

overlay = image.copy()
overlay[mask > 0] = (0, 255, 0)
overlay = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

preview_path = root / "val" / "preview_overlay.jpg"
cv2.imwrite(str(preview_path), overlay)

print("\nPreview saved:", preview_path)
print("Data checks passed.")