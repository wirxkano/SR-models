import cv2
import numpy as np
from pathlib import Path

def downsample_hr_folder(
    hr_folder,
    lr_folder,
    scale=4,
    interpolation=cv2.INTER_CUBIC
):
    hr_folder = Path(hr_folder)
    lr_folder = Path(lr_folder)
    lr_folder.mkdir(parents=True, exist_ok=True)

    image_exts = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']

    hr_images = sorted([f for f in hr_folder.iterdir() if f.suffix.lower() in image_exts])

    print(f"Downsampling {len(hr_images)} images (x{scale})")

    for img_path in hr_images:
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"Failed to read {img_path.name}")
            continue

        h, w = img.shape[:2]
        lr_w, lr_h = w // scale, h // scale

        lr_img = cv2.resize(
            img,
            (lr_w, lr_h),
            interpolation=interpolation
        )
        lr_img = np.clip(lr_img, 0, 255).astype(np.uint8)

        save_path = lr_folder / (img_path.stem + '.png')
        cv2.imwrite(str(save_path), lr_img)

    print("Done.")

datasetname = "Urban100"
downsample_hr_folder(
    hr_folder=f'/root/media/quoc-huy/eval-wir/srbenchmarks/{datasetname}/HR',
    lr_folder=f'/root/media/quoc-huy/eval-wir/srbenchmarks/{datasetname}/LRbic_opencv_x4_v2',
    scale=4
)

