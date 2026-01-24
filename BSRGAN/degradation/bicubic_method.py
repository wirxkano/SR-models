import os
import cv2
import numpy as np

def generate_LR_images(hr_folder, lr_folder, sf=4):
    """
    Generate LR images (bicubic downsampling) from HR images.
    
    Args:
        hr_folder (str): Path to folder containing HR images.
        lr_folder (str): Path to save LR images.
        sf (int): Scale factor (e.g., 2, 3, 4).
    """
    if not os.path.exists(lr_folder):
        os.makedirs(lr_folder)

    img_list = [f for f in os.listdir(hr_folder)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    print(f"Found {len(img_list)} images in {hr_folder}")

    for idx, img_name in enumerate(img_list, 1):
        img_path = os.path.join(hr_folder, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        if img is None:
            print(f"[Warning] Cannot read {img_name}, skipping.")
            continue

        # Crop to make divisible by sf
        h, w = img.shape[:2]
        h_cropped = h - h % sf
        w_cropped = w - w % sf
        img = img[:h_cropped, :w_cropped, :]

        # Bicubic downsampling
        lr = cv2.resize(img, (w_cropped // sf, h_cropped // sf),
                        interpolation=cv2.INTER_CUBIC)

        save_path = os.path.join(lr_folder, img_name)
        cv2.imwrite(save_path, lr)

        # print(f"[{idx}/{len(img_list)}] Saved: {save_path} (HR: {w_cropped}x{h_cropped} → LR: {lr.shape[1]}x{lr.shape[0]})")

    print("✅ Done generating LR images!")


if __name__ == "__main__":
    datasetname = "Urban100"
    hr_folder = f"/root/media/quoc-huy/eval-wir/srbenchmarks/{datasetname}/HR/"
    lr_folder = f"/root/media/quoc-huy/blur-DAN/{datasetname}/LRbic/"
    scale_factor = 4

    generate_LR_images(hr_folder, lr_folder, sf=scale_factor)
