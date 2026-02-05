import cv2
import numpy as np
from pathlib import Path
import torch
import pyiqa
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def img2tensor(img, device):
    # img: HWC, BGR, uint8 [0,255]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    return img


def bgr2y(image):
    """Convert BGR image to Y channel (uint8)."""
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    return ycrcb[:, :, 0]


def calculate_psnr_ssim_y(hr_img, sr_img):
    """PSNR / SSIM on Y channel."""
    assert hr_img.shape == sr_img.shape, "HR and SR must have same shape"

    hr_y = bgr2y(hr_img)
    sr_y = bgr2y(sr_img)

    psnr_val = psnr(hr_y, sr_y, data_range=255)
    ssim_val = ssim(hr_y, sr_y, data_range=255)

    return psnr_val, ssim_val


def process_image_folders(hr_folder, sr_folder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lpips_metric = pyiqa.create_metric('lpips', device=device)
    niqe_metric = pyiqa.create_metric('niqe', device=device)
    fid_metric = pyiqa.create_metric('fid', device=device)

    hr_path = Path(hr_folder)
    sr_path = Path(sr_folder)

    image_exts = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    hr_images = sorted([f for f in hr_path.iterdir() if f.suffix.lower() in image_exts])
    sr_images = sorted([f for f in sr_path.iterdir() if f.suffix.lower() in image_exts])

    if len(hr_images) != len(sr_images):
        print(f"Warning: HR ({len(hr_images)}) != SR ({len(sr_images)})")

    results = []
    print(f"Processing {len(sr_images)} image pairs...\n")

    for hr_file, sr_file in zip(hr_images, sr_images):
        hr_img = cv2.imread(str(hr_file))
        sr_img = cv2.imread(str(sr_file))

        if hr_img is None or sr_img is None:
            print(f"Error reading: {hr_file.name} or {sr_file.name}")
            continue

        if hr_img.shape != sr_img.shape:
            print(f"Shape mismatch: {hr_file.name}")
            continue

        psnr_val, ssim_val = calculate_psnr_ssim_y(hr_img, sr_img)

        hr_tensor = img2tensor(hr_img, device)
        sr_tensor = img2tensor(sr_img, device)

        with torch.no_grad():
            lpips_score = lpips_metric(sr_tensor, hr_tensor).item()
            niqe_score = niqe_metric(sr_tensor).item()

        results.append({
            'name': hr_file.name,
            'psnr': psnr_val,
            'ssim': ssim_val,
            'lpips': lpips_score,
            'niqe': niqe_score
        })

        print(f"{sr_file.name}")
        print(f"  PSNR (Y): {psnr_val:.4f} dB")
        print(f"  SSIM (Y): {ssim_val:.4f}")
        print(f"  LPIPS:   {lpips_score:.4f}")
        print(f"  NIQE:    {niqe_score:.4f}\n")

    # FID: directory-based
    with torch.no_grad():
        fid_score = fid_metric(str(sr_path), str(hr_path)).item()

    if results:
        avg_psnr = np.mean([r['psnr'] for r in results])
        avg_ssim = np.mean([r['ssim'] for r in results])
        avg_lpips = np.mean([r['lpips'] for r in results])
        avg_niqe = np.mean([r['niqe'] for r in results])

        print("=" * 50)
        print(f"Average PSNR (Y): {avg_psnr:.2f} dB")
        print(f"Average SSIM (Y): {avg_ssim:.4f}")
        print(f"Average LPIPS:   {avg_lpips:.4f}")
        print(f"Average NIQE:    {avg_niqe:.4f}")
        print(f"FID:             {fid_score:.4f}")
        print("=" * 50)

    return results


if __name__ == "__main__":
    dataset = "BSD100"

    hr_folder = f"/root/media/quoc-huy/eval-wir/srbenchmarks/{dataset}/HR"
    sr_folder = f"/root/quoc-huy/all-tested-results/metakernelgan/{dataset}"

    results = process_image_folders(hr_folder, sr_folder)
