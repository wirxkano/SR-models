import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def calculate_psnr_ssim_y(img, gt_img):
    if img.shape != gt_img.shape:
        h = min(img.shape[0], gt_img.shape[0])
        w = min(img.shape[1], gt_img.shape[1])
        img = img[:h, :w]
        gt_img = gt_img[:h, :w]

    if img.ndim == 3 and img.shape[2] == 3:
        img_y = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        gt_y  = cv2.cvtColor(gt_img, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    else:
        img_y = img
        gt_y  = gt_img

    img_y = img_y.astype(np.float32) / 255.0
    gt_y  = gt_y.astype(np.float32) / 255.0

    psnr_val = peak_signal_noise_ratio(gt_y, img_y, data_range=1.0)
    ssim_val = structural_similarity(gt_y, img_y, data_range=1.0)

    return psnr_val, ssim_val