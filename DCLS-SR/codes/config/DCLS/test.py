import argparse
import logging
import os.path
import sys
import time
from collections import OrderedDict
import torchvision.utils as tvutils
import matplotlib.pyplot as plt

import numpy as np
import torch
from IPython import embed

import codes.config.DCLS.options as option
from codes.config.DCLS.models import create_model

sys.path.insert(0, "../../")
import codes.utils as util
from codes.data import create_dataloader, create_dataset
# from codes.data.util import bgr2ycbcr

from skimage.metrics import peak_signal_noise_ratio as skimg_psnr
from skimage.metrics import structural_similarity as skimg_ssim
from skimage.color import rgb2ycbcr
import pyiqa
import cv2

#### options
parser = argparse.ArgumentParser()
parser.add_argument("-opt", type=str, required=True, help="Path to options YMAL file.")
opt = option.parse(parser.parse_args().opt, is_train=False)

opt = option.dict_to_nonedict(opt)

#### mkdir and logger
util.mkdirs(
    (
        path
        for key, path in opt["path"].items()
        if not key == "experiments_root"
        and "pretrain_model" not in key
        and "resume" not in key
    )
)

os.system("rm ./result")
# os.symlink(os.path.join(opt["path"]["results_root"], ".."), "./result")

util.setup_logger(
    "base",
    opt["path"]["log"],
    "test_" + opt["name"],
    level=logging.INFO,
    screen=True,
    tofile=True,
)
logger = logging.getLogger("base")
logger.info(option.dict2str(opt))

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt["datasets"].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info(
        "Number of test images in [{:s}]: {:d}".format(
            dataset_opt["name"], len(test_set)
        )
    )
    test_loaders.append(test_loader)

# load pretrained model by default
model = create_model(opt)
device = torch.device('cuda')
lpips_metric = pyiqa.create_metric('lpips', device=device)
niqe_metric = pyiqa.create_metric('niqe', device=device)

for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt["name"]  # path opt['']
    logger.info("\nTesting [{:s}]...".format(test_set_name))
    test_start_time = time.time()
    dataset_dir = os.path.join("/root/quoc-huy/all-tested-results/DCLS", test_set_name)
    util.mkdir(dataset_dir)

    test_results = OrderedDict()
    test_results["psnr"] = []
    test_results["ssim"] = []
    test_results["psnr_y"] = []
    test_results["ssim_y"] = []
    test_results["lpips"] = []
    test_results["niqe"] = []
    test_times = []

    for test_data in test_loader:
        single_img_psnr = []
        single_img_ssim = []
        single_img_psnr_y = []
        single_img_ssim_y = []
        need_GT = False if test_loader.dataset.opt["dataroot_GT"] is None else True
        img_path = test_data["GT_path"][0] if need_GT else test_data["LQ_path"][0]
        # img_name = img_path
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        #### input dataset_LQ
        model.feed_data(test_data["LQ"], test_data["GT"])
        tic = time.time()
        model.test()
        toc = time.time()
        test_times.append(toc - tic)

        visuals = model.get_current_visuals()
        SR_img = visuals["Batch_SR"]
        sr_img = util.tensor2img(visuals["SR"].squeeze())  # uint8

        suffix = opt["suffix"]
        if suffix:
            save_img_path = os.path.join(dataset_dir, 'imgs', img_name + suffix + ".png")
            save_blur_path = os.path.join(dataset_dir, 'kernels', img_name + suffix + ".png")
        else:
            save_img_path = os.path.join(dataset_dir, 'imgs', img_name + ".png")
            save_blur_path = os.path.join(dataset_dir, 'kernels', img_name + ".png")
        util.save_img(sr_img, save_img_path)
        
        ### Kernel
        blur_kernel = visuals["ker"].squeeze(0).detach().cpu().numpy()
        print('>>', blur_kernel.max(), blur_kernel.min())
        k_vis = np.clip(blur_kernel, 0, 1)
        k_vis = k_vis / k_vis.max()
        
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(k_vis, cmap='gray', interpolation='nearest')
        ax.axis('off')

        fig.savefig(
            save_blur_path,
            dpi=350,
            bbox_inches='tight',
            pad_inches=0
        )
        plt.close(fig)

        if need_GT:
            gt_img = util.tensor2img(visuals["GT"].squeeze())
            gt_img = gt_img / 255.0
            sr_img = sr_img / 255.0
            
            crop_border = opt["crop_border"] if opt["crop_border"] else opt["scale"]
            if crop_border == 0:
                cropped_sr_img = sr_img
                cropped_gt_img = gt_img
            else:
                cropped_sr_img = sr_img[
                    crop_border:-crop_border, crop_border:-crop_border
                ]
                cropped_gt_img = gt_img[
                    crop_border:-crop_border, crop_border:-crop_border
                ]

            psnr = util.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)
            ssim = util.calculate_ssim(cropped_sr_img * 255, cropped_gt_img * 255)

            test_results["psnr"].append(psnr)
            test_results["ssim"].append(ssim)

            if len(gt_img.shape) == 3:
                if gt_img.shape[2] == 3:  # RGB image
                    sr_img_y = rgb2ycbcr(cropped_sr_img)[:, :, 0]
                    gt_img_y = rgb2ycbcr(cropped_gt_img)[:, :, 0]
                    
                    psnr_y = skimg_psnr(sr_img_y, gt_img_y, data_range=255.0)
                    ssim_y = skimg_ssim(sr_img_y, gt_img_y, data_range=255.0)

                    test_results["psnr_y"].append(psnr_y)
                    test_results["ssim_y"].append(ssim_y)
                    sr_tensor = torch.from_numpy(cropped_sr_img.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
                    gt_tensor = torch.from_numpy(cropped_gt_img.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
                    
                    sr_tensor = torch.clamp(sr_tensor, 0, 1)
                    gt_tensor = torch.clamp(gt_tensor, 0, 1)
                    lpips_val = lpips_metric(sr_tensor, gt_tensor).item()
                    test_results['lpips'].append(lpips_val)
                    niqe_val = niqe_metric(sr_tensor).item()
                    test_results['niqe'].append(niqe_val)

                    logger.info(
                        "img:{:15s} - PSNR: {:.6f} dB; SSIM: {:.6f}; LPIPS: {:.4f}; NIQE: {:.4f}.".format(
                            img_name, psnr_y, ssim_y, lpips_val, niqe_val
                        )
                    )
            else:
                logger.info(
                    "img:{:15s} - PSNR: {:.6f} dB; SSIM: {:.6f}.".format(
                        img_name, psnr, ssim
                    )
                )

                test_results["psnr_y"].append(psnr)
                test_results["ssim_y"].append(ssim)
        else:
            logger.info(img_name)
            
    fid_metric = pyiqa.create_metric('fid', device=device)
    hr_dir = opt["datasets"]["test4"]["dataroot_GT"]
    fid_val = fid_metric(dataset_dir + "/imgs", hr_dir).item()

    ave_psnr = sum(test_results["psnr"]) / len(test_results["psnr"])
    ave_ssim = sum(test_results["ssim"]) / len(test_results["ssim"])
    logger.info(
        "----Average PSNR/SSIM results for {}----\n\tPSNR: {:.6f} dB; SSIM: {:.6f}\n".format(
            test_set_name, ave_psnr, ave_ssim
        )
    )
    if test_results["psnr_y"] and test_results["ssim_y"]:
        ave_psnr_y = sum(test_results["psnr_y"]) / len(test_results["psnr_y"])
        ave_ssim_y = sum(test_results["ssim_y"]) / len(test_results["ssim_y"])
        ave_lpips = sum(test_results['lpips']) / len(test_results['lpips'])
        ave_niqe = sum(test_results['niqe']) / len(test_results['niqe'])
        logger.info(
            "----Y channel, average PSNR/SSIM----\n\tPSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}\n".format(
                ave_psnr_y, ave_ssim_y
            )
        )
        
        logger.info(
            "----LPIPS: {:.4f}; FID: {:.4f}; NIQE: {:.4f}\n".format(
                ave_lpips, fid_val, ave_niqe
            )
        )

    print(f"average test time: {np.mean(test_times):.4f}")
