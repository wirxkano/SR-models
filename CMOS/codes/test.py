import os.path
import logging
import argparse
from collections import OrderedDict
import numpy as np
import torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import cv2
import options.options as option
import utils.util as util
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader
from models import create_model
from torchvision.transforms import ToPILImage
import utils.util as util
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import pyiqa
from skimage.color import rgb2ycbcr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#### options
parser = argparse.ArgumentParser()
parser.add_argument('--opt', type=str, default='options/test/NYUv2_BSR/test_stage3.yml', help='Path to options YMAL file.')
args = parser.parse_args()
opt = option.parse(args.opt, is_train=False)
opt = option.dict_to_nonedict(opt)
device_id = torch.cuda.current_device()

#### mkdir and logger
util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
             and 'pretrain_model' not in key and 'resume' not in key))
util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
                  screen=True, tofile=True)
logger = logging.getLogger('base')
logger.info(option.dict2str(opt))

# set random seed
util.set_random_seed(0)

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt['datasets'].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
    test_loaders.append(test_loader)

# load pretrained model by default
model = create_model(opt)

for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt['name']  # path opt['']
    logger.info('\nTesting [{:s}]...'.format(test_set_name))
    save_dir = "/root/quoc-huy/all-tested-results/CMOS/" + test_set_name
    dataset_dir = os.path.join("/root/quoc-huy/all-tested-results/CMOS", test_set_name)
    util.mkdir(dataset_dir)
    
    #### preprocessing for LR_img and kernel map
    prepro = util.SRMDPreprocessing(
        opt["scale"],
        random=False,
        l=opt["kernel_size"],
        add_noise=opt["test_noise"],
        noise_high=opt["noise"] / 255.0,
        add_jpeg=opt["test_jpeg"],
        jpeg_low=opt["jpeg"],
        rate_cln=-1,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        sig=opt["sig"],
        sig1=opt["sig1"],
        sig2=opt["sig2"],
        theta=opt["theta"],
        sig_min=opt["sig_min"],
        sig_max=opt["sig_max"],
        rate_iso=opt["rate_iso"],
        is_training=False,
        sv_mode=opt["sv_mode"],
    )
    
    lpips_metric = pyiqa.create_metric('lpips', device='cpu')
    fid_metric = pyiqa.create_metric('fid', device=device)
    niqe_metric = pyiqa.create_metric('niqe', device='cpu')

    n_classes = opt['n_classes']
    avg_psnr_y = 0.0
    avg_ssim_y = 0.0
    avg_lpips = 0.0
    avg_niqe = 0.0
    avg_psnr_b = 0.0
    avg_ssim_b = 0.0
    avg_miou = 0.0
    tp = [0] * n_classes
    fp = [0] * n_classes
    fn = [0] * n_classes
    idx = 0
    n_skipped = 0
    for test_data in test_loader:
        idx += 1
        
        real_image = True if test_loader.dataset.opt["dataroot_GT"] is None else False
        # print('>>>>',test_loader.dataset.opt)
        generate_online = (
            True
            if test_loader.dataset.opt["dataroot_GT"] is not None
            and test_loader.dataset.opt["dataroot_LR"] is None
            else False
        )
        img_path = test_data["LR_path"][0] if real_image else test_data["GT_path"][0]
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        if real_image:
            LR_img, LR_n_img, blur_map, kernel = (
                test_data["LR"],
                test_data["LR"],
                torch.ones(1, 1, 1),
                torch.ones(1, 1, opt["kernel_size"], opt["kernel_size"]),
            )
        elif generate_online:
            test_data["GT"] = test_data["GT"].to(device)
            LR_img, LR_n_img, blur_map, kernel = prepro(test_data["GT"], kernel=True)
            # print('blur size', blur_map.cpu().shape)
            # print('GT size', test_data["GT"].cpu().shape)
            # print('LR size', LR_img.shape)
            
            h, w = LR_img.shape[-2:]   # get height and width
            if w < 96:
                n_skipped += 1
                print(f"⚠️ Skipped: LR image too small ({h}x{w})")
                continue  # skip to next iteration
        else:
            # note that it is not sutible for non-blind testing! because kernel is zero by default
            LR_img, LR_n_img, blur_map, kernel = (
                test_data["LR"],
                test_data["LR"],
                torch.ones(1, 1, 1),
                torch.ones(1, 1, opt["kernel_size"], opt["kernel_size"]),
            )
        
        # Save SR images for reference
        img_name = os.path.splitext(os.path.basename(test_data['GT_path'][0]))[0]
        print(img_name)
        # GT_img, LR_img, blur_map, seg_map = \
        #     test_data['GT'], test_data['LR'], test_data['blur_map'], test_data['seg_map']
        GT_img = test_data['GT']
        seg_map = np.zeros_like(blur_map.cpu())
        seg_map = torch.from_numpy(np.ascontiguousarray(np.expand_dims(seg_map, axis=0))).squeeze(-1).float()
        model.feed_data(GT_img, LR_img, blur_map, seg_map)
        model.test()

        visuals = model.get_current_visuals()

        # deal with blur
        if opt['val_blur']:
            pass
            save_blur_path = os.path.join(dataset_dir, 'blur')
            util.mkdir(save_blur_path)
            save_blur_path = os.path.join(save_blur_path, '{:s}.png'.format(img_name))
            blur_map_est = visuals['blur_map_est'][0] / ((opt['kernel_size'] - 1) / 4.0)
            blur_map = visuals['blur_map'][0] / ((opt['kernel_size'] - 1) / 4.0)
            cv2.imwrite(save_blur_path, np.vstack((blur_map_est, blur_map)) * 255.)
            avg_psnr_b += util.calculate_psnr(blur_map_est, blur_map, 1.)
            avg_ssim_b += util.calculate_ssim(blur_map_est * 255., blur_map * 255.)

        # deal with seg
        if opt['val_seg']:
            pass
            save_seg_path = os.path.join(dataset_dir, 'seg')
            util.mkdir(save_seg_path)
            save_seg_path = os.path.join(save_seg_path, '{:s}.png'.format(img_name))
            _, seg_map_est = torch.max(visuals['seg_map_est'], dim=0)
            seg_map = visuals['seg_map']
            seg_map_est = np.array(seg_map_est).astype(np.uint8).astype(np.float32)  # visualization
            label_color = util.Colorize()(seg_map_est)
            label_save = ToPILImage()(label_color)
            label_save.save(save_seg_path)
            valid = (seg_map != 255)
            for i_part in range(0, n_classes):
                tmp_gt = (seg_map == i_part)
                tmp_pred = (seg_map_est == i_part)
                tp[i_part] += np.sum(tmp_gt & tmp_pred & valid)
                fp[i_part] += np.sum(~tmp_gt & tmp_pred & valid)
                fn[i_part] += np.sum(tmp_gt & ~tmp_pred & valid)

        # deal with sr image
        if opt['val_sr']:
            sr_img = util.tensor2img(visuals['SR']) 
            gt_img = util.tensor2img(visuals['GT'])
            save_img_path = os.path.join(dataset_dir, '{:s}.png'.format(img_name))
            util.save_img(sr_img, save_img_path)

            # calculate PSNR
            gt_img = gt_img / 255.
            sr_img = sr_img / 255.
            if gt_img.shape[2] == 3:  # RGB image
                # sr_img_y = bgr2ycbcr(sr_img, only_y=True)
                # gt_img_y = bgr2ycbcr(gt_img, only_y=True)
                sr_img = sr_img[:, :, ::-1]
                gt_img = gt_img[:, :, ::-1]
                sr_img_y = rgb2ycbcr(sr_img)[:, :, 0]
                gt_img_y = rgb2ycbcr(gt_img)[:, :, 0]
                
                h = min(sr_img_y.shape[0], gt_img_y.shape[0])
                w = min(sr_img_y.shape[1], gt_img_y.shape[1])
                
                crop_border = opt["crop_border"] if opt["crop_border"] else opt["scale"]

                # optional border crop first
                if crop_border > 0:
                    sr_img_y = sr_img_y[crop_border:-crop_border, crop_border:-crop_border]
                    gt_img_y = gt_img_y[crop_border:-crop_border, crop_border:-crop_border]

                # now align to the same size (bottom-right crop)
                h = min(sr_img_y.shape[0], gt_img_y.shape[0])
                w = min(sr_img_y.shape[1], gt_img_y.shape[1])

                cropped_sr_img_y = sr_img_y[0:h, 0:w]
                cropped_gt_img_y = gt_img_y[0:h, 0:w]
                psnr = peak_signal_noise_ratio(cropped_gt_img_y, cropped_sr_img_y, data_range=255)
                ssim = structural_similarity(cropped_gt_img_y, cropped_sr_img_y, data_range=255)
                # print('PSNR and SSIM', psnr, ssim)
                
                cropped_sr_img = sr_img[0:h, 0:w]
                cropped_gt_img = gt_img[0:h, 0:w]
                
                sr_rgb = torch.from_numpy(cropped_sr_img.copy()).permute(2, 0, 1).float().unsqueeze(0)
                gt_rgb = torch.from_numpy(cropped_gt_img.copy()).permute(2, 0, 1).float().unsqueeze(0)
                
                lpips_score = lpips_metric(sr_rgb, gt_rgb).item()
                niqe_score = niqe_metric(sr_rgb).item()
                # print('LPIPS:', lpips_score, 'NIQE:', niqe_score)

                avg_psnr_y += psnr
                avg_ssim_y += ssim
                avg_lpips += lpips_score
                avg_niqe += niqe_score
                
                del visuals
                torch.cuda.empty_cache()
                
        print(f"Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        print(f"Reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")          
    
    print('total img calculated', idx-n_skipped)
    avg_psnr_y = avg_psnr_y / (idx - n_skipped)
    avg_ssim_y = avg_ssim_y / (idx - n_skipped)
    avg_lpips = avg_lpips / (idx - n_skipped)
    avg_niqe = avg_niqe / (idx - n_skipped)
    avg_psnr_b = avg_psnr_b / idx
    avg_ssim_b = avg_ssim_b / idx
    jac = [0] * n_classes
    for i_part in range(0, n_classes):
        jac[i_part] = float(tp[i_part]) / max(float(tp[i_part] + fp[i_part] + fn[i_part]), 1e-8)
    avg_miou = np.mean(jac)
    
    fid_score = fid_metric(save_dir, test_loader.dataset.opt["dataroot_GT"])

    # log
    logger.info('# {}, Test # PSNR_Y: {:.2f}, SSIM_Y: {:.4f}, LPIPS: {:.4f}, FID: {:.4f}, NIQE: {:.4f} '.format(opt['name'], avg_psnr_y, avg_ssim_y, avg_lpips, fid_score, avg_niqe))
