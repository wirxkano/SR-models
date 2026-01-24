import os.path
import logging
import torch

from utils import utils_logger
from utils import utils_image as util
# from utils import utils_model
from models.network_rrdbnet import RRDBNet as net

import argparse
import cv2
import pyiqa
import numpy as np


"""
Spyder (Python 3.6-3.7)
PyTorch 1.4.0-1.8.1
Windows 10 or Linux
Kai Zhang (cskaizhang@gmail.com)
github: https://github.com/cszn/BSRGAN
        https://github.com/cszn/KAIR
If you have any question, please feel free to contact with me.
Kai Zhang (e-mail: cskaizhang@gmail.com)
by Kai Zhang ( March/2020 --> March/2021 --> )
This work was previously submitted to CVPR2021.

# --------------------------------------------
@inproceedings{zhang2021designing,
  title={Designing a Practical Degradation Model for Deep Blind Image Super-Resolution},
  author={Zhang, Kai and Liang, Jingyun and Van Gool, Luc and Timofte, Radu},
  booktitle={arxiv},
  year={2021}
}
# --------------------------------------------

"""

def img2tensor(img, device, norm='0,1'):
    # img: uint8 RGB, HWC
    img = img.astype(np.float32) / 255.0
    if norm == '-1,1':
        img = img * 2.0 - 1.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img.to(device)


def main():
    # ----------------------------------------    
    # Parse arguments for kaggle
    # ----------------------------------------
    datasetname = 'Set5'
    parser = argparse.ArgumentParser(description='BSRGAN')
    parser.add_argument('--model_path', type=str, default='model_zoo')
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--input_lq', type=str, default=f'/root/media/quoc-huy/eval-wir/srbenchmarks/{datasetname}/LR-bicubic-matlab/')
    parser.add_argument('--input_hq', type=str, default=f'/root/media/quoc-huy/eval-wir/srbenchmarks/{datasetname}/HR/')
    parser.add_argument('--output_path', type=str, default=f'/root/quoc-huy/all-tested-results/BSRGAN/{datasetname}-bic-matlab')
    parser.add_argument('--log_path', type=str, default='bicubic.log')
    
    args = parser.parse_args()

    utils_logger.logger_info('bicubic', log_path=args.log_path)
    logger = logging.getLogger('bicubic')

#    print(torch.__version__)               # pytorch version
#    print(torch.version.cuda)              # cuda version
#    print(torch.backends.cudnn.version())  # cudnn version

    lq_path = args.input_lq
    hq_path = args.input_hq

    # model_names = ['RRDB','ESRGAN','FSSR_DPED','FSSR_JPEG','RealSR_DPED','RealSR_JPEG']
    model_names = [args.model_name] if args.model_name else ['BSRGAN']    # 'BSRGANx2' for scale factor 2

    save_results = True
    real_dataset = False
    # real_dataset = False
    sf = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for model_name in model_names:
        if model_name in ['BSRGANx2']:
            sf = 2
        model_path = os.path.join(args.model_path, model_name+'.pth')          # set model path
        logger.info('{:>16s} : {:s}'.format('Model Name', model_name))

        # torch.cuda.set_device(0)      # set GPU ID
        # logger.info('{:>16s} : {:<d}'.format('GPU ID', torch.cuda.current_device()))
        torch.cuda.empty_cache()

        # --------------------------------
        # define network and load model
        # --------------------------------
        model = net(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=sf)  # define network

#            model_old = torch.load(model_path)
#            state_dict = model.state_dict()
#            for ((key, param),(key2, param2)) in zip(model_old.items(), state_dict.items()):
#                state_dict[key2] = param
#            model.load_state_dict(state_dict, strict=True)

        model.load_state_dict(torch.load(model_path), strict=True)
        model.eval()
        for k, v in model.named_parameters():
            v.requires_grad = False
        model = model.to(device)
        torch.cuda.empty_cache()
        
        lpips_metric = pyiqa.create_metric('lpips', device=device)
        niqe_metric  = pyiqa.create_metric('niqe', device=device)
        fid_metric   = pyiqa.create_metric('fid', device=device)

        avg_lpips = 0.0
        avg_niqe  = 0.0

        for _ in range(1):

            L_path = os.path.join('', lq_path)  # low-quality images path
            GT_path = os.path.join('', hq_path)  # ground-truth path
            
            #E_path = os.path.join(testsets, testset_L+'_'+model_name)
            E_path = os.path.join('', args.output_path)
            util.mkdir(E_path)

            logger.info('{:>16s} : {:s}'.format('Input Path', L_path))
            logger.info('{:>16s} : {:s}'.format('Output Path', E_path))
            idx = 0
            
            avg_psnr = 0.0
            avg_ssim = 0.0
            
            if real_dataset:
                for img in util.get_image_paths(L_path):
                    img_name, ext = os.path.splitext(os.path.basename(img))
                    
                    img_L = util.imread_uint(img, n_channels=3)
                    img_L = util.uint2tensor4(img_L)
                    img_L = img_L.to(device)
                    
                    # --------------------------------
                    # (2) inference
                    # --------------------------------
                    img_E = model(img_L)

                    # --------------------------------
                    # (3) img_E
                    # --------------------------------
                    img_E = util.tensor2uint(img_E)
                    
                    if save_results:
                        util.imsave(img_E, os.path.join(E_path, img_name+'.png'))
                return

            for img, img_gt in zip(util.get_image_paths(L_path), util.get_image_paths(GT_path)):

                # --------------------------------
                # (1) img_L
                # --------------------------------
                idx += 1
                img_name, ext = os.path.splitext(os.path.basename(img))
                logger.info('{:->4d} --> {:<s} --> x{:<d}--> {:<s}'.format(idx, model_name, sf, img_name+ext))

                img_L = util.imread_uint(img, n_channels=3)
                img_L = util.uint2tensor4(img_L)
                img_L = img_L.to(device)
                
                # --------------------------------
                # (2) inference
                # --------------------------------
                img_E = model(img_L)

                # --------------------------------
                # (3) img_E
                # --------------------------------
                img_E = util.tensor2uint(img_E)
                    
                # --------------------------------
                # (additional) img_gt for PSNR/SSIM
                # --------------------------------
                
                gt_name, _ = os.path.splitext(os.path.basename(img_gt))
                
                img_gt = util.imread_uint(img_gt, n_channels=3)
                h1, w1 = img_gt.shape[:2]
                img_gt = img_gt.copy()[:h1 - h1 % sf, :w1 - w1 % sf, ...]
                
                print('img_E.shape, img_gt.shape', img_E.shape, img_gt.shape)
                print('image_gt_name: ', gt_name)
                
                if img_gt.shape[2] == 3:
                    img_gt_y = cv2.cvtColor(img_gt, cv2.COLOR_RGB2YCrCb)[:, :, 0]
                    img_E_y = cv2.cvtColor(img_E, cv2.COLOR_RGB2YCrCb)[:, :, 0]
                else:
                    img_gt_y = img_gt
                    img_E_y = img_E
                    
                psnr = util.calculate_psnr(img_E_y, img_gt_y)
                ssim = util.calculate_ssim(img_E_y, img_gt_y)
                
                avg_psnr += psnr
                avg_ssim += ssim
                
                logger.info('----PSNR/SSIM for ({:s}, x{:d})---- : {:.4f}dB / {:.4f}'.format(lq_path, sf, psnr, ssim))
                
                img_E_rgb  = img_E
                img_gt_rgb = img_gt

                sr_lpips = img2tensor(img_E_rgb, device, norm='0,1')
                gt_lpips = img2tensor(img_gt_rgb, device, norm='0,1')

                with torch.no_grad():
                    lpips_val = lpips_metric(sr_lpips, gt_lpips).item()

                avg_lpips += lpips_val
                
                sr_niqe = img2tensor(img_E_rgb, device, norm='0,1')

                with torch.no_grad():
                    niqe_val = niqe_metric(sr_niqe).item()

                avg_niqe += niqe_val

                
                if save_results:
                    util.imsave(img_E, os.path.join(E_path, img_name+'.png'))
                
            avg_psnr = avg_psnr / idx
            avg_ssim = avg_ssim / idx
            avg_lpips /= idx
            avg_niqe  /= idx
            
            logger.info('Computing FID...')
            with torch.no_grad():
                fid_score = fid_metric(E_path, GT_path).item()
            
            print('----Average PSNR/SSIM for: {:.2f}dB / {:.4f}'.format(avg_psnr, avg_ssim))
            # logger.info('----Average PSNR/SSIM for ({:s}, x{:d})---- : {:.4f}dB / {:.4f}'.format(lq_path, sf, avg_psnr, avg_ssim))
            logger.info(' ')
            print('----Average LPIPS/ FID / NIQE---- : {:.4f} / {:.4f} / {:.4f}'.format(avg_lpips, fid_score, avg_niqe))
            # logger.info('----Average LPIPS / NIQE---- : {:.4f} / {:.4f}'.format(avg_lpips, avg_niqe))

            # print('----FID---- : {:.4f}'.format(fid_score))
            # logger.info('----FID---- : {:.4f}'.format(fid_score))


if __name__ == '__main__':

    main()
