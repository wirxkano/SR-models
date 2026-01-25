import argparse
import cv2
import glob
import os
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from logger import Logger
from calc_psnr_ssim import calculate_psnr_ssim_y
# import pyiqa
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.color import rgb2ycbcr
import torch
import numpy as np

def measure_gflops(model):
    import torch
    import time
    import numpy as np
    from torch.profiler import profile, ProfilerActivity
    
    device = torch.device("cuda")
    channels = 3
    height = 256
    width = 256 
    # inputs = torch.randn(batch_size, channels, height, width).to(device)
    inputs = np.random.randint(0, 256, size=(width, height, channels), dtype=np.uint8)
    
    ### Warm-up
    with torch.no_grad():
        for _ in range(5):
            model.enhance(inputs, outscale=4)
            if device.type == 'cuda':
                torch.cuda.synchronize()
                
    with torch.no_grad():
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            with_flops=True
        ) as prof:
            if device.type == 'cuda':
                    torch.cuda.synchronize()
            start = time.perf_counter_ns()
            model.enhance(inputs, outscale=4)
            
            if device.type == 'cuda':
                    torch.cuda.synchronize()
            end = time.perf_counter_ns()
            
    total_flops = sum([event.flops for event in prof.key_averages()])
    print(f"\n#### Total FLOPs: {total_flops:,}")
    print(f"#### Total FLOPs: {total_flops / 1e9:.2f} GFLOPs")
    print(f"#### Execution Time: {(end - start) / 1e6:.2f} ms")
    print(f"#### Throughput: {total_flops / (end - start):.2f} GFLOPS")
    

def main():
    """Inference demo for Real-ESRGAN.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='inputs', help='Input image or folder')
    parser.add_argument(
        '-n',
        '--model_name',
        type=str,
        default='RealESRGAN_x4plus',
        help=('Model names: RealESRGAN_x4plus | RealESRNet_x4plus | RealESRGAN_x4plus_anime_6B | RealESRGAN_x2plus | '
              'realesr-animevideov3 | realesr-general-x4v3'))
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder')
    parser.add_argument(
        '-dn',
        '--denoise_strength',
        type=float,
        default=0.5,
        help=('Denoise strength. 0 for weak denoise (keep noise), 1 for strong denoise ability. '
              'Only used for the realesr-general-x4v3 model'))
    parser.add_argument('-s', '--outscale', type=float, default=4, help='The final upsampling scale of the image')
    parser.add_argument(
        '--model_path', type=str, default=None, help='[Option] Model path. Usually, you do not need to specify it')
    parser.add_argument('--suffix', type=str, default='', help='Suffix of the restored image')
    parser.add_argument('-t', '--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
    parser.add_argument('--face_enhance', action='store_true', help='Use GFPGAN to enhance face')
    parser.add_argument(
        '--fp32', action='store_true', help='Use fp32 precision during inference. Default: fp16 (half precision).')
    parser.add_argument(
        '--alpha_upsampler',
        type=str,
        default='realesrgan',
        help='The upsampler for the alpha channels. Options: realesrgan | bicubic')
    parser.add_argument(
        '--ext',
        type=str,
        default='auto',
        help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs')
    parser.add_argument(
        '-g', '--gpu-id', type=int, default=None, help='gpu device to use (default=None) can be 0,1,2 for multi-gpu')
    parser.add_argument(
        '--ground_truth', type=str, default=None, help='Ground truth path')
    parser.add_argument(
        '--real_dataset', type=bool, default=False, help='Flag to calculate PSNR and SSIM')

    args = parser.parse_args()

    # determine models according to model names
    args.model_name = args.model_name.split('.')[0]
    if args.model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
    elif args.model_name == 'RealESRNet_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
    elif args.model_name == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet model with 6 blocks
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
    elif args.model_name == 'RealESRGAN_x2plus':  # x2 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
    elif args.model_name == 'realesr-animevideov3':  # x4 VGG-style model (XS size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
    elif args.model_name == 'realesr-general-x4v3':  # x4 VGG-style model (S size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        netscale = 4
        file_url = [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
        ]

    # determine model paths
    if args.model_path is not None:
        model_path = args.model_path
    else:
        model_path = os.path.join('weights', args.model_name + '.pth')
        if not os.path.isfile(model_path):
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            for url in file_url:
                # model_path will be updated
                model_path = load_file_from_url(
                    url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

    # use dni to control the denoise strength
    dni_weight = None
    if args.model_name == 'realesr-general-x4v3' and args.denoise_strength != 1:
        wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
        model_path = [model_path, wdn_model_path]
        dni_weight = [args.denoise_strength, 1 - args.denoise_strength]

    # restorer
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=not args.fp32,
        gpu_id=args.gpu_id)
    
    measure_gflops(upsampler)
    return

    if args.face_enhance:  # Use GFPGAN for face enhancement
        from gfpgan import GFPGANer
        face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=args.outscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler)
    os.makedirs(args.output, exist_ok=True)

    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*')))
    
    if not args.real_dataset and args.ground_truth:
        gt_paths = sorted(glob.glob(os.path.join(args.ground_truth, '*')))
    else:
        gt_paths = paths

    avg_psnr = 0
    avg_ssim = 0
    avg_lpips = 0
    avg_niqe = 0
    cnt = 0
    logger = Logger(name="test")
    app_logger = logger.get_logger()
    app_logger.info(f"Testing {args.input}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_metric = pyiqa.create_metric('lpips', device=device)
    niqe_metric = pyiqa.create_metric('niqe', device=device)
    
    for idx, (path, gt_path) in enumerate(zip(paths, gt_paths)):
        cnt += 1
        imgname, extension = os.path.splitext(os.path.basename(path))
        # print('Testing', idx, imgname)

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        gt_img = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
        if type(img) is None:
            print("Error!!!")
            continue
        
        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = 'RGBA'
        else:
            img_mode = None

        try:
            if args.face_enhance:
                _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
            else:
                output, _ = upsampler.enhance(img, outscale=args.outscale)
        except RuntimeError as error:
            print('Error', error)
            print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
        else:
            if args.ext == 'auto':
                extension = extension[1:]
            else:
                extension = args.ext
            if img_mode == 'RGBA':  # RGBA images should be saved in png format
                extension = 'png'
            if args.suffix == '':
                save_path = os.path.join(args.output, f'{imgname}.{extension}')
            else:
                save_path = os.path.join(args.output, f'{imgname}_{args.suffix}.{extension}')
            
            if not args.real_dataset:
                h = min(output.shape[0], gt_img.shape[0])
                w = min(output.shape[1], gt_img.shape[1])

                if output.ndim == 3:
                    output = output[:h, :w, :]
                    gt_img = gt_img[:h, :w, :]
                else:
                    output = output[:h, :w]
                    gt_img = gt_img[:h, :w]
                
                output_clipped = np.clip(output, 0, 255)
                gt_img_clipped = np.clip(gt_img, 0, 255)
                
                if len(output.shape) == 3 and output.shape[2] == 3:  # RGB image
                    output_rgb = cv2.cvtColor(output_clipped.astype(np.uint8), cv2.COLOR_BGR2RGB) / 255.0
                    gt_img_rgb = cv2.cvtColor(gt_img_clipped.astype(np.uint8), cv2.COLOR_BGR2RGB) / 255.0
                    
                    # print(output_rgb[0, 0])
                    # print(gt_img_rgb[0, 0])
                    
                    output_y = rgb2ycbcr(output_rgb)[:, :, 0]
                    gt_img_y = rgb2ycbcr(gt_img_rgb)[:, :, 0]
                    
                    # print(output_y.shape, gt_img_y.shape)
                    
                    psnr = peak_signal_noise_ratio(gt_img_y, output_y, data_range=255.0)
                    ssim = structural_similarity(gt_img_y, output_y, data_range=255.0)
                    
                    avg_psnr += psnr
                    avg_ssim += ssim
                    output_tensor = torch.from_numpy(output_rgb.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
                    gt_tensor = torch.from_numpy(gt_img_rgb.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
                    output_tensor = torch.clamp(output_tensor, 0, 1)
                    gt_tensor = torch.clamp(gt_tensor, 0, 1)
                    
                    lpips_val = lpips_metric(output_tensor, gt_tensor)
                    avg_lpips += lpips_val.item()
                    niqe_val = niqe_metric(output_tensor)
                    avg_niqe += niqe_val.item()
                    
            cv2.imwrite(save_path, output)
    
    if not args.real_dataset:
        avg_psnr /= cnt
        avg_ssim /= cnt
        avg_lpips /= cnt
        avg_niqe /= cnt
        fid_metric = pyiqa.create_metric('fid', device=device)
        fid_val = fid_metric(args.output, args.ground_truth).item()
        app_logger.info(f'Average PSNR_Y: {avg_psnr:.2f}, SSIM_Y: {avg_ssim:.4f}, LPIPS: {avg_lpips:.4f}, FID: {fid_val:.4f}, NIQE: {avg_niqe:.4f}')
        app_logger.info('============================================')


if __name__ == '__main__':
    main()
