import os
import utility
import torch
from decimal import Decimal
import torch.nn.functional as F
from utils import util
from pathlib import Path
import cv2
import numpy as np
from torchvision import transforms
import pyiqa
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.color import rgb2ycbcr

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.model_E = torch.nn.DataParallel(self.model.get_model().E, range(self.args.n_GPUs))
        self.loss = my_loss
        self.contrast_loss = torch.nn.CrossEntropyLoss().cuda()
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)

        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()

    def train(self):
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1

        # lr stepwise
        if epoch <= self.args.epochs_encoder:
            lr = self.args.lr_encoder * (self.args.gamma_encoder ** (epoch // self.args.lr_decay_encoder))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            lr = self.args.lr_sr * (self.args.gamma_sr ** ((epoch - self.args.epochs_encoder) // self.args.lr_decay_sr))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        self.ckp.write_log('[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr)))
        self.loss.start_log()
        self.model.train()

        degrade = util.SRMDPreprocessing(
            self.scale[0],
            kernel_size=self.args.blur_kernel,
            blur_type=self.args.blur_type,
            sig_min=self.args.sig_min,
            sig_max=self.args.sig_max,
            lambda_min=self.args.lambda_min,
            lambda_max=self.args.lambda_max,
            noise=self.args.noise
        )

        timer = utility.timer()
        losses_contrast, losses_sr = utility.AverageMeter(), utility.AverageMeter()

        for batch, (hr, _, idx_scale) in enumerate(self.loader_train):
            hr = hr.cuda()                              # b, n, c, h, w
            lr, b_kernels = degrade(hr)                 # bn, c, h, w

            self.optimizer.zero_grad()

            timer.tic()
            # forward
            ## train degradation encoder
            if epoch <= self.args.epochs_encoder:
                _, output, target = self.model_E(im_q=lr[:,0,...], im_k=lr[:,1,...])
                loss_constrast = self.contrast_loss(output, target)
                loss = loss_constrast

                losses_contrast.update(loss_constrast.item())
            ## train the whole network
            else:
                sr, output, target = self.model(lr)
                loss_SR = self.loss(sr, hr[:,0,...])
                loss_constrast = self.contrast_loss(output, target)
                loss = loss_constrast + loss_SR

                losses_sr.update(loss_SR.item())
                losses_contrast.update(loss_constrast.item())

            # backward
            loss.backward()
            self.optimizer.step()
            timer.hold()

            if epoch <= self.args.epochs_encoder:
                if (batch + 1) % self.args.print_every == 0:
                    self.ckp.write_log(
                        'Epoch: [{:03d}][{:04d}/{:04d}]\t'
                        'Loss [contrastive loss: {:.3f}]\t'
                        'Time [{:.1f}s]'.format(
                            epoch, (batch + 1) * self.args.batch_size, len(self.loader_train.dataset),
                            losses_contrast.avg,
                            timer.release()
                        ))
            else:
                if (batch + 1) % self.args.print_every == 0:
                    self.ckp.write_log(
                        'Epoch: [{:04d}][{:04d}/{:04d}]\t'
                        'Loss [SR loss:{:.3f} | contrastive loss: {:.3f}]\t'
                        'Time [{:.1f}s]'.format(
                            epoch, (batch + 1) * self.args.batch_size, len(self.loader_train.dataset),
                            losses_sr.avg, losses_contrast.avg,
                            timer.release(),
                        ))

        self.loss.end_log(len(self.loader_train))

        # save model
        target = self.model.get_model()
        model_dict = target.state_dict()
        keys = list(model_dict.keys())
        for key in keys:
            if 'E.encoder_k' in key or 'queue' in key:
                del model_dict[key]
        torch.save(
            model_dict,
            os.path.join(self.ckp.dir, 'model', 'model_{}.pt'.format(epoch))
        )

    # def test(self):
    #     self.ckp.write_log('\nEvaluation:')
    #     self.ckp.add_log(torch.zeros(1, len(self.scale)))
    #     self.model.eval()

    #     timer_test = utility.timer()
        
    #     if self.args.custom_testset:
    #         scale = 4
    #         eval_psnr = 0
    #         eval_ssim = 0
    #         folder_len = 0
    #         with torch.no_grad():
    #             lr_path = Path(self.args.lr_folder)
    #             if self.args.calc_psnr:
    #                 hr_path = Path(self.args.hr_folder)
    #             else:
    #                 hr_path = Path(self.args.lr_folder)
                
    #             print("=======",lr_path, hr_path)
    
    #             image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    #             lr_images = sorted([f for f in lr_path.iterdir() if f.suffix.lower() in image_extensions])
    #             hr_images = sorted([f for f in hr_path.iterdir() if f.suffix.lower() in image_extensions])
    #             folder_len = len(lr_images)
    #             if len(lr_images) != len(hr_images):
    #                 print(f"Warning: Number of LR images ({len(lr_images)}) != Number of HR images ({len(hr_images)})")
                
    #             device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
                
    #             for lr_file, hr_file in zip(lr_images, hr_images):
    #                 lr_img = cv2.imread(str(lr_file))
    #                 hr_img = cv2.imread(str(hr_file))
                    
    #                 filename = hr_file.name
                    
    #                 lr = torch.from_numpy(lr_img.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
                    
    #                 # inference
    #                 timer_test.tic()
    #                 sr = self.model(lr)
    #                 timer_test.hold()
                    
    #                 _, _, sr_h, sr_w = sr.shape
    #                 hr_h, hr_w = hr_img.shape[:2]
                    
    #                 if sr_h != hr_h or sr_w != hr_w:
    #                     h_start = (hr_h - sr_h) // 2
    #                     w_start = (hr_w - sr_w) // 2
    #                     hr_img = hr_img[h_start:h_start+sr_h, w_start:w_start+sr_w]
                        
    #                 hr = torch.from_numpy(hr_img.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    #                 sr = utility.quantize(sr, self.args.rgb_range)
    #                 hr = utility.quantize(hr, self.args.rgb_range)
                    
    #                 sr_fixed = sr[:, [2, 1, 0], :, :]                    
    #                 saved_img = transforms.ToPILImage()(sr_fixed.squeeze(0).clamp(0, 255).byte())
    #                 save_dir = "/root/quoc-huy/all-tested-results/DASR_iso/Set5"
    #                 if not os.path.exists(save_dir):
    #                     os.makedirs(save_dir, exist_ok=True)
                        
    #                 # saved_img.save(os.path.join(save_dir, filename))


    #                 # metrics
    #                 if self.args.calc_psnr:
    #                     eval_psnr += utility.calc_psnr(
    #                         sr, hr, scale, self.args.rgb_range,
    #                         benchmark=True
    #                     )
    #                     eval_ssim += utility.calc_ssim(
    #                         sr, hr, scale,
    #                         benchmark=True
    #                     )

    #                 # save results
    #                 if self.args.save_results:
    #                     save_list = [sr]
    #                     filename = filename[0]
    #                     self.ckp.save_results(filename, save_list, scale)
                
    #             self.ckp.log[-1, 0] = eval_psnr / folder_len
    #             self.ckp.write_log(
    #                 '[Epoch {}---{} x{}]\tPSNR: {:.2f} SSIM: {:.4f}'.format(
    #                     self.args.resume,
    #                     self.args.data_test,
    #                     scale,
    #                     eval_psnr / folder_len,
    #                     eval_ssim / folder_len,
    #                 ))
    #         return
    #     with torch.no_grad():
    #         for idx_scale, scale in enumerate(self.scale):
    #             self.loader_test.dataset.set_scale(idx_scale)
    #             eval_psnr = 0
    #             eval_ssim = 0

    #             degrade = util.SRMDPreprocessing(
    #                 self.scale[0],
    #                 kernel_size=self.args.blur_kernel,
    #                 blur_type=self.args.blur_type,
    #                 sig=self.args.sig,
    #                 lambda_1=self.args.lambda_1,
    #                 lambda_2=self.args.lambda_2,
    #                 theta=self.args.theta,
    #                 noise=self.args.noise
    #             )

    #             for idx_img, (hr, filename, what) in enumerate(self.loader_test):
    #                 hr = hr.cuda()                      # b, 1, c, h, w
    #                 hr = self.crop_border(hr, scale)
    #                 lr, _ = degrade(hr, random=False)   # b, 1, c, h, w
    #                 hr = hr[:, 0, ...]                  # b, c, h, w

    #                 # inference
    #                 timer_test.tic()
    #                 sr = self.model(lr[:, 0, ...])
    #                 timer_test.hold()

    #                 sr = utility.quantize(sr, self.args.rgb_range)
    #                 hr = utility.quantize(hr, self.args.rgb_range)

    #                 # metrics
    #                 eval_psnr += utility.calc_psnr(
    #                     sr, hr, scale, self.args.rgb_range,
    #                     benchmark=self.loader_test.dataset.benchmark
    #                 )
    #                 eval_ssim += utility.calc_ssim(
    #                     sr, hr, scale,
    #                     benchmark=self.loader_test.dataset.benchmark
    #                 )

    #                 # save results
    #                 if self.args.save_results:
    #                     save_list = [sr]
    #                     filename = filename[0]
    #                     self.ckp.save_results(filename, save_list, scale)

    #             self.ckp.log[-1, idx_scale] = eval_psnr / len(self.loader_test)
    #             self.ckp.write_log(
    #                 '[Epoch {}---{} x{}]\tPSNR: {:.3f} SSIM: {:.4f}'.format(
    #                     self.args.resume,
    #                     self.args.data_test,
    #                     scale,
    #                     eval_psnr / len(self.loader_test),
    #                     eval_ssim / len(self.loader_test),
    #                 ))


    def test(self):
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model.eval()

        timer_test = utility.timer()
        
        device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        lpips_metric = pyiqa.create_metric('lpips', device=device)
        niqe_metric = pyiqa.create_metric('niqe', device=device)
        
        if self.args.custom_testset:
            scale = 4
            eval_psnr = 0
            eval_ssim = 0
            eval_lpips = 0
            eval_niqe = 0
            folder_len = 0
            
            with torch.no_grad():
                lr_path = Path(self.args.lr_folder)
                if self.args.calc_psnr:
                    hr_path = Path(self.args.hr_folder)
                else:
                    hr_path = Path(self.args.lr_folder)
                
                image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
                lr_images = sorted([f for f in lr_path.iterdir() if f.suffix.lower() in image_extensions])
                hr_images = sorted([f for f in hr_path.iterdir() if f.suffix.lower() in image_extensions])
                folder_len = len(lr_images)
                if len(lr_images) != len(hr_images):
                    print(f"Warning: Number of LR images ({len(lr_images)}) != Number of HR images ({len(hr_images)})")
                
                for lr_file, hr_file in zip(lr_images, hr_images):
                    lr_img = cv2.imread(str(lr_file))
                    hr_img = cv2.imread(str(hr_file))
                    
                    filename = hr_file.name
                    
                    lr = torch.from_numpy(lr_img.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
                    
                    # inference
                    timer_test.tic()
                    sr, degradation = self.model(lr)
                    timer_test.hold()
                    
                    _, _, sr_h, sr_w = sr.shape
                    hr_h, hr_w = hr_img.shape[:2]
                    
                    if sr_h != hr_h or sr_w != hr_w:
                        h_start = (hr_h - sr_h) // 2
                        w_start = (hr_w - sr_w) // 2
                        hr_img = hr_img[h_start:h_start+sr_h, w_start:w_start+sr_w]
                        
                    hr = torch.from_numpy(hr_img.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

                    sr = utility.quantize(sr, self.args.rgb_range)
                    hr = utility.quantize(hr, self.args.rgb_range)
                    
                    # Normalize to [0, 1] for metrics calculation
                    sr_norm = sr / self.args.rgb_range
                    hr_norm = hr / self.args.rgb_range
                    
                    sr_fixed = sr[:, [2, 1, 0], :, :]                    
                    saved_img = transforms.ToPILImage()(sr_fixed.squeeze(0).clamp(0, 255).byte())
                    save_dir = self.args.save_folder
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)
                        
                    saved_img.save(os.path.join(save_dir, filename))

                    # metrics
                    if self.args.calc_psnr:
                        sr_np = sr_norm.cpu().numpy()[0].transpose(1, 2, 0)  # (H, W, C)
                        hr_np = hr_norm.cpu().numpy()[0].transpose(1, 2, 0)  # (H, W, C)
                        
                        sr_y = rgb2ycbcr(sr_np)[:, :, 0]
                        hr_y = rgb2ycbcr(hr_np)[:, :, 0]
                        
                        psnr_val = peak_signal_noise_ratio(hr_y, sr_y, data_range=255.0)
                        ssim_val = structural_similarity(hr_y, sr_y, data_range=255.0)
                        
                        eval_psnr += psnr_val
                        eval_ssim += ssim_val
                        
                        lpips_val = lpips_metric(sr_norm, hr_norm)
                        niqe_val = niqe_metric(sr_norm)
                        
                        eval_lpips += lpips_val.item()
                        eval_niqe += niqe_val.item()
                        
                    # save results
                    # if self.args.save_results:
                    #     save_list = [sr]
                    #     filename = filename[0]
                    #     self.ckp.save_results(filename, save_list, scale)
                
                # Calculate FID on all collected images
                eval_fid = 0
                if self.args.calc_psnr:
                    fid_metric = pyiqa.create_metric('fid', device=device)
                    fid_val = fid_metric(save_dir, self.args.hr_folder)
                    eval_fid = fid_val.item()
                
                self.ckp.log[-1, 0] = eval_psnr / folder_len
                self.ckp.write_log(
                    '[Epoch {}---{} x{}]\tPSNR: {:.2f} SSIM: {:.4f} LPIPS: {:.4f} FID: {:.4f} NIQE: {:.4f}'.format(
                        self.args.resume,
                        self.args.data_test,
                        scale,
                        eval_psnr / folder_len,
                        eval_ssim / folder_len,
                        eval_lpips / folder_len,
                        eval_fid,
                        eval_niqe / folder_len,
                    ))
            return
            
        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                self.loader_test.dataset.set_scale(idx_scale)
                eval_psnr = 0
                eval_ssim = 0
                eval_lpips = 0
                eval_niqe = 0
                
                degrade = util.SRMDPreprocessing(
                    self.scale[0],
                    kernel_size=self.args.blur_kernel,
                    blur_type=self.args.blur_type,
                    sig=self.args.sig,
                    lambda_1=self.args.lambda_1,
                    lambda_2=self.args.lambda_2,
                    theta=self.args.theta,
                    noise=self.args.noise
                )

                for idx_img, (hr, filename, what) in enumerate(self.loader_test):
                    hr = hr.cuda()                      # b, 1, c, h, w
                    hr = self.crop_border(hr, scale)
                    lr, _ = degrade(hr, random=False)   # b, 1, c, h, w
                    hr = hr[:, 0, ...]                  # b, c, h, w

                    # inference
                    timer_test.tic()
                    sr, degradation = self.model(lr[:, 0, ...])
                    timer_test.hold()

                    sr = utility.quantize(sr, self.args.rgb_range)
                    hr = utility.quantize(hr, self.args.rgb_range)
                    
                    # Normalize to [0, 1] for metrics calculation
                    sr_norm = sr / self.args.rgb_range
                    hr_norm = hr / self.args.rgb_range

                    # Convert to numpy for skimage (calculate in Y channel)
                    sr_np = sr_norm.cpu().numpy()[0].transpose(1, 2, 0)  # (H, W, C)
                    hr_np = hr_norm.cpu().numpy()[0].transpose(1, 2, 0)  # (H, W, C)
                    
                    # Convert RGB to YCbCr and extract Y channel
                    sr_y = rgb2ycbcr(sr_np)[:, :, 0]
                    hr_y = rgb2ycbcr(hr_np)[:, :, 0]
                    
                    # Calculate PSNR and SSIM on Y channel
                    psnr_val = peak_signal_noise_ratio(hr_y, sr_y, data_range=255.0)
                    ssim_val = structural_similarity(hr_y, sr_y, data_range=255.0)
                    
                    eval_psnr += psnr_val
                    eval_ssim += ssim_val
                    
                    # Calculate LPIPS and NIQE using pyiqa (in RGB)
                    lpips_val = lpips_metric(sr_norm, hr_norm)
                    niqe_val = niqe_metric(sr_norm)
                    
                    eval_lpips += lpips_val.item()
                    eval_niqe += niqe_val.item()
                    
                    # save results
                    if self.args.save_results:
                        save_list = [sr]
                        filename = filename[0]
                        self.ckp.save_results(filename, save_list, scale)
                
                # Calculate FID on all collected images
                eval_fid = 0
                if len(all_preds) > 0:
                    fid_metric = pyiqa.create_metric('fid', device=device)
                    fid_val = fid_metric(self.args.save_folder, self.args.hr_folder)
                    eval_fid = fid_val.item()

                self.ckp.log[-1, idx_scale] = eval_psnr / len(self.loader_test)
                self.ckp.write_log(
                    '[Epoch {}---{} x{}]\tPSNR: {:.3f} SSIM: {:.4f} LPIPS: {:.4f} FID: {:.2f} NIQE: {:.4f}'.format(
                        self.args.resume,
                        self.args.data_test,
                        scale,
                        eval_psnr / len(self.loader_test),
                        eval_ssim / len(self.loader_test),
                        eval_lpips / len(self.loader_test),
                        eval_fid,
                        eval_niqe / len(self.loader_test),
                    )
                )
    
    def crop_border(self, img_hr, scale):
        b, n, c, h, w = img_hr.size()

        img_hr = img_hr[:, :, :, :int(h//scale*scale), :int(w//scale*scale)]

        return img_hr

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs_encoder + self.args.epochs_sr

