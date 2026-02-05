import argparse
import yaml
import pdb
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import datasets
import models
import utils
from models.controller import *
from datasets.blur import SRMDPreprocessing

import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision import transforms
import os
import pyiqa
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.color import rgb2ycbcr
from torchvision.utils import make_grid
import math


def batched_predict(model, inp, coord, cell, bsize):
	n = coord.shape[1]
	ql = 0
	preds = []
		
	w = utils.wave(inp, model.wav)
	feature = model.encoder(w) # fix
	model.SR.gen_feat(inp, feature)

	while ql < n:
		qr = min(ql + bsize, n)
		pred = model.SR.query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :])
		preds.append(pred)
		ql = qr
	pred = torch.cat(preds, dim=1)
	return pred


# def evaluate(loader, model, device, save_dir, scale=None, data_norm=None, eval_type=None, eval_bsize=None, verbose=False):
# 	model.eval()
	
# 	if data_norm is None:
# 		data_norm = {
# 			'inp': {'sub': [0], 'div': [1]},
# 			'gt': {'sub': [0], 'div': [1]}
# 		}
# 	t = data_norm['inp']
# 	inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).to(device)
# 	inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).to(device)
# 	t = data_norm['gt']
# 	gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).to(device)
# 	gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).to(device)

# 	# val_res_s, val_res_p, val_res_l = utils.Averager(), utils.Averager(), utils.Averager()
# 	# lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
 
# 	val_res_psnr = utils.Averager()
# 	val_res_ssim = utils.Averager()
# 	val_res_lpips = utils.Averager()
# 	val_res_fid = utils.Averager()
# 	val_res_niqe = utils.Averager()
	
# 	lpips_metric = pyiqa.create_metric('lpips', device=device)
# 	niqe_metric = pyiqa.create_metric('niqe', device=device)
 
# 	all_preds = []
# 	all_gts = []
				
# 	pbar = tqdm(loader, leave=False, desc='test')
# 	i = 0
# 	for batch in pbar:
# 		for k, v in batch.items(): 
# 			batch[k] = v.to(device) 

# 		if eval_bsize is None:
# 			with torch.no_grad():
# 				shape = batch['coord'].shape
# 				pred = model(batch['inp'], batch['coord'], batch['cell'])
					
# 		else:
# 			with torch.no_grad():
# 				inp = (batch['inp'] - inp_sub) / inp_div
# 				pred = batched_predict(model, inp,
# 					batch['coord'], batch['cell'], eval_bsize)
# 				pred = pred * gt_div + gt_sub
# 				pred.clamp_(0, 1)

# 		if eval_type is not None: # reshape for shaving-eval
# 			try:
# 				shape = [batch['inp'].shape[0], batch['shape'][0,1].item() , batch['shape'][0,2].item(), 3]
# 				pred = pred.view(*shape) \
# 					.permute(0, 3, 1, 2).contiguous()
# 				batch['gt'] = batch['gt'].view(*shape) \
# 					.permute(0, 3, 1, 2).contiguous()
# 			except:
# 				pdb.set_trace()

# 		with torch.no_grad():
# 			l = lpips(pred, batch['gt'])
# 			p = utils.calc_psnr(pred, batch['gt'], dataset='benchmark')
# 			s = utils.calc_ssim(pred, batch['gt'])
   
# 		saved_img = pred.squeeze(0)
# 		saved_img = transforms.ToPILImage()(saved_img)
# 		saved_img.save(os.path.join(save_dir, str(i) + '.png'))
# 		i += 1

# 		val_res_l.add(l.item(), batch['gt'].shape[0])
# 		val_res_p.add(p.item(), batch['gt'].shape[0])
# 		val_res_s.add(s.item(), batch['gt'].shape[0])
# 		if verbose:
# 			pbar.set_description('psnr: {:.4f}, ssim: {:.4f}, lpips: {:.4f}'\
# 				.format(val_res_p.item(), val_res_s.item(),  val_res_l.item()))

# 	return val_res_p.item(), val_res_s.item(), val_res_l.item()

def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default), BGR channel order
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)

def evaluate(loader, model, device, hr_dir, save_dir, scale=None, data_norm=None, eval_type=None, eval_bsize=None, verbose=False):
	model.eval()
 
	if data_norm is None:
		data_norm = {
			'inp': {'sub': [0], 'div': [1]},
			'gt': {'sub': [0], 'div': [1]}
		}
	t = data_norm['inp']
	inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).to(device)
	inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).to(device)
	t = data_norm['gt']
	gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).to(device)
	gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).to(device)

	val_res_psnr = utils.Averager()
	val_res_ssim = utils.Averager()
	val_res_lpips = utils.Averager()
	val_res_fid = utils.Averager()
	val_res_niqe = utils.Averager()
	
	lpips_metric = pyiqa.create_metric('lpips', device='cpu')
	niqe_metric = pyiqa.create_metric('niqe', device='cpu')
	
	pbar = tqdm(loader, leave=False, desc='test')
	i = 1
	for batch in pbar:
		pred_kernel = False
		for k, v in batch.items(): 
			batch[k] = v.to(device) 
		if eval_bsize is None:
			with torch.no_grad():
				shape = batch['coord'].shape
				pred, kernel = model(batch['inp'], batch['coord'], batch['cell'])
				pred_kernel = True
		else:
			with torch.no_grad():
				inp = (batch['inp'] - inp_sub) / inp_div
				pred = batched_predict(model, inp,
					batch['coord'], batch['cell'], eval_bsize)
				pred = pred * gt_div + gt_sub
				pred.clamp_(0, 1)

		if eval_type is not None: # reshape for shaving-eval
			try:
				shape = [batch['inp'].shape[0], batch['shape'][0,1].item(), batch['shape'][0,2].item(), 3]
				pred = pred.view(*shape).permute(0, 3, 1, 2).contiguous()
				batch['gt'] = batch['gt'].view(*shape).permute(0, 3, 1, 2).contiguous()
			except:
				pdb.set_trace()
    
		# if pred_kernel:
		# 	blur_kernel = tensor2img(kernel, np.float32)
		# 	k_vis = np.clip(blur_kernel, 0, 1)
		# 	k_vis = k_vis / k_vis.max()
			
		# 	fig, ax = plt.subplots(figsize=(4, 4))
		# 	ax.imshow(k_vis, cmap='gray', interpolation='nearest')
		# 	ax.axis('off')
   
		# 	save_ker_path = os.path.join(save_dir, 'kernels', 'img_' + str(i).zfill(3) + '.png')

		# 	fig.savefig(
		# 		save_ker_path,
		# 		dpi=350,
		# 		bbox_inches='tight',
		# 		pad_inches=0
		# 	)
		# 	plt.close(fig)

		with torch.no_grad():
			pred_np = pred.cpu().numpy()
			gt_np = batch['gt'].cpu().numpy()
			
			batch_psnr = 0
			batch_ssim = 0
			batch_size = pred_np.shape[0]
			
			for b in range(batch_size):
				pred_img = pred_np[b].transpose(1, 2, 0)
				gt_img = gt_np[b].transpose(1, 2, 0)
				
				pred_y = rgb2ycbcr(pred_img)[:, :, 0]
				gt_y = rgb2ycbcr(gt_img)[:, :, 0]
    
				psnr_val = peak_signal_noise_ratio(gt_y, pred_y, data_range=255.0)
				ssim_val = structural_similarity(gt_y, pred_y, data_range=255.0)
				
				batch_psnr += psnr_val
				batch_ssim += ssim_val
			
			batch_psnr /= batch_size
			batch_ssim /= batch_size
			
			# Calculate LPIPS and NIQE using pyiqa (in RGB)
			lpips_val = lpips_metric(pred, batch['gt']).mean()
			niqe_val = niqe_metric(pred).mean()
			
		saved_img = pred.squeeze(0)
		saved_img = transforms.ToPILImage()(saved_img)
		saved_img.save(os.path.join(save_dir, 'imgs', 'img_' + str(i).zfill(3) + '.png'))
		i += 1

		val_res_psnr.add(batch_psnr, batch_size)
		val_res_ssim.add(batch_ssim, batch_size)
		val_res_lpips.add(lpips_val.item(), batch_size)
		val_res_niqe.add(niqe_val.item(), batch_size)
		
		if verbose:
			pbar.set_description('psnr: {:.4f}, ssim: {:.4f}, lpips: {:.4f}, niqe: {:.4f}'\
				.format(val_res_psnr.item(), val_res_ssim.item(), val_res_lpips.item(), val_res_niqe.item()))
   
		del pred_np, gt_np, pred_img, gt_img, pred_y, gt_y, saved_img
		del batch, pred, lpips_val, niqe_val
  
		torch.cuda.empty_cache()
		print(f"Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
		print(f"Reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")   
  	
	# Calculate FID on all collected images (in RGB)
	with torch.no_grad():
		fid_metric = pyiqa.create_metric('fid', device='cpu')
		fid_val = fid_metric(save_dir + "/imgs", hr_dir)
		val_res_fid.add(fid_val.item(), 1)

	return {
		'psnr': val_res_psnr.item(),
		'ssim': val_res_ssim.item(),
		'lpips': val_res_lpips.item(),
		'fid': val_res_fid.item(),
		'niqe': val_res_niqe.item()
	}

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_config', help='model config file path', default='configs/train-div2k/train_liif.yaml')
	parser.add_argument('--model_weight', default='your path to model_weight.pth')
	parser.add_argument('--test_config', help='test config file path', default='configs/test/test-set5-2.yaml')
	args = parser.parse_args()

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	with open(args.test_config, 'r') as f:
			testconfig = yaml.load(f, Loader=yaml.FullLoader)
	loader = datasets.make_data_loaders(testconfig, DDP=False, state='test')
		
	with open(args.model_config, 'r') as f:
		modelconfig = yaml.load(f, Loader=yaml.FullLoader)
  
	# print("="*50, testconfig)
		
	model = models.make(modelconfig['model'], args={'config': modelconfig}, load_sd=args.model_weight).to(device)
	save_dir = testconfig["test_dataset"]["dataset"]["args"]["save_dir"]
	if not os.path.exists(save_dir):
		os.makedirs(save_dir, exist_ok=True)
  
	hr_dir = testconfig["test_dataset"]["dataset"]["args"]["root_path_2"]
  
	res = evaluate(loader, model, device,
		hr_dir=hr_dir,
		save_dir=save_dir,
		data_norm=testconfig.get('data_norm'),
		eval_type=testconfig.get('eval_type'),
		eval_bsize=testconfig.get('eval_bsize'),
		verbose=True)
	psnr, ssim, lpips, fid, niqe = res['psnr'], res['ssim'], res['lpips'], res['fid'], res['niqe']
	print('psnr: {:.2f}, ssim: {:.4f}, lpips: {:.4f}'.format(psnr, ssim, lpips))
	print('fid: {:.4f}, niqe: {:.4f}'.format(fid, niqe))

