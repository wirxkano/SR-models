import time
from pytorch_wavelets import DWTForward

import torch
import torch.nn as nn

import models
from models import register
from utils import wave, make_coord
from datasets import blur

def cuda_step_time(tag, fn):
    if not torch.cuda.is_available():
        start = time.perf_counter()
        out = fn()
        end = time.perf_counter()
        print(f"{tag}: {(end-start)*1000:.2f} ms (CPU)")
        return out

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    starter.record()

    out = fn()

    ender.record()
    torch.cuda.synchronize()

    print(f"{tag}: {starter.elapsed_time(ender):.2f} ms")
    return out


@register('models')
class Model(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		# hyperparameter
		self.bs = config['batch_size']
		self.qpt = config['sample_q']
		self.N = config['num_pt'] 
		self.inp_size = config['inp_size']
		
		# model
		spec = config['wavelet']
		self.wav = DWTForward(**spec).to(self.device)
		self.blur = blur.BatchBlur(**config['blur'])
		spec = config['model']
		self.encoder = models.make(spec['degrade'], load_sd=spec['path'], freeze=True, key='degrade').to(self.device)
		self.SR = models.make(spec['SR']).to(self.device)
		self.kernel = models.make(spec['kernel'], load_sd=spec['path'], freeze=True, key='kernel').to(self.device) 

		if config.get('data_norm') is None:
			config['data_norm'] = {
				'inp': {'sub': [0], 'div': [1]},
				'gt': {'sub': [0], 'div': [1]}
			}
		# data normalize
		data_norm = config['data_norm']
		t = data_norm['inp']
		self.inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).to(self.device) 
		self.inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).to(self.device)
		t = data_norm['gt']
		self.gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).to(self.device)
		self.gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).to(self.device)
	
	# def forward(self, lr, coord=None, cell=None, scale=None, kernel=None, state='test'):
	# 	with torch.no_grad():
	# 		# wavelet transform
	# 		w = wave(lr, self.wav) 
	# 		feature = self.encoder(w) # fix
		
	# 	if state == 'train':
	# 		kernel = self.kernel(feature) # (B, k, k)
	# 		lr_coord = make_coord((self.inp_size, self.inp_size))[None, ...]\
	# 					.expand(self.bs, -1, -1).to(self.device)
	# 		lr_cell = torch.ones_like(lr_coord).to(self.device)
	# 		lr_cell[..., 0] *= 2 / self.inp_size
	# 		lr_cell[..., 1] *= 2 / self.inp_size
			
	# 		coord = torch.cat((coord, lr_coord), dim=1)
	# 		cell = torch.cat((cell, lr_cell), dim=1)
   
	# 	else:
	# 		kernel = self.kernel(feature) # (B, k, k)
		
	# 	inp = (lr - self.inp_sub) / self.inp_div
	# 	pred_rgb = self.SR(inp, coord, cell, feature)
	# 	pred_rgb = pred_rgb*self.gt_div+self.gt_sub
	# 	pred_rgb.clamp_(0, 1)
		
	# 	if state == 'train':
	# 		pred_rgb, lr_clean = torch.split(pred_rgb, [self.qpt, self.inp_size**2], dim=1)
	# 		lr_blur = self.blur(lr_clean.permute(0, 2, 1).\
	# 				reshape(self.bs,  -1, self.inp_size, self.inp_size), kernel, 21)
	# 		### if add noise, these have to uncomment
	# 		#noise_level = torch.rand(self.bs, 1, 1, 1).to(self.device) * 10 
	# 		#noise = torch.randn_like(lr_blur).mul_(noise_level)
	# 		#lr_blur.add_(noise)
	# 		return pred_rgb, lr_blur, kernel
					  
	# 	return pred_rgb, kernel

	def forward(self, lr, coord=None, cell=None, scale=None, kernel=None, state='test'):
		with torch.no_grad():

			w = cuda_step_time("wavelet", lambda: wave(lr, self.wav))

			feature = cuda_step_time("encoder", lambda: self.encoder(w))

			kernel = cuda_step_time("kernel net", lambda: self.kernel(feature))

			inp = cuda_step_time("normalize",
								lambda: (lr - self.inp_sub) / self.inp_div)

			def sr_block():
				out = self.SR(inp, coord, cell, feature)
				out = out * self.gt_div + self.gt_sub
				out.clamp_(0, 1)
				return out

			pred_rgb = cuda_step_time("SR forward", sr_block)

			print("==========")

		return pred_rgb, kernel
	