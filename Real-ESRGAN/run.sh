# -i /root/media/quoc-huy/eval-wir/real-esrgan/Set5 

# python inference_realesrgan.py \
#   -i /root/media/quoc-huy/blur-DAN/Set5/LRbic \
#   -o /root/quoc-huy/all-tested-results/Real-ESRGAN/Set5-type1v2 \
#   --ground_truth /root/media/quoc-huy/eval-wir/srbenchmarks/Set5/HR \
#   --model_name RealESRGAN_x4plus \
#   --model_path /root/quoc-huy/Real-ESRGAN/experiments/pretrained_models/net_g_95000_gan_custom.pth

# python inference_realesrgan.py \
#   -i /root/media/quoc-huy/blur-DAN/Set14/LRbic \
#   -o /root/quoc-huy/all-tested-results/Real-ESRGAN/Set14-type1v2 \
#   --ground_truth /root/media/quoc-huy/eval-wir/srbenchmarks/Set14/HR \
#   --model_name RealESRGAN_x4plus \
#   --model_path /root/quoc-huy/Real-ESRGAN/experiments/pretrained_models/net_g_95000_gan_custom.pth

# python inference_realesrgan.py \
#   -i /root/media/quoc-huy/blur-DAN/BSD100/LRbic \
#   -o /root/quoc-huy/all-tested-results/Real-ESRGAN/BSD100-type1v2 \
#   --ground_truth /root/media/quoc-huy/eval-wir/srbenchmarks/BSD100/HR \
#   --model_name RealESRGAN_x4plus \
#   --model_path /root/quoc-huy/Real-ESRGAN/experiments/pretrained_models/net_g_95000_gan_custom.pth

# python inference_realesrgan.py \
#   -i /root/media/quoc-huy/blur-DAN/Urban100/LRbic \
#   -o /root/quoc-huy/all-tested-results/Real-ESRGAN/Urban100-type1v2 \
#   --ground_truth /root/media/quoc-huy/eval-wir/srbenchmarks/Urban100/HR \
#   --model_name RealESRGAN_x4plus \
#   --model_path /root/quoc-huy/Real-ESRGAN/experiments/pretrained_models/net_g_95000_gan_custom.pth

python inference_realesrgan.py \
  -i /root/media/quoc-huy/eval-wir/srbenchmarks/Set5/LR_bsrgan \
  -o /root/quoc-huy/all-tested-results/Real-ESRGAN/Set5-LR_bsrgan \
  --ground_truth /root/media/quoc-huy/eval-wir/srbenchmarks/Set5/HR \
  --model_name RealESRGAN_x4plus \

# python inference_realesrgan.py \
#   -i /root/media/quoc-huy/eval-wir/srbenchmarks/Set14/LR \
#   -o /root/quoc-huy/all-tested-results/Real-ESRGAN/Set14-type5 \
#   --ground_truth /root/media/quoc-huy/eval-wir/srbenchmarks/Set14/HR \
#   --model_name RealESRGAN_x4plus \
#   --model_path /root/quoc-huy/Real-ESRGAN/experiments/pretrained_models/net_g_95000_gan_custom.pth

# python inference_realesrgan.py \
#   -i /root/media/quoc-huy/eval-wir/srbenchmarks/BSD100/LR \
#   -o /root/quoc-huy/all-tested-results/Real-ESRGAN/BSD100-type5 \
#   --ground_truth /root/media/quoc-huy/eval-wir/srbenchmarks/BSD100/HR \
#   --model_name RealESRGAN_x4plus \
#   --model_path /root/quoc-huy/Real-ESRGAN/experiments/pretrained_models/net_g_95000_gan_custom.pth

# python inference_realesrgan.py \
#   -i /root/media/quoc-huy/eval-wir/srbenchmarks/Urban100/LR \
#   -o /root/quoc-huy/all-tested-results/Real-ESRGAN/Urban100-type5 \
#   --ground_truth /root/media/quoc-huy/eval-wir/srbenchmarks/Urban100/HR \
#   --model_name RealESRGAN_x4plus \
#   --model_path /root/quoc-huy/Real-ESRGAN/experiments/pretrained_models/net_g_95000_gan_custom.pth

# python inference_realesrgan.py \
#   -i /root/quoc-huy/BSRGAN/testsets/RealSRSet \
#   -o /root/quoc-huy/all-tested-results/Real-ESRGAN/RealSRSet-custom \
#   --ground_truth /root/media/quoc-huy/eval-set/Set5 \
#   --model_name RealESRGAN_x4plus \
#   --model_path /root/quoc-huy/Real-ESRGAN/experiments/pretrained_models/net_g_95000_gan_custom.pth \
#   --real_dataset True
