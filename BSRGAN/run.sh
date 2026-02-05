# python main_test_bsrgan.py \
#   --input_lq /root/media/quoc-huy/blur-DAN/Set5/LRbic \
#   --output_path /root/quoc-huy/all-tested-results/BSRGAN/Set5-type5 \
#   --input_hq /root/media/quoc-huy/eval-wir/srbenchmarks/Set5/HR \
#   --model_name BSRGAN_custom \
#   --model_path /root/quoc-huy/BSRGAN/model_zoo

# python main_test_bsrgan.py \
#   --input_lq /root/media/quoc-huy/blur-DAN/Set14/LRbic \
#   --output_path /root/quoc-huy/all-tested-results/BSRGAN/Set14-type5 \
#   --input_hq /root/media/quoc-huy/eval-wir/srbenchmarks/Set14/HR \
#   --model_name BSRGAN_custom \
#   --model_path /root/quoc-huy/BSRGAN/model_zoo

# python main_test_bsrgan.py \
#   --input_lq /root/media/quoc-huy/blur-DAN/BSD100/LRbic \
#   --output_path /root/quoc-huy/all-tested-results/BSRGAN/BSD100-type5 \
#   --input_hq /root/media/quoc-huy/eval-wir/srbenchmarks/BSD100/HR \
#   --model_name BSRGAN_custom \
#   --model_path /root/quoc-huy/BSRGAN/model_zoo

# python main_test_bsrgan.py \
#   --input_lq /root/media/quoc-huy/blur-DAN/Urban100/LRbic \
#   --output_path /root/quoc-huy/all-tested-results/BSRNet/Urban100-type1v2 \
#   --input_hq /root/media/quoc-huy/eval-wir/srbenchmarks/Urban100/HR \
#   --model_name BSRNet_custom \
#   --model_path /root/quoc-huy/BSRGAN/model_zoo

python main_test_bsrgan.py \
  --input_lq /root/media/quoc-huy/eval-wir/srbenchmarks/Set5/LR-bicubic-matlab-v2 \
  --output_path /root/quoc-huy/all-tested-results/BSRGAN/Set5-LR-bicubic-matlab-v2 \
  --input_hq /root/media/quoc-huy/eval-wir/srbenchmarks/Set5/HR \
  --model_name BSRGAN \
  --model_path /root/quoc-huy/BSRGAN/model_zoo