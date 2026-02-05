# noise-free degradations with isotropic Gaussian blurs
# python test.py --test_only \
#                --dir_data='/root/media/quoc-huy/eval-wir/srbenchmarks' \
#                --data_test='Set14' \
#                --model='blindsr' \
#                --scale='4' \
#                --resume=600 \
#                --blur_type='iso_gaussian' \
#                --noise=0.0 \
#                --sig=0


# general degradations with anisotropic Gaussian blurs and noises
# python test.py --test_only \
#                --dir_data='/root/media/quoc-huy/eval-wir/srbenchmarks' \
#                --data_test='Set5' \
#                --model='blindsr' \
#                --scale='4' \
#                --resume=600 \
#                --blur_type='aniso_gaussian' \
#                --noise=0.0 \
#                --theta=0.0 \
#                --lambda_1=0.2 \
#                --lambda_2=4.0


######## custom setting for arbitrary LQ dataset
python test.py --test_only \
               --custom_testset=True \
               --hr_folder='/root/media/quoc-huy/eval-wir/srbenchmarks/Set5/HR' \
               --lr_folder='/root/media/quoc-huy/eval-wir/srbenchmarks/Set5/LR-bicubic-matlab-v2' \
               --save_folder='/root/quoc-huy/all-tested-results/DASR/Set5-LR-bicubic-matlab-v2' \
               --dir_data='/root/media/quoc-huy/eval-wir/srbenchmarks' \
               --data_test='Set5' \
               --model='blindsr' \
               --scale='4' \
               --resume=600 \
               --blur_type='iso_gaussian' \
               --noise=0 \
               --sig=0.0 \
               --calc_psnr=True

######## custom setting for focus-aware degradation
# python test.py --test_only \
#                --custom_testset=True \
#                --hr_folder='/root/media/quoc-huy/eval-wir/srbenchmarks/Set5/HR' \
#                --lr_folder='/root/media/quoc-huy/eval-wir/srbenchmarks/Set5/LR' \
#                --dir_data='/root/media/quoc-huy/eval-wir/srbenchmarks' \
#                --data_test='Set5' \
#                --model='blindsr' \
#                --scale='4' \
#                --resume=479 \
#                --blur_type='focus_aware' \
#                --noise=0 \
#                --sig=0.0

# python test.py --test_only \
#                --custom_testset=True \
#                --hr_folder='/root/media/quoc-huy/eval-wir/srbenchmarks/Set14/HR' \
#                --lr_folder='/root/media/quoc-huy/eval-wir/srbenchmarks/Set14/LR' \
#                --dir_data='/root/media/quoc-huy/eval-wir/srbenchmarks' \
#                --data_test='Set14' \
#                --model='blindsr' \
#                --scale='4' \
#                --resume=479 \
#                --blur_type='focus_aware' \
#                --noise=0 \
#                --sig=0.0

# python test.py --test_only \
#                --custom_testset=True \
#                --hr_folder='/root/media/quoc-huy/eval-wir/srbenchmarks/BSD100/HR' \
#                --lr_folder='/root/media/quoc-huy/eval-wir/srbenchmarks/BSD100/LR' \
#                --dir_data='/root/media/quoc-huy/eval-wir/srbenchmarks' \
#                --data_test='BSD100' \
#                --model='blindsr' \
#                --scale='4' \
#                --resume=479 \
#                --blur_type='focus_aware' \
#                --noise=0 \
#                --sig=0.0

# python test.py --test_only \
#                --custom_testset=True \
#                --hr_folder='/root/media/quoc-huy/eval-wir/srbenchmarks/Urban100/HR' \
#                --lr_folder='/root/quoc-huy/BSRGAN/testsets/RealSRSet' \
#                --dir_data='/root/media/quoc-huy/eval-wir/srbenchmarks' \
#                --data_test='Urban100' \
#                --model='blindsr' \
#                --scale='4' \
#                --resume=479 \
#                --blur_type='focus_aware' \
#                --noise=0 \
#                --sig=0.0
