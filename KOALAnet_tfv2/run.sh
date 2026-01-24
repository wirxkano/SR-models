# python main.py --phase 'train' --training_stage 3 --training_data_path '/root/media/quoc-huy/div2k_data/DIV2K_train_HR/DIV2K_train_HR' --validation_data_path '/root/media/quoc-huy/div2k_data/DIV2K_valid_HR/DIV2K_valid_HR'

# python main.py --phase 'test' --test_data_path '/root/quoc-huy/KOALAnet/testset/Set5/LR/X4/imgs' --test_label_path '/root/quoc-huy/KOALAnet/testset/Set5/HR' --test_ckpt_path '/root/quoc-huy/KOALAnet/pretrained'
# python main.py --phase 'test' --test_data_path '/root/quoc-huy/KOALAnet/testset/Set14/LR/X4/imgs' --test_label_path '/root/quoc-huy/KOALAnet/testset/Set14/HR' --test_ckpt_path '/root/quoc-huy/KOALAnet/pretrained'
python main.py --phase 'test'\
 --test_data_path '/root/quoc-huy/materials/Urban100/imgs' \
 --test_label_path '/root/quoc-huy/KOALAnet/testset/Urban100/HR' \
 --test_ckpt_path '/root/quoc-huy/KOALAnet/pretrained' \
 --save_dir /root/quoc-huy/all-tested-results/KOALAnet/Urban100
# python main.py --phase 'test' --test_data_path '/root/quoc-huy/KOALAnet/testset/Urban100/LR/X4/imgs' --test_label_path '/root/quoc-huy/KOALAnet/testset/Urban100/HR' --test_ckpt_path '/root/quoc-huy/KOALAnet/pretrained'


# python main.py --phase 'test' --test_data_path '/root/media/quoc-huy/blur-DAN/Set5/LRbic' --test_label_path '/root/quoc-huy/KOALAnet/testset/Set5/HR' --test_ckpt_path '/root/quoc-huy/KOALAnet/pretrained'
# python main.py --phase 'test' --test_data_path '/root/media/quoc-huy/blur-DAN/Set14/LRbic' --test_label_path '/root/quoc-huy/KOALAnet/testset/Set14/HR' --test_ckpt_path '/root/quoc-huy/KOALAnet/pretrained'
# python main.py --phase 'test' --test_data_path '/root/media/quoc-huy/blur-DAN/BSD100/LRbic' --test_label_path '/root/quoc-huy/KOALAnet/testset/BSD100/HR' --test_ckpt_path '/root/quoc-huy/KOALAnet/pretrained'
# python main.py --phase 'test' --test_data_path '/root/media/quoc-huy/blur-DAN/Urban100/LRbic' --test_label_path '/root/quoc-huy/KOALAnet/testset/Urban100/HR' --test_ckpt_path '/root/quoc-huy/KOALAnet/pretrained'

