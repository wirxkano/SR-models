# python train.py --opt options/train/train_stage1.yml
# python train.py --opt options/train/train_stage2.yml
# python train.py --opt options/train/train_stage3.yml

# python test.py --opt options/test/test_stage1.yml
# python test.py --opt options/test/test_stage2.yml
cd codes
python test.py --opt /root/quoc-huy/MANet/codes/options/test/test_stage3.yml

# python prepare_testset.py --opt /root/quoc-huy/MANet/codes/options/test/prepare_testset.yml
