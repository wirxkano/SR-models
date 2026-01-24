# CUDA_VISIBLE_DEVICES=0 python train_degrade.py \
# --config configs/degradation.yaml \
# --tag degradation --savedir save \
# --queue

# CUDA_VISIBLE_DEVICES=0 python train_SR.py \
# --config configs/train-div2k/train_SR.yaml \
# --tag sr-model \
# --savedir save

CUDA_VISIBLE_DEVICES=0 python test.py \
--model_config /root/quoc-huy/arbitrary_scale_blind_SR/configs/train-div2k/train_SR.yaml \
--model_weight /root/quoc-huy/arbitrary_scale_blind_SR/checkpoints/epoch-best-author.pth \
--test_config /root/quoc-huy/arbitrary_scale_blind_SR/configs/test/test.yaml
