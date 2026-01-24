import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

ckpt_dir = "/root/quoc-huy/KOALAnet_tfv2/pretrained/upsampling_network_x4"
ckpt_path = tf.train.latest_checkpoint(ckpt_dir)

print("Using checkpoint:", ckpt_path)

reader = tf.train.NewCheckpointReader(ckpt_path)
vars = reader.get_variable_to_shape_map()

print("Number of variables:", len(vars))
for k in sorted(vars.keys()):
    print(k)
    
# print("="*50)
# for k in sorted(vars):
#     if 'conv2d' in k and 'Adam' not in k:
#         print(k)
