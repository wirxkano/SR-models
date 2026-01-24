import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = False
sess = tf.Session(config=config)
run_meta = tf.RunMetadata()
opts = tf.profiler.ProfileOptionBuilder.float_operation()
# Run with profiling
output_test_p = sess.run(
    output_test, 
    feed_dict={test_lr: test_lr_p},
    options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
    run_metadata=run_meta
)

# Calculate FLOPs
flops = tf.profiler.profile(
    sess.graph,
    run_meta=run_meta,
    options=opts
)

gflops = flops.total_float_ops / 1e9
print(f'GFLOPs: {gflops:.2f}')
