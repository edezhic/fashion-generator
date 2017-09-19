%cd ./model

import os
import tensorflow as tf
import keras
from utils import *
from GAN import GAN

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)

# You can download fashion-mnist dataset here https://github.com/zalandoresearch/fashion-mnist
# Place .gz files in model/data/fashion-mnist folder
dataset_name = 'fashion-mnist'
epoch_num = 50
batch_size = 64
z_dim = 60
checkpoint_dir = 'checkpoint'
result_dir = 'results'
log_dir = 'logs'
weights_dir = '../weights'

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)) as sess:
        # declare instance for GAN
        gan = GAN(sess, epoch=epoch_num, batch_size=batch_size, z_dim=z_dim, dataset_name=dataset_name,
                 checkpoint_dir=checkpoint_dir, result_dir=result_dir, log_dir=log_dir)

        # build graph
        gan.build_model()

        # show network architecture
        show_all_variables()

        # launch the graph in a session
        gan.train()
        print(" [*] Training finished!")

        # visualize learned generator
        gan.visualize_results(epoch_num-1)
        print(" [*] Testing finished!")

        dump_checkpoint_vars(gan.checkpoint_path, output_dir=weights_dir, remove_variables_regex=".*(discriminator|adam|power|apply|YF)")
        print(" [*] Weights saved!")
