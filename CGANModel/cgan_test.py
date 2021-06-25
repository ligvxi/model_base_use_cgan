from CGAN import CGAN
from utils import show_all_variables

import tensorflow as tf


def main():
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        gan = CGAN(sess,
                   epoch=201,
                   batch_size=100,
                   z_dim=84,
                   dataset_name="data/pong",
                   checkpoint_dir="checkpoint",
                   result_dir="results",
                   log_dir="logs")

        # build graph
        gan.build_model()

        # show network architecture
        show_all_variables()

        gan.init_value()

        gan.load()
        gan.visualize_results(epoch="test", size=20)



if __name__ == '__main__':
    main()
