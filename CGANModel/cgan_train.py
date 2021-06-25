from CGAN import CGAN
from utils import show_all_variables

import tensorflow as tf


def main():
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        gan = CGAN(sess,
                   epoch=201,
                   batch_size=128,
                   z_dim=16,
                   dataset_name="budget",
                   checkpoint_dir="budget_checkpoint",
                   result_dir="budget_results",
                   log_dir="logs")

        # build graph
        gan.build_model()

        # show network architecture
        show_all_variables()

        gan.init_value()
        gan.visualize_results(epoch=888)
        gan.get_mse()

        # launch the graph in a session
        gan.train()



if __name__ == '__main__':
    main()
