# -*- coding: utf-8 -*-
from __future__ import division
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys

sys.path.append("/home/lab/Pycharm_Projects/liguixi/cgandqn/DQN-master/CGANModel/")
import time
from ops import *
from utils import *
import random

np.set_printoptions(threshold=np.inf)


class CGAN(object):
    model_name = "CGAN"  # name for checkpoint

    def __init__(self, sess, epoch, batch_size, z_dim, dataset_name, checkpoint_dir, result_dir, log_dir):
        self.sess = sess
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        self.epoch = epoch
        self.batch_size = batch_size

        self.model_dir = "./checkpoint/cgan"
        # self.model_dir = "./CGANModel/checkpoint/cgan"

        self.input_height = 84
        self.input_width = 84
        self.output_height = 84
        self.output_width = 84

        self.z_dim = z_dim  # dimension of noise-vector
        self.a_dim = 6  # dimension of condition-vector (action)
        self.c_dim = 4  # pic channel

        # train
        self.learning_rate = 0.0002
        self.beta1 = 0.5

        # test
        self.sample_num = 64  # number of generated images to be saved

        # load mnist
        # self.data_X, self.data_y = load_mnist(self.dataset_name)
        self.obs_, self.obs, self.action = load_pong(self.dataset_name)  # s_, s, a
        self.test_obs_, self.test_obs, self.test_action = load_pong(self.dataset_name + "_test")  # s_, s, a
        n = len(self.test_obs_)
        self.test_z = np.ones([n, self.z_dim])
        print('train_shape:', len(self.obs_))
        print('test_shape:', n)

        # get number of batches for a single epoch
        self.num_batches = len(self.obs_) // self.batch_size

    def discriminator(self, s_, s, a, is_training=True, reuse=False):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
        with tf.variable_scope("discriminator", reuse=reuse):
            x = conv_cond_concat(s_, s, a)
            net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='d_conv1'))
            net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='d_conv2'), is_training=is_training, scope='d_bn2'))
            net = tf.layers.flatten(net)
            net = lrelu(bn(linear(net, 1024, scope='d_fc3'), is_training=is_training, scope='d_bn3'))
            out_logit = linear(net, 1, scope='d_fc4')
            out = tf.nn.sigmoid(out_logit)

            return out, out_logit, net

    def generator(self, z, s, a, is_training=True, reuse=False):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
        # not use z
        with tf.variable_scope("generator", reuse=reuse):
            shape = tf.shape(s)[0]
            s = tf.reshape(s, [shape, 84 * 84 * 4])
            z = concat([z, s, a], 1)
            net = tf.nn.relu(bn(linear(z, 1024, scope='g_fc1'), is_training=is_training, scope='g_bn1'))
            net = tf.nn.relu(bn(linear(net, 128 * 7 * 7, scope='g_fc2'), is_training=is_training, scope='g_bn2'))
            net = tf.nn.sigmoid(bn(linear(net, 84 * 84 * 4, scope='g_fc3'), is_training=is_training, scope='g_bn3'))
            out = tf.reshape(net, [shape, 84, 84, 4])
            return out

    def build_model(self):
        """ Graph Input """
        self.s_ = tf.placeholder(tf.float32, [None, 84, 84, 4], name='s_')
        self.s = tf.placeholder(tf.float32, [None, 84, 84, 4], name='s')
        self.a = tf.placeholder(tf.float32, [None, 6], name='a')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        """ Loss Function """

        # output of D for real images
        D_real, D_real_logits, _ = self.discriminator(self.s_, self.s, self.a, is_training=True, reuse=False)

        # output of D for fake images
        G = self.generator(self.z, self.s, self.a, is_training=True, reuse=False)
        D_fake, D_fake_logits, _ = self.discriminator(G, self.s, self.a, is_training=True, reuse=True)

        # get loss for discriminator
        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real)))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake)))
        self.d_loss = d_loss_real + d_loss_fake
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake)))

        """ Training """
        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]

        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                .minimize(self.d_loss, var_list=d_vars)
            self.g_optim = tf.train.AdamOptimizer(self.learning_rate * 5, beta1=self.beta1) \
                .minimize(self.g_loss, var_list=g_vars)

        # self.clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in d_vars]

        """" Testing """
        # for test
        self.fake_images = self.generator(self.z, self.s, self.a, is_training=False, reuse=True)
        print(self.fake_images.get_shape(), 'fake-image')

        """ Summary """
        d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
        d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)
        d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)

        # final summary operations
        self.g_sum = tf.summary.merge([d_loss_fake_sum, g_loss_sum])
        self.d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum])

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name, self.sess.graph)

    def init_value(self):
        # initialize all variables
        tf.global_variables_initializer().run()

    def train(self):
        self.load()

        start_epoch = 0
        start_batch_id = 0
        counter = 1

        # loop for epoch
        start_time = time.time()
        M = []
        min_mse = 999
        for epoch in range(start_epoch, self.epoch):
            # get batch data
            print("now epoch is:", epoch)
            for idx in range(start_batch_id, self.num_batches):
                batch_next_obs = self.obs_[idx * self.batch_size:(idx + 1) * self.batch_size]
                batch_obs = self.obs[idx * self.batch_size:(idx + 1) * self.batch_size]
                batch_action = self.action[idx * self.batch_size:(idx + 1) * self.batch_size]

                batch_z = np.ones([self.batch_size, self.z_dim])

                # update D network
                # self.sess.run(self.clip_D)
                _, summary_str, d_loss = self.sess.run([self.d_optim, self.d_sum, self.d_loss],
                                                       feed_dict={self.s_: batch_next_obs, self.s: batch_obs,
                                                                  self.a: batch_action, self.z: batch_z})

                self.writer.add_summary(summary_str, counter)

                # update G network
                _, summary_str, g_loss = self.sess.run([self.g_optim, self.g_sum, self.g_loss],
                                                       feed_dict={self.s: batch_obs, self.a: batch_action,
                                                                  self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                # display training status
                counter += 1
                # print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, " \
                #     % (epoch, idx, self.num_batches, time.time() - start_time, d_loss, g_loss))
            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model
            mse = self.get_mse()
            print('epoch:', epoch, ',mse:', mse)
            M.append(mse)

            if mse < min_mse:
                print('save:', epoch)
                min_mse = mse
                self.save()
                self.visualize_results(epoch)
        with open("mse.txt", 'w', encoding='utf8') as f:
            f.write(str(M))

    def get_mse(self):
        samples = self.sess.run(self.fake_images,
                                feed_dict={self.z: self.test_z, self.s: self.test_obs, self.a: self.test_action})
        mse = (np.square(self.test_obs_ - samples)).mean()
        return mse

    def visualize_results(self, epoch, size=5):
        # product fake imag from test set
        print('gen fake image:...')
        random_list = random.sample(range(len(self.test_obs)), size)
        next_s_list = np.array([self.test_obs_[i] for i in random_list])
        s_list = np.array([self.test_obs[i] for i in random_list])
        a_list = np.array([self.test_action[i] for i in random_list])
        z_sample = np.ones((size, self.z_dim))
        # (10, 16)(10, 84, 84, 1)(10, 6)
        print([a for a in a_list], 'g_test')
        samples = self.sess.run(self.fake_images,
                                feed_dict={self.z: z_sample, self.s: s_list, self.a: a_list})  # (5, 84, 84, 4)

        obs = np.transpose(samples, (0, 3, 1, 2))
        x = np.transpose(next_s_list, (0, 3, 1, 2))
        print(obs.shape, x.shape, 'aaaaaaaaaaaa')
        for i in range(size):
            for j in range(4):
                plt.subplot(2, 4, j + 1)
                plt.imshow(x[i][j])
                plt.subplot(2, 4, j + 5)
                plt.imshow(obs[i][j])
            plt.savefig("results/epoch{}_{}.png".format(str(epoch), str(i)))

    def save(self, epoch="_"):
        model_path = self.model_dir + epoch
        self.saver.save(self.sess, model_path)

    def load(self, epoch='_'):
        model_path = self.model_dir + epoch
        try:
            self.saver.restore(self.sess, model_path)
            print('load success')
            return True
        except Exception as e:
            print(e, 'load error')
            return False

    def get_next_s(self, s, a):
        s_list = np.reshape(s, [1, 84, 84, 4])
        a_list = np.zeros((1, 6), dtype=np.uint8)
        a = np.eye(6)[a]
        a_list[0] = a
        z_sample = np.ones((1, self.z_dim))
        # (1, 16)(1, 84, 84, 1)(1, 6)
        samples = self.sess.run(self.fake_images,
                                feed_dict={self.z: z_sample, self.s: s_list, self.a: a_list})

        return samples[0]


if __name__ == '__main__':
    pass
