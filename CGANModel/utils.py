"""
Most codes from https://github.com/carpedm20/DCGAN-tensorflow
"""
from __future__ import division
import scipy.misc
import numpy as np
import matplotlib
# matplotlib.use('AGG')
import matplotlib.pyplot as plt
import os
import gzip

import tensorflow as tf
import tensorflow.contrib.slim as slim


def load_mnist(dataset_name):
    data_dir = os.path.join("./data", dataset_name)

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data

    data = extract_data(
        data_dir + '/train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
    trX = data.reshape((60000, 28, 28, 1))

    data = extract_data(data_dir + '/train-labels-idx1-ubyte.gz', 60000, 8, 1)
    trY = data.reshape((60000))

    data = extract_data(
        data_dir + '/t10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
    teX = data.reshape((10000, 28, 28, 1))

    data = extract_data(data_dir + '/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
    teY = data.reshape((10000))

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1.0

    print(X.shape, y_vec.shape, "数据集的shape")
    return X / 255., y_vec


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def get_image(image_path, input_height, input_width, resize_height=64, resize_width=64, crop=True, grayscale=False):
    image = imread(image_path, grayscale)
    return transform(image, input_height, input_width, resize_height, resize_width, crop)


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def imread(path, grayscale=False):
    if (grayscale):
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)


def merge_images(images, size):
    return inverse_transform(images)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3, 4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3] == 1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:, :, 0]
        return img
    else:
        raise ValueError(
            'in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')


def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)


def center_crop(x, crop_h, crop_w, resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h) / 2.))
    i = int(round((w - crop_w) / 2.))
    return scipy.misc.imresize(x[j:j + crop_h, i:i + crop_w], [resize_h, resize_w])


def transform(image, input_height, input_width, resize_height=64, resize_width=64, crop=True):
    if crop:
        cropped_image = center_crop(
            image, input_height, input_width, resize_height, resize_width)
    else:
        cropped_image = scipy.misc.imresize(
            image, [resize_height, resize_width])
    return np.array(cropped_image) / 127.5 - 1.


def inverse_transform(images):
    return (images + 1.) / 2.


""" Drawing Tools """


# borrowed from https://github.com/ykwon0407/variational_autoencoder/blob/master/variational_bayes.ipynb


def save_scattered_image(z, id, z_range_x, z_range_y, name='scattered_image.jpg'):
    N = 10
    plt.figure(figsize=(8, 6))
    plt.scatter(z[:, 0], z[:, 1], c=np.argmax(id, 1), marker='o',
                edgecolor='none', cmap=discrete_cmap(N, 'jet'))
    plt.colorbar(ticks=range(N))
    axes = plt.gca()
    axes.set_xlim([-z_range_x, z_range_x])
    axes.set_ylim([-z_range_y, z_range_y])
    plt.grid(True)
    plt.savefig(name)


# borrowed from https://gist.github.com/jakevdp/91077b0cae40f8f8244a


def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


def load_pong(DATA_DIR="data/pong"):
    filelist = os.listdir(DATA_DIR)
    N = 100000
    if DATA_DIR[-4:] == "test":
        N = 3000
    obs_data = np.zeros((N, 84, 84, 4), dtype=np.uint8)
    a_data = np.zeros((N, 6), dtype=np.uint8)
    obs__data = np.zeros((N, 84, 84, 4), dtype=np.uint8)

    idx = 0
    for i, file in enumerate(filelist):
        obs = np.load(os.path.join(DATA_DIR, file))['obs']
        a_temp = np.load(os.path.join(DATA_DIR, file))['action']
        action = np.eye(6)[a_temp]
        obs_ = np.load(os.path.join(DATA_DIR, file))['obs_']

        l = len(obs)
        if (idx + l) > (N):
            break

        obs_data[idx:idx + l] = obs
        a_data[idx:idx + l] = action
        obs__data[idx:idx + l] = obs_

        idx += l
        del obs_, obs, a_temp, action

    s = obs_data[0:idx]
    a = a_data[0:idx]
    s_ = obs__data[0:idx]

    return s_, s, a


def load_pong_dqn(DATA_DIR="data/pong"):
    filelist = os.listdir(DATA_DIR)
    N = 10000
    obs_data = np.zeros((N, 84, 84, 4), dtype=np.uint8)
    a_data = np.zeros((N, 1), dtype=np.uint8)
    obs__data = np.zeros((N, 84, 84, 4), dtype=np.uint8)
    r_data = np.zeros((N, 1), dtype=np.int8)
    d_data = np.zeros((N, 1), dtype=np.uint8)

    idx = 0
    for i, file in enumerate(filelist):
        obs = np.load(os.path.join(DATA_DIR, file))['obs']
        n = len(obs)
        a = np.load(os.path.join(DATA_DIR, file))['action']
        a = np.reshape(a, [n, 1])
        obs_ = np.load(os.path.join(DATA_DIR, file))['obs_']
        reward = np.load(os.path.join(DATA_DIR, file))['r']
        reward = np.reshape(reward, [n, 1])
        d = np.reshape(np.load(os.path.join(DATA_DIR, file))['d'], [n, 1])

        l = len(obs)
        if (idx + l) > (N):
            break

        obs_data[idx:idx + l] = obs
        a_data[idx:idx + l] = a
        obs__data[idx:idx + l] = obs_
        r_data[idx:idx + l] = reward
        d_data[idx:idx + l] = d

        idx += l

    s = obs_data[0:idx]
    a = a_data[0:idx]
    s_ = obs__data[0:idx]
    r = r_data[0:idx]
    d = d_data[0:idx]

    return s_, s, a, r, d


def get_length(DATA_DIR="data/pong"):
    filelist = os.listdir(DATA_DIR)
    length = 0
    for i, file in enumerate(filelist):
        obs = np.load(os.path.join(DATA_DIR, file))['obs']
        length += len(obs)
    return length


def pic_handle(image, shape):
    # image_shape = (shape,84,84,4)
    H = W = 16
    ratio = H / 84
    results = np.zeros([shape, H, W, 4])
    for i in range(shape):
        for h in range(H):
            for w in range(W):
                x1 = int(h / ratio)
                x2 = int((h + 1) / ratio)
                y1 = int(w / ratio)
                y2 = int((w + 1) / ratio)
                for k in range(x1, x2):
                    for v in range(y1, y2):
                        results[i][h][w][0] = max(results[i][h][w][0],image[i][k][v][0])
                        results[i][h][w][1] = max(results[i][h][w][0],image[i][k][v][1])
                        results[i][h][w][2] = max(results[i][h][w][0],image[i][k][v][2])
                        results[i][h][w][3] = max(results[i][h][w][0],image[i][k][v][3])

    return results


if __name__ == '__main__':
    s_, s, a = load_pong(DATA_DIR="data/pong_test")
    print(s_.shape, a.shape, s.shape)
    s_ = pic_handle(s_[:3], 3)
    s_ = s_[0]
    obs = np.transpose(s_, (2, 0, 1))
    for j in range(4):
        plt.subplot(1, 4, j + 1)
        plt.imshow(obs[j])
    plt.show()
