import sys

sys.path.append("C:\\Users\\mi\\Desktop\\projects\\cgandqn\\DQN-master\\CGANModel")
from ops import *
from utils import *

is_training = True
#
s = tf.ones([10, 84, 84, 4])
a = tf.ones([10, 6])
# z = tf.ones([10, 16])

shape = tf.shape(s)[0]
net = lrelu(conv2d(s, 4, 5, 5, 4, 4, name='g_conv1'))
net = lrelu(bn(conv2d(net, 4, 5, 5, 4, 4, name='g_conv2'), is_training=is_training, scope='g_bn1'))
print(net.get_shape())
a = tf.reshape(a, [shape, 1, 1, 6])
a = a * tf.ones([shape, 6, 6, 6])
net = concat([net, a], 3)
net = tf.layers.flatten(net)
net = tf.nn.relu(bn(linear(net, 1024, scope='g_fc1'), is_training=is_training, scope='g_bn2'))
net = tf.nn.sigmoid(bn(linear(net, 84*84*4, scope='g_fc2'), is_training=is_training, scope='g_bn3'))

out = tf.reshape(net, [shape, 84, 84, 4])

print(net.get_shape())
