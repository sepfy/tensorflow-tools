import tensorflow as tf
import numpy as np
from util import load_images
from util import create_label


class VGGNet():
  def __init__(self):
    pass

  def get_weight_var(self.shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  def get_bias_var(self.shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  def conv2d(self, inputs, weight):
    return tf.nn.conv2d(inputs, weight, strides=[1, 1, 1, 1], padding='SAME')


  def conv_layer(self, inputs, ksize, c, fc, name):
    with tf.variable_scope(name):
      weight = self.get_weight_var([ksize, ksize, c, fc])
      bias = self.get_bias_var([fc])
      out = tf.nn.relu(self.conv2d(inputs, weight) + bias)
    return out

  def maxpool_layer(self, inputs, name):
    with tf.variable_scope(name):
      out = tf.nn.max_pool(inputs, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    return out

  def fc_layer(self, inputs, n, m, name):
    with tf.variable_scope(name):
      weight = self.get_weight_var([n, m])
      bias = self.get_bias_var([m])
      out = tf.nn.relu(tf.matmul(inputs, weight) + bias)
    return out


  def build_net(self, inputs):

    self.conv11 = self.conv_layer(inputs, 3, 3, 64, "conv1-1")
    self.pool11 = self.maxpool_layer(self.conv11, "pool1-1")
    self.conv12 = self.conv_layer(self.pool11, 3, 64, 64, "conv1-2")
    self.pool12 = self.maxpool_layer(self.conv12, "pool1-2")


pdata = load_images("t_pdata", 1700)
plabel = create_label((pdata.shape[0], 2), 0)

ndata = load_images("t_ndata", 1700)
nlabel = create_label((ndata.shape[0], 2), 1)

data = np.concatenate((pdata, ndata), 0)
label = np.concatenate((plabel, nlabel), 0)

print(data.shape)
print(label.shape)

sess = tf.InteractiveSession() 
x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
y_ = tf.placeholder(tf.float32, shape=[None, 2])
keep_prob = tf.placeholder(tf.float32)
y = network(x, keep_prob)
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver() 

sess.run(tf.global_variables_initializer())
for i in range(2000):

  s = 0
  for j in range(170):
    xdata = np.concatenate((pdata[j*10:(j+1)*10,:,:,:], \
                          ndata[j*10:(j+1)*10,:,:,:]), 0)
    ydata = np.concatenate((plabel[j*10:(j+1)*10,:], \
                          nlabel[j*10:(j+1)*10,:]), 0)
    train_step.run(feed_dict={x: xdata, y_: ydata, keep_prob: 0.5})
    s = s + accuracy.eval(feed_dict={x: xdata, y_: ydata, keep_prob: 1.0})
  s = s/170.0
  print("epoach %d, accuracy %g"%(i, s))
  if s > 0.99:
    break


v_pdata = load_images("t_pdata", 500)
v_plabel = create_label((pdata.shape[0], 2), 0)

v_ndata = load_images("t_ndata", 500)
v_nlabel = create_label((ndata.shape[0], 2), 1)

v_data = np.concatenate((v_pdata, v_ndata), 0)
v_label = np.concatenate((v_plabel, v_nlabel), 0)

# validation
s = 0
for j in range(50):
  ds = 10*j
  de = 10*(j+1)
  s = s + accuracy.eval(feed_dict={x:v_data[ds:de], y_: v_label[ds:de], keep_prob: 1.0})
s = s/50.0
print("validation accuracy %g"%(s))

saver.save(sess, "output/smoke")
