import tensorflow as tf
import numpy as np
from util import load_images
from util import create_label
import tensorflow.contrib.slim as slim
import time

def VGG12(net, keep_prob):


  net = slim.conv2d(net, 32, [3, 3], stride=1, scope="conv1")
  net = slim.max_pool2d(net, [2, 2], scope='pool1')
  net = slim.conv2d(net, 64, [3, 3], stride=1, scope="conv2")
  net = slim.max_pool2d(net, [2, 2], scope='pool2')
  
  net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv3')
  #net = slim.separable_conv2d(net, None, [3, 3], depth_multiplier=1, stride=1)
  #net = slim.conv2d(net, 128, [1, 1], stride=1)
  #net = slim.separable_conv2d(net, None, [3, 3], depth_multiplier=1, stride=1)
  #net = slim.conv2d(net, 128, [1, 1], stride=1)
  net = slim.max_pool2d(net, [2, 2], scope='pool3')

  #net = slim.repeat(net, 2, slim.conv2d, 256, [3, 3], scope='conv4')
  net = slim.separable_conv2d(net, None, [3, 3], depth_multiplier=1, stride=1)
  net = slim.conv2d(net, 256, [1, 1], stride=1)
  net = slim.separable_conv2d(net, None, [3, 3], depth_multiplier=1, stride=1)
  net = slim.conv2d(net, 256, [1, 1], stride=1)
  net = slim.max_pool2d(net, [2, 2], scope='pool4')

  #net = slim.repeat(net, 2, slim.conv2d, 256, [3, 3], scope='conv5')
  net = slim.separable_conv2d(net, None, [3, 3], depth_multiplier=1, stride=1)
  net = slim.conv2d(net, 256, [1, 1], stride=1)
  net = slim.separable_conv2d(net, None, [3, 3], depth_multiplier=1, stride=1)
  net = slim.conv2d(net, 256, [1, 1], stride=1)

  net = slim.max_pool2d(net, [2, 2], scope='pool5')

  net = slim.flatten(net, scope="flat1")

  net = slim.fully_connected(net, 1024, scope='fc1')
  net = slim.dropout(net, keep_prob, scope='drop1')
  net = slim.fully_connected(net, 1024, scope='fc2')
  net = slim.dropout(net, keep_prob, scope='drop2')
  net = slim.fully_connected(net, 2, activation_fn=None, scope='fc3')

  return net

pdata = load_images("t_pdata", 170)
plabel = create_label((pdata.shape[0], 2), 0)

ndata = load_images("t_ndata", 170)
nlabel = create_label((ndata.shape[0], 2), 1)

data = np.concatenate((pdata, ndata), 0)
label = np.concatenate((plabel, nlabel), 0)

print(data.shape)
print(label.shape)

def mobilenet(inputs):
  mobilenet_conv_defs = [
    {"kernel": [3, 3], "stride": 1, "depth": 64},
    {"kernel": [3, 3], "stride": 2, "depth": 128},
    {"kernel": [3, 3], "stride": 1, "depth": 128},
    {"kernel": [3, 3], "stride": 2, "depth": 256},
    {"kernel": [3, 3], "stride": 1, "depth": 256},
    {"kernel": [3, 3], "stride": 2, "depth": 512},
    {"kernel": [3, 3], "stride": 1, "depth": 512},
    {"kernel": [3, 3], "stride": 1, "depth": 512},
    {"kernel": [3, 3], "stride": 1, "depth": 512},
    {"kernel": [3, 3], "stride": 1, "depth": 512},
    {"kernel": [3, 3], "stride": 1, "depth": 512},
    {"kernel": [3, 3], "stride": 2, "depth": 1024},
    {"kernel": [3, 3], "stride": 1, "depth": 1024}
  ]
  net = slim.conv2d(inputs, 32, [3, 3], stride=2, scope="conv2d_0")
  for idx, conv_def in enumerate(mobilenet_conv_defs):
    kernel = conv_def["kernel"]
    stride = conv_def["stride"]
    depth = conv_def["depth"]
    scope = "conv2d_depthwise_" + str(idx)
    net = slim.separable_conv2d(net, None, kernel, depth_multiplier=1, stride=stride, scope=scope)
    scope = "conv2d_pointwise_" + str(idx)
    net = slim.conv2d(net, depth, [1, 1], stride=1, scope=scope)
  net = slim.avg_pool2d(net, [7, 7], stride=1, scope="avg_pool2d_0")
  net = slim.flatten(net, scope="flattern0")
  print(net.shape)
  net = slim.fully_connected(net, 2, scope='fc0')
  return net

sess = tf.InteractiveSession() 

x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
y_ = tf.placeholder(tf.float32, shape=[None, 2])
keep_prob = tf.placeholder(tf.float32)

outputs = VGG12(x, keep_prob)
softmax = slim.softmax(outputs, scope="softmax")
loss = slim.losses.softmax_cross_entropy(outputs, y_)
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
correct_prediction = tf.equal(tf.argmax(outputs,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver() 

sess.run(tf.global_variables_initializer())
for i in range(2000):

  s = 0
  
  for j in range(17):
    xdata = np.concatenate((pdata[j*10:(j+1)*10,:,:,:], \
                          ndata[j*10:(j+1)*10,:,:,:]), 0)
    ydata = np.concatenate((plabel[j*10:(j+1)*10,:], \
                          nlabel[j*10:(j+1)*10,:]), 0)
    train_step.run(feed_dict={x: xdata, y_: ydata, keep_prob: 1.0})
    start = time.time()
    s = s + accuracy.eval(feed_dict={x: xdata, y_: ydata, keep_prob: 1.0})
    #print("inference time = " + str(1000*(time.time()-start)))
  s = s/17.0
  print("epoach %d, accuracy %g"%(i, s))
  if s > 0.99:
    break

'''
v_pdata = load_images("t_pdata", 100)
v_plabel = create_label((pdata.shape[0], 2), 0)

v_ndata = load_images("t_ndata", 100)
v_nlabel = create_label((ndata.shape[0], 2), 1)

v_data = np.concatenate((v_pdata, v_ndata), 0)
v_label = np.concatenate((v_plabel, v_nlabel), 0)

# validation
s = 0
for j in range(10):
  ds = 10*j
  de = 10*(j+1)
  s = s + accuracy.eval(feed_dict={x:v_data[ds:de], y_: v_label[ds:de], keep_prob: 1.0})
s = s/10.0
print("validation accuracy %g"%(s))
'''
saver.save(sess, "output/smoke")
