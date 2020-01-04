import numpy as np
import pickle
import tensorflow as tf
from tensorflow.python.framework import graph_util

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def LeNet(inputs):

    W_conv1 = weight_variable([5, 5, 3, 20])
    b_conv1 = bias_variable([20])

    W_conv2 = weight_variable([5, 5, 20, 50])
    b_conv2 = bias_variable([50])

    W_conv3 = weight_variable([5, 5, 50, 500])
    b_conv3 = bias_variable([500])

    W_fc1 = weight_variable([500, 10])
    b_fc1 = bias_variable([10])

    h_conv1 = conv2d(inputs, W_conv1) + b_conv1
    h_bn1   = tf.layers.batch_normalization(h_conv1)
    h_relu1 = tf.nn.relu(h_bn1)
    h_pool1 = max_pool_2x2(h_relu1)

    h_conv2 = conv2d(h_pool1, W_conv2) + b_conv2
    h_bn2   = tf.layers.batch_normalization(h_conv2)
    h_relu2 = tf.nn.relu(h_bn2)
    h_pool2 = max_pool_2x2(h_relu2)

    h_conv3 = conv2d(h_pool2, W_conv3) + b_conv3
    h_bn3   = tf.layers.batch_normalization(h_conv3)
    h_relu3 = tf.nn.relu(h_bn3)

    h_relu3_flat = tf.reshape(h_relu3, [-1, 500])

    outputs = tf.nn.softmax(tf.matmul(h_relu3_flat, W_fc1) + b_fc1, name="softmax")
    return outputs

def one_hot_encode(x):
    encoded = np.zeros((len(x), 10))
    
    for idx, val in enumerate(x):
        encoded[idx][val] = 1
    encoded = np.asarray(encoded, dtype=np.float32) 
    return encoded

def load_cfar10_batch(path, batch_id):
    with open(path + '/data_batch_' + str(batch_id), mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')
        
    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']
    features = np.asarray(features, dtype=np.float32)
    features = (features - 127.5)/127.5 
    return features, one_hot_encode(labels)

train_data = None
train_label = None
for i in range(1, 6):
  features, labels = load_cfar10_batch("cifar-10-batches-py", i)
  if train_data is None:
    train_data = features
    train_label = labels
  else:
    train_data = np.concatenate((train_data, features), axis=0)
    train_label = np.concatenate((train_label, labels), axis=0)


inputs = tf.placeholder(tf.float32, [None, 32, 32, 3], "input")
targets = tf.placeholder(tf.float32, [None, 10])

outputs = LeNet(inputs)

cross_entropy = -tf.reduce_sum(targets*tf.log(outputs))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess = tf.InteractiveSession()

sess.run(tf.global_variables_initializer())
constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['softmax'])

BATCH = 100
for i in range(1000):

    step = BATCH*i%50000
    batch_xs = train_data[step:step+BATCH, :, : ,:]
    batch_ys = train_label[step:step+BATCH, :]
    #print(batch_ys)
    #print(batch_xs.shape)
    #print(batch_ys.shape)
    
    if i % 10 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            inputs: batch_xs, targets: batch_ys})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    
    train_step.run(feed_dict={inputs: batch_xs, targets: batch_ys})
with tf.gfile.FastGFile("cifar_frozen_model.pb", mode='wb') as f:
  f.write(constant_graph.SerializeToString())



