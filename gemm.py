import numpy as np
import pickle
import tensorflow as tf
from tensorflow.python.framework import graph_util
import time

def getms():
  return int(round(time.time() * 1000))

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


A = weight_variable([784, 75])
B = weight_variable([75, 20])

C = tf.matmul(A, B)


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
s = getms()
C.eval()
print("gemm..." + str(getms() - s) + " ms")




