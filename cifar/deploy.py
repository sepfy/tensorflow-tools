import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

import time

def getms():
  return int(round(time.time() * 1000))

sess = tf.Session(config=
    tf.ConfigProto(inter_op_parallelism_threads=1,
                   intra_op_parallelism_threads=1,
                   device_count = {"GPU": 0}))
'''
sess = tf.Session()
'''
with gfile.FastGFile("cifar_frozen_model.pb", "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

sess.run(tf.global_variables_initializer())

# Show all op 
for op in sess.graph.get_operations():
   print(op.name,op.values())

x = sess.graph.get_tensor_by_name("input:0")
softmax = sess.graph.get_tensor_by_name("softmax:0")
batch_xs = np.random.random_sample((1, 32, 32, 3))
out = sess.run(softmax, feed_dict={x: batch_xs})
#print(batch_xs.shape)
batch_xs = np.random.random_sample((1, 32, 32, 3))
start = getms()
out = sess.run(softmax, feed_dict={x: batch_xs})
#print(out)
print("forward...... " + str(getms() - start) + " ms")

