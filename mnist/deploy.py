import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.platform import gfile

import time

def getms():
  return int(round(time.time() * 1000))


sess = tf.Session()
with gfile.FastGFile("mnist_frozen_model.pb", "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

#sess = tf.Session(config=
#    tf.ConfigProto(inter_op_parallelism_threads=1,
#                   intra_op_parallelism_threads=1))
sess.run(tf.global_variables_initializer())

# Show all op 
for op in sess.graph.get_operations():
   print(op.name,op.values())

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print(mnist.test.labels[0])
x = sess.graph.get_tensor_by_name("input:0")
batch_xs = mnist.test.images[0].reshape([1, 28, 28, 1])
softmax = sess.graph.get_tensor_by_name("softmax:0")
start = getms()
out = sess.run(softmax, feed_dict={x: batch_xs})

print(out)
print("forward...... " + str(getms() - start) + " ms")

