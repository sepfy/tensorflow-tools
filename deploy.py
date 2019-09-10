#-*- coding:utf-8 -*-
import numpy as np
import argparse 
import tensorflow as tf
from util import load_images
from util import create_label
 
def load_graph(frozen_graph_filename):
    # We parse the graph_def file
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
 
    # We load the graph_def in the default graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def, 
            input_map=None, 
            return_elements=None, 
            name="prefix", 
            op_dict=None, 
            producer_op_list=None
        )
    return graph
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default="output/frozen_model.pb", type=str, help="Frozen model file to import")
    args = parser.parse_args()
    #加载已经将参数固化后的图
    graph = load_graph(args.frozen_model_filename)
 
    # We can list operations
    #op.values() gives you a list of tensors it produces
    #op.name gives you the name
    #输入,输出结点也是operation,所以,我们可以得到operation的名字
    for op in graph.get_operations():
        print(op.name,op.values())
        # prefix/Placeholder/inputs_placeholder
        # ...
        # prefix/Accuracy/predictions
    #操作有:prefix/Placeholder/inputs_placeholder
    #操作有:prefix/Accuracy/predictions
    #为了预测,我们需要找到我们需要feed的tensor,那么就需要该tensor的名字
    #注意prefix/Placeholder/inputs_placeholder仅仅是操作的名字,prefix/Placeholder/inputs_placeholder:0才是tensor的名字
    x = graph.get_tensor_by_name('prefix/Placeholder:0')
    y = graph.get_tensor_by_name('prefix/softmax/Softmax:0')
    keep_prob = graph.get_tensor_by_name('prefix/Placeholder_2:0')
       
    v_pdata = load_images("t_pdata", 100)
    v_plabel = create_label((v_pdata.shape[0], 2), 0)

    v_ndata = load_images("t_ndata", 100)
    v_nlabel = create_label((v_ndata.shape[0], 2), 1)

    v_data = np.concatenate((v_pdata, v_ndata), 0)
    v_label = np.concatenate((v_plabel, v_nlabel), 0)
 
    with tf.Session(graph=graph) as sess:
      s = 0
      for j in range(10):
        ds = 10*j
        de = 10*(j+1)
        #s = s + accuracy.eval(feed_dict={x:v_data[ds:de], y_: v_label[ds:de], keep_prob: 1.0})
        out = sess.run(y, feed_dict={x: v_data[ds:de], keep_prob:1.0}) 
        pred = np.argmax(out, axis=1)
        label = np.argmax(v_label[ds:de], axis=1)
        res = np.mean(pred-label)
        s += res/10.0
      print("validation accuracy %g"%(s))

        #y_out = sess.run(y, feed_dict={
        #    x: [[3, 5, 7, 4, 5, 1, 1, 1, 1, 1]] # < 45
        #})
        #print(y_out) # [[ 0.]] Yay!
    print ("finish")

