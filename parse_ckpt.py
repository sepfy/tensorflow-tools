import os
from tensorflow.python import pywrap_tensorflow
 
# code for finall ckpt
#checkpoint_path = "output/smoke"
checkpoint_path = "results/graph.chkp"
 
# Read data from checkpoint file
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
# Print tensor name and values
for key in var_to_shape_map:
    print("tensor_name: ", key)
    #print(reader.get_tensor(key))



