import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('logs\\trains2\\model.ckpt-352000.meta')
    saver.restore(sess, "logs\\trains2\\model.ckpt-352000")

freeze_graph.freeze_graph(input_graph = "logs\\trains2\\model.pbtxt",  input_saver = "",
             input_binary = False, input_checkpoint = "logs\\trains2\\model.ckpt-352000", output_node_names = "y_",
             restore_op_name = "save/restore_all", filename_tensor_name = "save/Const:0",
             output_graph = "streetnumber.pb", clear_devices = True, initializer_nodes = "")

input_graph_def = tf.GraphDef()
with tf.gfile.Open(output_frozen_graph_name, "r") as f:
    data = f.read()
    input_graph_def.ParseFromString(data)

output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def,
        ["input"], 
        ["y_"],
        tf.float32.as_datatype_enum)

f = tf.gfile.FastGFile("optimized_streetnumber.pb", "w")
f.write(output_graph_def.SerializeToString())