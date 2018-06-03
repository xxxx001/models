
import tensorflow as tf
zero_out_module = tf.load_op_library('./word2vec_ops.so')
with tf.Session(''):
 c= zero_out_module.zero_out([[1, 2], [3, 4]]).eval()
 print (c)
