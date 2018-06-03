"""Test for version 2 of the zero_out op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

zero_out_op_2 = tf.load_op_library('./word2vec_ops.so')

with tf.Session(''):
     result = zero_out_op_2.zero_out2([[6, 5, 4], [3, 2, 1]])
     print (result.eval())
