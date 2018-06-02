import tensorflow as tf

word2vec = tf.load_op_library('./word2vec_ops.so')

with tf.Session(''):
   xx= word2vec.skipgram_word2vec(filename='text8', batch_size=500, window_size=5,min_count=5, subsample=0.001)
   print xx
