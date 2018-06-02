import tensorflow as tf 
print (tf.__version__)
hello = tf.constant('hello tensorflow!')
sess = tf.Session()
x = sess.run(hello)
print x
a = tf.constant(10)
b = tf.constant(32)
c = sess.run(a+b)
print c
sess.close()
