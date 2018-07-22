import tensorflow as tf

with tf.Graph().as_default():
    a = tf.constant([5, 3, 2, 7, 1, 4], dtype=tf.int32)
    b = tf.constant([4, 6, 3], dtype=tf.int32)
    a2 = tf.reshape(a, [6, 1])
    b2 = tf.reshape(b, [1, 3])
    result = tf.matmul(a2, b2)

    with tf.Session() as sess:
        print(result.eval())
