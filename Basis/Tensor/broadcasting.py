import tensorflow as tf

# Numpy广播见 learnTensorflow/numpy/broadcast
# 在数学中，您只能对形状相同的张量执行元素级运算（例如，相加和等于）。
# 不过，在 TensorFlow 中，您可以对张量执行传统意义上不可行的运算。
# TensorFlow 支持广播（一种借鉴自 Numpy 的概念）。
# 利用广播，元素级运算中的较小数组会增大到与较大数组具有相同的形状。例如，通过广播：
# 如果指令需要大小为 [6] 的张量，则大小为 [1] 或 [] 的张量可以作为运算数。
# 如果指令需要大小为 [4, 6] 的张量，则以下任何大小的张量都可以作为运算数。
#   ·[1, 6]
#   ·[6]
#   ·[]
# 如果指令需要大小为 [3, 5, 6] 的张量，则以下任何大小的张量都可以作为运算数。
#       [1, 5, 6]
#       [3, 1, 6]
#       [3, 5, 1]
#       [1, 1, 1]
#       [5, 6]
#       [1, 6]
#       [6]
#       [1]
#       []

with tf.Graph().as_default():
    # Create a six=element vector (1-D tensor)
    primes = tf.constant([2, 3, 5, 7, 11, 13], dtype=tf.int32)

    # Create a constant scalar with value1
    ones = tf.constant(1, dtype=tf.int32)

    # Add the two tensors. The resulting tensor is a six-element vector.
    just_beyond_primes = tf.add(primes, ones)

    with tf.Session() as sess:
        print(just_beyond_primes.eval())
