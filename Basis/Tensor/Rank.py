import tensorflow as tf

# Rank 0
mammal = tf.Variable('Elephant', tf.string)
ignition = tf.Variable(451, tf.int16)
floating = tf.Variable(3.14159265359, tf.float64)
its_complicated = tf.Variable(12.3 - 4.85j, tf.complex64)  # Converts two real numbers to a complex number.

# Rank 1
mystr = tf.Variable()
