import tensorflow as tf
"""
线性回归程序的格式
"""

# Set up a linear classfier
classifier = tf.estimator.LinearClassifier()

# Train the model on some example data.
classifier.train(input_fn=train_input_fn, steps=2000)

# Use it to predict
predictions = classifier.predict(input_fn=predict_input_fn)
