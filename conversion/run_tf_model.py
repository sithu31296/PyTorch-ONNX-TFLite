import tensorflow as tf

tf_model_path = 'model_tf'

model = tf.saved_model.load(tf_model_path)
model.trainable = False

input_tensor = tf.random.uniform([1, 3, 640, 640])

out = model(**{'input': input_tensor})
print(out)