import tensorflow as tf
from tensorflow.keras.layers import Layer


class Residual(Layer):
    def __init__(self, fn, **kwargs):
        super().__init__(**kwargs)
        self.fn = fn

    def build(self, shape):
        init_gates = tf.zeros((shape[1:]), dtype=tf.float32)
        self.alpha = tf.Variable(init_gates, name='rezero_alpha')

    def call(self, inputs, **kwargs):
        return self.alpha * self.fn(inputs) + inputs
