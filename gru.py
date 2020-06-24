import numpy as np
import tensorflow as tf
from tensorflow.keras.regularizers import l2


class GRU(tf.keras.layers.Layer):

    def __init__(self, config):
        super(GRU, self).__init__()

        dense_layer_dim = config["dense_layer_dim"]
        dropout_rate = config["dropout_rate"]

        #Activation = tanh for gpu, but sigmoid makes the warning go away. The warning however is just about graph optimization, and the unoptimized graph is still a valid one.
        self.gru = tf.keras.layers.GRU(dense_layer_dim,activation="tanh")
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dense = tf.keras.layers.Dense(2, activation='softmax')

    def call(self, input_data):
        x = self.gru(input_data)
        output = self.dense(x)

        return output

