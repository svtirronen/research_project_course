import numpy as np
import tensorflow as tf
from tensorflow.keras.regularizers import l2




############################# ATTENTION FUNCTION ###############################################
def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead) 
    but it must be broadcastable for addition.

    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable 
          to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
    output, attention_weights
    """
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  
    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
    return output, attention_weights





########################## MULTI-HEAD ATTENTION LAYER ##########################################
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, 
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights
    

############################### ENCODER LAYER ########################################
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim_x, num_heads, layer_dnn_dim, rate):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(input_dim_x, num_heads)

        #POINTWISE FFN
        self.ffn = tf.keras.Sequential([
        tf.keras.layers.Dense(layer_dnn_dim, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(input_dim_x)  # (batch_size, seq_len, d_model)
        ])

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)


    def call(self, x, training, mask):

        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2
    
############################### ENCODER  ########################################

class Encoder(tf.keras.layers.Layer):
    def __init__(self,config,
                 layer_dff=2048):
        super(Encoder, self).__init__()

        #UNPACK THE CONFIG DICT 
        num_layers = config["num_layers"]
        num_heads = config["num_heads"]
        drop_out_rate = config["drop_out_rate"]
        layer_dnn_dim = config["layer_dnn_dim"]

        self.input_dim_y = config["input_dim_y"]
        self.input_dim_x = config["input_dim_x"]


        self.pos_encoding = self.positional_encoding(self.input_dim_y,self.input_dim_x)
        self.enc_layers = [EncoderLayer(self.input_dim_x, num_heads, layer_dnn_dim, drop_out_rate)for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(drop_out_rate)
        
        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(1,kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(2, activation='softmax')
        ])

    def get_angles(self,pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self,position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)



    def call(self, x, training, mask):
        
        print("ENCODER STARTS")
        #Add position encoding.
        x *= tf.math.sqrt(tf.cast(self.input_dim_x, tf.float32))
        x += self.pos_encoding[:, :self.input_dim_y, :]
        #Dropout layer
        x = self.dropout(x, training=training)
        #Encoder layers
        for layer in self.enc_layers:
            x = layer(x, training, mask)
        x = self.dense(x)
        return x  # (batch_size, input_seq_len, d_model)

