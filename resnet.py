import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2




class ResNetBlock(tf.keras.layers.Layer):
    def __init__(self,filters,conv_size):
        super(ResNetBlock, self).__init__()
        
        self.conv = [
            layers.Conv2D(filters,conv_size, activation='relu', padding='same'),
            layers.Conv2D(filters, conv_size, activation=None, padding='same')
        ]
        self.batch_norm = [
            layers.BatchNormalization(),
            layers.BatchNormalization()
        ]
        
        self.res = tf.keras.Sequential([
          layers.Add(),
          layers.Activation('relu')
        ])


        
    def call(self, input_data):
        x = self.conv[0](input_data)
        x = self.batch_norm[0](x)
        x = self.conv[1](x)
        x = self.batch_norm[1](x)
        x = self.res([x, input_data])
        return x       
        


class ResNet(tf.keras.layers.Layer):
    def __init__(self, num_resnet_blocks, layer_filters,pooling_size, conv_size, stride_size, output_dim_x):
        super(ResNet, self).__init__()
        self.conv_size = conv_size
        self.conv = [
            layers.Conv2D(layer_filters, conv_size,strides=stride_size, activation='relu',data_format="channels_last"),
            layers.Conv2D(layer_filters, conv_size,strides=stride_size, activation='relu',data_format="channels_last"),
            layers.Conv2D(layer_filters, conv_size, activation='relu',data_format="channels_last")
        ]
        self.max_pooling = layers.MaxPooling2D(pooling_size)   
        self.resnet_blocks = [
            ResNetBlock(filters=layer_filters, conv_size=conv_size) for _ in range(num_resnet_blocks)
        ]
        self.dense = tf.keras.layers.Dense(1)


        
    def call(self, input_data):
        x = tf.expand_dims(input_data,3)
        x = self.conv[0](x)
        x = self.conv[1](x)
        x = self.max_pooling(x)
        for block in self.resnet_blocks:
            x = block(x)
        x = self.conv[2](x)
        x = tf.reduce_mean(x,3)
        x = tf.linalg.matrix_transpose(x)
        return x

        

class TemporalFeatureExtractor(tf.keras.layers.Layer):
    def __init__(self,config):
        super(TemporalFeatureExtractor, self).__init__()

        #UNPACK THE CONFIG DICT
        num_resnet_blocks = config["num_resnet_blocks"]
        layer_filters = config["layer_filters"]
        pooling_size = config["pooling_size"]
        conv_size = config["conv_size"]
        stride_size = config["stride_size"]
        drop_out_rate = config["drop_out_rate"]
        output_dim_x = config["output_dim_x"]
        l2_rate = config["l2_rate"]
        self.standalone = bool(config["standalone"])
        self.input_x = config["input_dim_x"]
        self.input_y = config["input_dim_y"]


        #INIT THE RESNET
        self.resnet = ResNet(num_resnet_blocks, layer_filters,pooling_size, conv_size, stride_size, output_dim_x)

        #INIT THE FINAL LAYERS FOR MODELS 1,2,4,5
        if not self.standalone:
            self.dense = layers.Dense(output_dim_x, activation='relu', kernel_regularizer=l2(l2_rate), bias_regularizer=l2(l2_rate))
            self.dropout = layers.Dropout(drop_out_rate) 
    
        #FOR MODELS 3 and 6
        if self.standalone:
            self.flatten = layers.Flatten()
            self.softmax = layers.Dense(2, activation='softmax', kernel_regularizer=l2(l2_rate), bias_regularizer=l2(l2_rate))
            self.dropout = layers.Dropout(drop_out_rate) 


    def call(self, input_data):
        x = self.resnet(input_data)

        if not self.standalone: #MODELS 1,2,4,5
            x = self.dense(x)
            x = self.dropout(x)
        else: #MODELS 3 and 6
            x = self.flatten(x)
            x = self.dropout(x)
            x = self.softmax(x)

        return x


