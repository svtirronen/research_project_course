import numpy as np
import os
import pickle
import math
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import datetime as dt
from datetime import datetime, timedelta
from data_processor import DataProcessor
from resnet import TemporalFeatureExtractor
from encoder import Encoder
from gru import GRU
from datetime import datetime, timedelta


class ModelTrainer(object):

    def __init__(self, model_type=None, dummy=False, config_file=None):
        self.dummy = dummy
        #LOGGING FOLDERS
        self.log_dir = "tf_logs/"
        if not os.path.isdir(self.log_dir):
            os.mkdir(self.log_dir)
        self.cp_dir = "tf_models/"
        if not os.path.isdir(self.cp_dir):
            os.mkdir(self.cp_dir)
        self.config_dir = "model_configs/train/"
        if not os.path.isdir(self.config_dir):
            os.mkdir(self.config_dir)

        #IF THE CONFIG FILE WAS GIVEN AS INPUT
        #USES THAT
        if config_file!=None: 
            config_path = self.config_dir+config_file
            with open(config_path, "r") as f:
                config = json.load(f)
            self.warm_start = True
            
        #IF THE CONFIG FILE WAS NOT GIVEN AS INPUT
        #THE DEFAULT FILE WILL BE USED
        else:
            config_file = "main_config.json"
            config_path = f"{self.config_dir.split('/')[0]}/{config_file}"
            with open(config_path, "r") as f:
                config = json.load(f)        
    
            #Checks the model type input
            assert type(model_type) == str, f"Invalid model type:{model_type}"       
            assert True == (model_type in config.keys()), f"Given model type was not found in config file: {model_type}"
            
            #DUMMY vs REAL
            version_str = "dummy" if self.dummy else "real"
            config["data"] = config["data"][version_str]
            

            #DROP USELESS KEYS FROM THE DICT
            for key in list(config.keys()):
                if key not in ["data",model_type]:config.pop(key)

            self.warm_start = False

        #DATA PROCESSOR
        self.dp = DataProcessor(config["data"], self.dummy)

        #CREATE THE MODEL
        self.model = self.build_model(config)

        #STORE THE CONFIG DICT FOR LATER
        self.config = config


        
        
    ################### MODEL BUILDING #################

    def build_model(self,config):
        assert 2 == len(config.keys()), f"Invalid config dict: {config}"
        assert "data" in config.keys(), f"Invalid config dict: {config}"
        key_arr =np.array(list(config.keys()))
        model_key = key_arr[key_arr!="data"][0]

        #PICK THE DATA TYPE
        self.feature_type = config[model_key]["data_type"]

        #PICK THE DATA DIMS
        y_dim = config["data"][self.feature_type]["input_dim_y"]
        x_dim = config["data"][self.feature_type]["input_dim_x"]


        #BUILD THE MODELS
        if model_key in ["model1", "model4"]:
            
            #PREPARE THE RESNET CONFIG DICT
            resnet_dict = config[model_key]["resnet"]
            resnet_dict["input_dim_y"] = x_dim
            resnet_dict["input_dim_x"] = y_dim

            #INIT THE RESNET
            inputs = keras.Input(shape=(y_dim,x_dim))
            x = TemporalFeatureExtractor(resnet_dict)(inputs)

            #PREPARE ENCODER CONFIG DICT
            encoder_dict = config[model_key]["encoder"]
            encoder_dict["input_dim_x"] = x.shape[2]
            encoder_dict["input_dim_y"] = x.shape[1]

            outputs = Encoder(encoder_dict)(x, False, None)

        elif model_key in ["model2", "model5"]:
            
            #PREPARE THE RESNET CONFIG DICT
            resnet_dict = config[model_key]["resnet"]
            resnet_dict["input_dim_y"] = x_dim
            resnet_dict["input_dim_x"] = y_dim

            #PREPARE THE GRU CONGIF DICT
            gru_dict = config[model_key]["gru"]

            inputs = keras.Input(shape=(y_dim, x_dim))
            x = TemporalFeatureExtractor(resnet_dict)(inputs)
            outputs = GRU(gru_dict)(x)

        elif model_key in ["model3", "model6"]:
           
            #PREPARE THE RESNET CONFIG DICT
            resnet_dict = config[model_key]["resnet"]
            resnet_dict["input_dim_y"] = x_dim
            resnet_dict["input_dim_x"] = y_dim

            inputs = keras.Input(shape=(y_dim, x_dim))
            outputs = TemporalFeatureExtractor(resnet_dict)(inputs) 
             

        

        model = keras.Model(inputs, outputs)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['acc'])

        return model     



    ########################## TRAINING DATA ##################################
    def get_data(self,at,length_norm_method):
        self.ds_train, self.ds_dev = self.dp.data_pipeline(at=at, feature_type=self.feature_type, length_norm_method=length_norm_method)

    ############################ MODEL TRAINING ##################################


    def scheduler(self,epoch):
        base_val = 0.001 if not self.warm_start else 0.00001
        if epoch < 3:
            return base_val
        else:
            return (base_val * tf.math.exp(0.1 * (3-epoch))).numpy()

    def save_model_config(self):
        fn = datetime.now().strftime("%Y%m%d-%H%M%S")+".json"
        with open(f"{self.config_dir}{fn}", "w") as f:
            json.dump(self.config, f)


    def create_callback_list(self,at):
        #TENSORBOARD
        training_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir=self.log_dir + training_time
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir,
                                    write_images=True,
                                    write_graph=True,
                                    histogram_freq=1, 
                                    profile_batch = 100000000
                                    )
        #MODEL CHECKPOINTS
        cp_dir = f"{self.cp_dir}{at}"                
        if not os.path.isdir(cp_dir):
            os.mkdir(cp_dir)    
        cp_dir += "/"+training_time                 
        if not os.path.isdir(cp_dir):
            os.mkdir(cp_dir)
        cp_dir += "/weights.{epoch:04d}.ckpt"
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=cp_dir,
                                                        save_weights_only=True,
                                                        verbose=0,
                                                        save_freq='epoch')
        
        #LEARNING RATE SCHEDULER
        scheduler_callback = tf.keras.callbacks.LearningRateScheduler(self.scheduler)
        
        return [tensorboard_callback, cp_callback, scheduler_callback]


    def train_model(self,at, verbose=0):
        callbacks = self.create_callback_list(at)
       
        self.model.fit(self.ds_train, epochs=self.dp.epochs,
                steps_per_epoch=self.dp.steps_per_epoch,
                validation_steps = self.dp.validation_steps,
                validation_data=self.ds_dev,
                verbose=verbose,
                callbacks=callbacks)


    def run_training_process(self,at,length_norm_method, warm_start=False, verbose=0):
        self.get_data(at,length_norm_method)
        self.save_model_config()
        self.train_model(at,verbose=verbose)



