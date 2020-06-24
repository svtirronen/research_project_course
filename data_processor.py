import numpy as np
import os
import pickle
import math
import tensorflow as tf
import soundfile as sf
import librosa
from scipy import signal
from datetime import datetime,timedelta


class DataProcessor(object):
    '''
    Basically just a wrapper for related functionality. 
    Workflow:
    1) Form feature files by running extract features
    2) Use to get data for training by running form_tf_batches

    '''

    def __init__(self, config, dummy):
        self.dummy = dummy
        self.features_root = "features"

        for i in range(4):
            d = "../" + i*"../"
            if "ASVspoof_root" in os.listdir(d):
                self.data_root = f"{d}ASVspoof_root"
                break


        self.proto_ext = {"train": "trn", "dev": "trl", "eval": "trl"}

        self.raw_feature_len = int(16000 * 2.4) #samplerate =16000
        #Spectrogram settings
        self.n_fft_samples = 4095 #FFT bins we use is 2048
        self.fft_window_ms = 50 #in milliseconds
        self.fft_shift_ms = 20 #in milliseconds

        #Data configs
        x_dim_fft = config["fft"]["input_dim_x"] #utterance length
        y_dim_fft = config["fft"]["input_dim_y"] #fft bins
        self.data_shape_fft = (y_dim_fft,x_dim_fft)

        x_dim_raw = config["raw"]["input_dim_x"] #utterance length
        y_dim_raw = config["raw"]["input_dim_y"] #generally this is 1
        self.data_shape_raw = (y_dim_raw,x_dim_raw)

        self.n_batch = config["n_batch"]
        self.epochs = config["epochs"]
    

    


    ################## FEATURE EXTRACTION #####################################
    def form_file_lists(self,protocol_file):
        '''
        Reads the given protocol file and forms two lists of file names: bonafide and spoof

        INPUTS:
        str: of the path to the file

        OUTPUTS:
        two lists of str file names

        '''
        f = open(protocol_file, "r")
        c = f.read()
        f.close()

        c = np.array(c.split("\n")) #Filter empty rows
        c = c[c!=""]

        arr = np.array([np.array(row.split(" "))[[1,-1]] for row in c])

        spoof_files = arr[:,0][arr[:,1]=="spoof"]
        bonafide_files = arr[:,0][arr[:,1]=="bonafide"]

        return bonafide_files, spoof_files
        


    def create_feature_file(self,feature_type,at,length_norm_method,st="train",verbose=False,shrink_eval=False):
        '''
        Saves the features to tf record file
        Does it for both bonafide and spoof separately

        INPUTS
        feature_type: type of the extracted feature
        at: access type (LA/PA)
        st: set type (train/dev/eval)
        verbose: bool to tell if progress information should be printed
        length_norm_method: str 'pick' or 'crop'
        '''
        assert True == (feature_type in ["raw","fft"]), f"Invalid feature_type parameter"
        assert True == (length_norm_method in ["pick", "crop"]), f"Invalid method: {method}"
        
        data_path = "/".join([self.data_root, f"{at}", f"ASVspoof2019_{at}_{st}", "flac/"])
        protocol_path = "/".join([self.data_root, f"{at}", f"ASVspoof2019_{at}_cm_protocols", f"ASVspoof2019.{at}.cm.{st}.{self.proto_ext[st]}.txt"])
        
        

        bonafide_files, spoof_files = self.form_file_lists(protocol_path)
        #Drop the half of the data to make it fit to memory
        if at=="PA" and shrink_eval:
            bonafide_files = bonafide_files[:len(bonafide_files)//2]
            spoof_files = spoof_files[:len(spoof_files)//2]

        
        for c, files in [("spoof", spoof_files), ("bonafide", bonafide_files)]:
            #if c=="spoof":
                #continue #Done already
            if verbose:
                print(f"Extracting {at} {st} {c}")
            
            path = f"{self.features_root}/{at}_{st}_{c}_{feature_type}_{length_norm_method}"

            tfr_path = f"{path}.tfrecord"
            with tf.io.TFRecordWriter(tfr_path) as writer:
                self.extract_features(feature_type,files, data_path, tfr_writer=writer,length_norm_method=length_norm_method) 

        
    
    def extract_features(self,feature_type,files,data_path,tfr_writer,length_norm_method):
        '''
        Extracts features from given set of audio filenames.
        Saves to a file either using pkl of tfr

        INPUTS:
        feature_type: type of the extracted feature
        files: a list of file names
        data_path: a path to the folder that contains the data
        tfr_writer: tf.io.TFRecordWriter
        length_norm_method: str

        '''
        assert True == (feature_type in ["raw","fft"]), f"Invalid feature_type parameter"
        assert True == (length_norm_method in ["pick", "crop"]), f"Invalid method: {method}"

        lens = []
        for fn in files:
            f_path = data_path + fn + ".flac"

            data, samplerate = sf.read(f_path) 

            if feature_type=="fft":
                n_window = int(self.fft_window_ms*(samplerate / 1000)) # samples in window
                n_shift = int(self.fft_shift_ms*(samplerate / 1000))
                n_overlap = n_window-n_shift

                ##MAGNITUDE WAS THE BEST IN THE EARLIER RESEARCH

                # f, t, Sxx = signal.spectrogram(data, fs=samplerate, noverlap=n_overlap, window =signal.hamming(n_window),nfft=n_samples,mode="psd")
                # f, t, Sxx = signal.spectrogram(data, fs=samplerate, noverlap=n_overlap, window =signal.hamming(n_window),nfft=n_samples,mode="phase")
                _, _, Sxx = signal.spectrogram(data, fs=samplerate, noverlap=n_overlap, window =signal.hamming(n_window),nfft=self.n_fft_samples,mode="magnitude")
                
                #Normalize the lengths
                data = self.length_normalization(Sxx, method=length_norm_method, n_tar=self.data_shape_fft[1])

            else:
                data = self.length_normalization(data.reshape((1,-1)),method=length_norm_method,n_tar=self.data_shape_raw[1])
            
            #Serialize
            example = self.serialize_example(data)
            tfr_writer.write(example) #Efficiently writing stuff after each iteration



    def length_normalization(self,arr,method,n_tar):
        '''
        Makes a random cropping or duplication of columns to match
        the n_tar in axis 1.
        
        If the method is pick: For each crop, takes two columns and changes that 
        to the mean of the two columns. For each duplication, takes two columns
        and adds their mean in between of them

        If the method is crop: for each crop, just randomly decides if it's
        cropped from the beginning or from the end. For each duplication,
        the array restarts from the beginning when it reaches the end.

        INPUTS:
        arr: assumed to be 2d array
        method: str 'pick' or 'crop'
        n_tar: int target length of array

        OUTPUTS:
        the modified arr
        '''

        assert True == (method in ["pick", "crop"]), f"Invalid method: {method}"
        n = arr.shape[1]
        n_diff = abs(n_tar - n)
        if n_diff==0:
            return arr

        if method == "pick":
            #Ind is a reversed sorted list of indices, so that the
            #operations can be done consecutively without worrying about
            #the fact that in each operation, the index changes for every
            #element that is located after the element in question
            
            if n > n_tar:
                ind = np.sort(np.random.choice(range(n), n_diff, replace=False))[::-1]
                for i in ind:
                    if i==0:
                        arr[:,1] = np.mean(arr[:,:2],axis=1)
                    else:
                        arr[:,i-1] = np.mean(arr[:,i-1:i+1],axis=1)                   
                    arr = np.delete(arr,i,axis=1)
            elif n < n_tar:
                m = math.floor(int(n_diff/n)) #m in case the n_diff is more than the length of array
                multi_ind = np.sort([i for j in range(m) for i in range(n)]).astype(int)
                pick_ind = np.sort(np.random.choice(range(n), n_diff-(m*n), replace=False))[::-1]
                ind = np.sort(np.concatenate([multi_ind,pick_ind]))[::-1]
                for i in ind:
                    if i == (n-1):
                        avg = arr[:,-1]
                    else:                  
                        avg = np.mean(arr[:,i:i+2],axis=1)
                    arr = np.insert(arr,i+1,avg, axis=1)
        else:
            
            if n > n_tar:
                r = np.random.randint(0,2) #either 1 or 0
                arr = arr[:,n_diff:] if r==0 else arr[:,:-n_diff]    

            else:
                
                mult = n_diff//n
                mults = [arr for _ in range(mult)]
                
                slice_ = [arr[:,:n_diff-mult*n]]
                concats = mults + slice_
                
                arr = np.concatenate([arr]+concats,axis=1)

        assert arr.shape[1] == n_tar, f"Invalid data dimensions: {arr.shape[1]} != {n_tar}"
        return arr


    def display_spectrogram(self,f,t,Sxx):
        plt.figure()
        plt.pcolormesh(t, f,10*np.log10(Sxx)) # dB spectrogram
        # plt.pcolormesh(t, f,Sxx) # Lineal spectrogram
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [seg]')
        plt.title('Spectrogram with scipy.signal',size=16)
        plt.show()


    ############# DUMMY FILES ######################

    def save_dummy_files(self,feature_type,length_norm_method,at="LA", even=False):
        assert True == (feature_type in ["raw","fft"]), f"Invalid feature type: {feature_type}"
        
        y,x = self.data_shape_fft if feature_type=="fft" else self.data_shape_raw

        n=10
        if even:
            n_spoof = n_bona = int(n/2)
        else:
            n_spoof = int(n*0.9)
            n_bona = n-n_spoof

        # #TRAIN DATA
        # arr_bonaX_train = np.array([[[int(str(1)+str(k)) for j in range(x)]for i in range(y)] for k in range(n_bona)])
        # arr_spoofX_train = np.array([[[int(str(2)+str(k)) for j in range(x)]for i in range(y)] for k in range(n_spoof)])
        # # #DEV DATA
        # arr_bonaX_dev = np.array([[[int(str(3)+str(k)) for j in range(x)]for i in range(y)] for k in range(n_bona)])
        # arr_spoofX_dev = np.array([[[int(str(4)+str(k)) for j in range(x)]for i in range(y)] for k in range(n_spoof)])
        # # #EVAL DATA
        # arr_bonaX_eval = np.array([[[int(str(5)+str(k)) for j in range(x)]for i in range(y)] for k in range(n_bona)])
        # arr_spoofX_eval = np.array([[[int(str(6)+str(k)) for j in range(x)]for i in range(y)] for k in range(n_spoof)])

        #TRAIN DATA
        arr_bonaX_train = np.array([[[1 for j in range(x)]for i in range(y)] for k in range(n_bona)])
        arr_spoofX_train = np.array([[[1 for j in range(x)]for i in range(y)] for k in range(n_spoof)])
        # #DEV DATA
        arr_bonaX_dev = np.array([[[1 for j in range(x)]for i in range(y)] for k in range(n_bona)])
        arr_spoofX_dev = np.array([[[1 for j in range(x)]for i in range(y)] for k in range(n_spoof)])
        # #EVAL DATA
        arr_bonaX_eval = np.array([[[1 for j in range(x)]for i in range(y)] for k in range(n_bona)])
        arr_spoofX_eval = np.array([[[1 for j in range(x)]for i in range(y)] for k in range(n_spoof)]) 

        #First, the bonafides
        
        path = f"features/dummy/{at}_train_bonafide_{feature_type}_{length_norm_method}.tfrecord"
        with tf.io.TFRecordWriter(path) as writer:
            for i in range(n_bona):
                example = self.serialize_example(arr_bonaX_train[i,:,:])
                writer.write(example)
        path = f"features/dummy/{at}_dev_bonafide_{feature_type}_{length_norm_method}.tfrecord"
        with tf.io.TFRecordWriter(path) as writer:
            for i in range(n_bona):
                example = self.serialize_example(arr_bonaX_dev[i,:,:])
                writer.write(example)
        path = f"features/dummy/{at}_eval_bonafide_{feature_type}_{length_norm_method}.tfrecord"
        with tf.io.TFRecordWriter(path) as writer:
            for i in range(n_bona):
                example = self.serialize_example(arr_bonaX_dev[i,:,:])
                writer.write(example) 
        
        #Then spoof
        path = f"features/dummy/{at}_train_spoof_{feature_type}_{length_norm_method}.tfrecord"
        with tf.io.TFRecordWriter(path) as writer:
            for i in range(n_spoof):
                example = self.serialize_example(arr_spoofX_train[i,:,:])
                writer.write(example)
        path = f"features/dummy/{at}_dev_spoof_{feature_type}_{length_norm_method}.tfrecord"
        with tf.io.TFRecordWriter(path) as writer:
            for i in range(n_spoof):
                example = self.serialize_example(arr_spoofX_dev[i,:,:])
                writer.write(example)
        path = f"features/dummy/{at}_eval_spoof_{feature_type}_{length_norm_method}.tfrecord"
        with tf.io.TFRecordWriter(path) as writer:
            for i in range(n_spoof):
                example = self.serialize_example(arr_spoofX_eval[i,:,:])
                writer.write(example) 

################# DATA PIPELINE ######################
    def parse_data(self,example_proto):
        '''
        Parses encoded examples
        '''
        # Parse the input `tf.Example` proto using the dictionary above.
        feature_description = {
            'data': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True)
            # 'fft': tf.io.FixedLenFeature([2048*120], tf.float32)
        }

        return tf.io.parse_single_example(example_proto, feature_description)
         
    def serialize_example(self, arr):
        '''
        Serializes input array using tf.Example
        '''
        feature = {"data":tf.train.Feature(float_list=tf.train.FloatList(value=arr.reshape(-1)))}
        proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return proto.SerializeToString() 


    def compute_data_statistics(self,at,ft,ds,n,norm_type):
        '''
        computes mean and standard deviation from ds
        '''
        assert True == (norm_type in ["row_wise","spectrogram_wise"])
        assert True == (ft in ["fft","raw"])
        ds_tmp = ds.map(lambda x,y: x, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        #FUNCTION FOR COMPUTING THE STATS WITH DS.REDUCE
        def mean_func(old_state,val):
            if norm_type == "row_wise":
                #Mean and std for each particular sample
                mean = tf.math.reduce_mean(val,axis=1) 
                std = tf.math.reduce_std(val,axis=1)
            else:
                #Mean and std for each particular sample
                mean = [tf.math.reduce_mean(val)]
                std = [tf.math.reduce_std(val)]
            new_state = old_state + tf.concat([mean,std], axis=0)
            return new_state

        #EXTRACTS THE STATS
        y_dim, x_dim = self.data_shape_fft if ft=="fft" else self.data_shape_raw
        if norm_type == "row_wise":
            init_state = tf.Variable([0.]*2*y_dim)
        else:
            init_state = tf.Variable([0.,0.])
        res = (ds_tmp.reduce(init_state, mean_func) / n).numpy()
        mean = res[:y_dim] if norm_type=="row_wise" else res[0]
        std = res[y_dim:] if norm_type=="row_wise" else res[1]

        #SAVES THE FILES
        if not os.path.isdir("stats"):
            os.mkdir("stats")
        now = datetime.now().strftime("%Y%m%d-%H%M%S")
        dir_path = f"stats/{now}"
        os.mkdir(dir_path) 

        postfix = norm_type.split("_")[0]
        mean_path = f"{dir_path}/{at}_mean_{postfix}.pkl"
        std_path = f"{dir_path}/{at}_std_{postfix}.pkl"

        with open(mean_path, "wb") as f_mean:
            pickle.dump(mean, f_mean)
        with open(std_path, "wb") as f_std:       
            pickle.dump(std, f_std)
        return mean_path, std_path


    def data_pipeline(self, at, feature_type, length_norm_method, norm_type="spectrogram_wise"):
        '''
        Just creates a data pipeline
        '''
        assert True == (norm_type in ["row_wise", "spectrogram_wise"])
        assert True == (feature_type in ["raw","fft"])

        placeholder = "dummy/" if self.dummy else ""
        train_spoof_path = f"{self.features_root}/{placeholder}{at}_train_spoof_{feature_type}_{length_norm_method}.tfrecord"
        train_bonafide_path = f"{self.features_root}/{placeholder}{at}_train_bonafide_{feature_type}_{length_norm_method}.tfrecord"
        dev_spoof_path = f"{self.features_root}/{placeholder}{at}_dev_spoof_{feature_type}_{length_norm_method}.tfrecord"
        dev_bonafide_path = f"{self.features_root}/{placeholder}{at}_dev_bonafide_{feature_type}_{length_norm_method}.tfrecord"
        eval_spoof_path = f"{self.features_root}/{placeholder}{at}_eval_spoof_{feature_type}_{length_norm_method}.tfrecord"
        eval_bonafide_path = f"{self.features_root}/{placeholder}{at}_eval_bonafide_{feature_type}_{length_norm_method}.tfrecord"



        #GET THE DATA
        train_spoof = tf.data.TFRecordDataset(train_spoof_path).map(self.parse_data,num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_bonafide = tf.data.TFRecordDataset(train_bonafide_path).map(self.parse_data,num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dev_spoof = tf.data.TFRecordDataset(dev_spoof_path).map(self.parse_data,num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dev_bonafide = tf.data.TFRecordDataset(dev_bonafide_path).map(self.parse_data,num_parallel_calls=tf.data.experimental.AUTOTUNE)
        eval_spoof = tf.data.TFRecordDataset(eval_spoof_path).map(self.parse_data,num_parallel_calls=tf.data.experimental.AUTOTUNE)
        eval_bonafide = tf.data.TFRecordDataset(eval_bonafide_path).map(self.parse_data,num_parallel_calls=tf.data.experimental.AUTOTUNE)

        #RESHAPE
        y_dim, x_dim = self.data_shape_fft if feature_type == "fft" else self.data_shape_raw
        train_spoof = train_spoof.map(lambda x: tf.reshape(x["data"], (y_dim,x_dim)),num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_bonafide = train_bonafide.map(lambda x: tf.reshape(x["data"], (y_dim,x_dim)),num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dev_spoof = dev_spoof.map(lambda x: tf.reshape(x["data"], (y_dim,x_dim)),num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dev_bonafide = dev_bonafide.map(lambda x: tf.reshape(x["data"], (y_dim,x_dim)),num_parallel_calls=tf.data.experimental.AUTOTUNE)       
        eval_spoof = eval_spoof.map(lambda x: tf.reshape(x["data"], (y_dim,x_dim)),num_parallel_calls=tf.data.experimental.AUTOTUNE)
        eval_bonafide = eval_bonafide.map(lambda x: tf.reshape(x["data"], (y_dim,x_dim)),num_parallel_calls=tf.data.experimental.AUTOTUNE)  


        #COMBINE TRAIN AND DEV, later picks part of dev and includes it to train
        spoof = train_spoof.concatenate(dev_spoof)
        bonafide = train_bonafide.concatenate(dev_bonafide)

        #ADD THE LABELS
        spoof = spoof.map(lambda x: (x,tf.constant(1)),num_parallel_calls=tf.data.experimental.AUTOTUNE)
        bonafide = bonafide.map(lambda x: (x,tf.constant(0)),num_parallel_calls=tf.data.experimental.AUTOTUNE)   

        #GET EVAL ADDITION READY
        n_eval_addition = 10000
        a = int(n_eval_addition*0.9)
        b = n_eval_addition - a
        eval_spoof = eval_spoof.map(lambda x: (x,tf.constant(1)),num_parallel_calls=tf.data.experimental.AUTOTUNE).take(a)
        eval_bonafide = eval_bonafide.map(lambda x: (x,tf.constant(0)),num_parallel_calls=tf.data.experimental.AUTOTUNE).take(b)
        eval_set = eval_spoof.concatenate(eval_bonafide)
        

        #DO TRAIN TEST SPLIT FOR SPOOF AND BONAFIDE SEPARATELY
        train_test_ratio = 0.8
        n_spoof = spoof.reduce(0, lambda x, _: x + 1).numpy()
        n_train_spoof = int(n_spoof * train_test_ratio)
        n_dev_spoof = n_spoof - n_train_spoof

        train_spoof = spoof.take(n_train_spoof)
        dev_spoof = spoof.skip(n_train_spoof)
     
        n_bonafide = bonafide.reduce(0, lambda x, _: x + 1).numpy()
        n_train_bonafide = int(n_bonafide * train_test_ratio)
        n_dev_bonafide = n_bonafide - n_train_bonafide
        

        train_bonafide = bonafide.take(n_train_bonafide)
        dev_bonafide = bonafide.skip(n_train_bonafide)

        #OVERSAMPLE TRAIN BONAFIDE
        oversampling_rate = int(n_spoof / n_bonafide)
        bona_overs = train_bonafide.flat_map(lambda x, _: tf.data.Dataset.from_tensors(x).repeat(oversampling_rate))     
        #needs to add the labels again, since the previous line drops them. feel free to mod
        bona_overs = bona_overs.map(lambda x: (x,tf.constant(0)),num_parallel_calls=tf.data.experimental.AUTOTUNE)   
        
        #COMBINE BONAFIDE AND SPOOF       
        ds_train = train_spoof.concatenate(bona_overs).shuffle(100000, seed=1234)
        ds_dev = dev_spoof.concatenate(dev_bonafide)

        #ADD SOME PART OF EVAL SET INTO DEV SET
        ds_dev = eval_set.concatenate(ds_dev).shuffle(100000, seed=1234)

        #FORM A 'REAL' (not oversampled) VERSION OF TEST SET FOR COMPUTING THE STATS FROM IT
        stats_set = train_spoof.concatenate(train_bonafide)

        #COMPUTE THE STATS FROM THE REAL (not oversampled) TRAINING SET
        n_train = n_train_spoof + n_train_bonafide


        mean_path, std_path = self.compute_data_statistics(at,feature_type, stats_set, n=n_train, norm_type=norm_type)
        

        #NORMALIZE OVERSAMPLED TRAIN SET AND NON-OVERSAMPLED TEST SET
        with open(mean_path, "rb") as f_mean:
            mean_arr = pickle.load(f_mean)
        with open(std_path, "rb") as f_std:
            std_arr = pickle.load(f_std)

        if norm_type == "row_wise":  
            mean_tf = tf.constant(mean_arr.reshape(y_dim,1), dtype=tf.float32)
            std_tf = tf.constant(std_arr.reshape(y_dim,1), dtype=tf.float32) + tf.constant(1e-9, dtype=tf.float32)#+epsilon                  
        elif norm_type == "spectrogram_wise":
            mean_tf = tf.constant(mean_arr, dtype=tf.float32)
            std_tf = tf.constant(std_arr, dtype=tf.float32)  + tf.constant(1e-9, dtype=tf.float32)#+epsilon  
        
        ds_train = ds_train.map(lambda x,y: ((x-mean_tf)/std_tf, y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_dev = ds_dev.map(lambda x,y: ((x-mean_tf)/std_tf, y), num_parallel_calls=tf.data.experimental.AUTOTUNE)

        #ONLY TAKE N (to speed up initial experiments on PA)
        # N_train = 50000
        # N_dev = 50000
        # ds_train = ds_train.take(N_train)
        # ds_dev = ds_dev.take(N_dev)

        #CACHE, REPEAT, BATCH 
        ds_train = ds_train.cache().repeat().batch(batch_size=self.n_batch)
        ds_dev = ds_dev.cache().repeat().batch(batch_size=self.n_batch)

        #PREFETCH
        ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
        ds_dev = ds_dev.prefetch(tf.data.experimental.AUTOTUNE)

        #FINALLY SET SOME VARIABLES
        n_train = n_train_spoof + (n_train_bonafide * oversampling_rate)
        n_dev = n_dev_bonafide + n_dev_spoof + n_eval_addition
        self.steps_per_epoch = math.ceil(n_train / self.n_batch)
        self.validation_steps = math.ceil(n_dev / self.n_batch)

        return ds_train, ds_dev

    def eval_pipeline(self,at,feature_type,length_norm_method,stats_folder):
        '''
        Builds a data pipeline to be used in evaluation 

        INPUTS:
        at: access type
        stats_folder: path to the folder that contains the stats for normalization

        OUTPUTS:
        eval_bonafide: dataset
        eval_spoof: dataset

        '''
        assert True == (feature_type in ["fft","raw"])
        placeholder = "dummy/" if self.dummy else ""
        
        
        spoof_path = f"{self.features_root}/{placeholder}{at}_eval_spoof_{feature_type}_{length_norm_method}.tfrecord"
        bonafide_path = f"{self.features_root}/{placeholder}{at}_eval_bonafide_{feature_type}_{length_norm_method}.tfrecord"

        #GET THE SPECTROGRAMS
        spoof = tf.data.TFRecordDataset(spoof_path).map(self.parse_data,num_parallel_calls=tf.data.experimental.AUTOTUNE)
        bonafide = tf.data.TFRecordDataset(bonafide_path).map(self.parse_data,num_parallel_calls=tf.data.experimental.AUTOTUNE)

        #RESHAPE
        y_dim, x_dim = self.data_shape_fft if feature_type=="fft" else self.data_shape_raw
        spoof = spoof.map(lambda x: tf.reshape(x["data"], (y_dim,x_dim)),num_parallel_calls=tf.data.experimental.AUTOTUNE)
        bonafide = bonafide.map(lambda x: tf.reshape(x["data"], (y_dim,x_dim)),num_parallel_calls=tf.data.experimental.AUTOTUNE)


        #NORMALIZE THE DATA SETS
        f_list = os.listdir(stats_folder)
        mean_fns = [fn for fn in f_list if "mean" in fn]
        std_fns = [fn for fn in f_list if "std" in fn]
        assert len(std_fns)==1, f"Incorrect amount of std files: {len(std_fns)} in folder {stats_folder}"
        assert len(std_fns)==1, f"Incorrect amount of mean files: {len(mean_fns)} in folder {stats_folder}"
        mean_path = stats_folder + mean_fns[0]
        std_path = stats_folder + std_fns[0]


        mean_postfix = mean_path.split("_")[-1].split(".")[0]
        std_postfix = std_path.split("_")[-1].split(".")[0]
        assert mean_postfix==std_postfix, f"Problem in data statistic file names: {mean_path}, {std_path}"
        assert True ==(mean_postfix in ["row","spectrogram"]), f"Problem in data statistic file names: {mean_postfix}"
        postfix = mean_postfix 

        with open(mean_path, "rb") as f_mean:
            mean_arr = pickle.load(f_mean)
        with open(std_path, "rb") as f_std:
            std_arr = pickle.load(f_std)

        if postfix == "row":  
            mean_tf = tf.constant(mean_arr.reshape(y_dim,1), dtype=tf.float32)
            std_tf = tf.constant(std_arr.reshape(y_dim,1), dtype=tf.float32) + tf.constant(1e-9, dtype=tf.float32)#+epsilon                  
        elif postfix == "spectrogram":
            mean_tf = tf.constant(mean_arr, dtype=tf.float32)
            std_tf = tf.constant(std_arr, dtype=tf.float32)  + tf.constant(1e-9, dtype=tf.float32)#+epsilon  
        

        spoof = spoof.map(lambda x: (x-mean_tf)/std_tf, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        bonafide = bonafide.map(lambda x: (x-mean_tf)/std_tf, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        #BATCH
        spoof = spoof.batch(batch_size=self.n_batch)
        bonafide = bonafide.batch(batch_size=self.n_batch)


        return spoof, bonafide


