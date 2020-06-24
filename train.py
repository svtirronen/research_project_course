from model_trainer import ModelTrainer
from evaluator import Evaluator
import tensorflow as tf
import sys

#################### SETTINGS #######################
length_norm_method = "pick"

####################################################




tf.debugging.set_log_device_placement(False)
# strategy = tf.distribute.MirroredStrategy()
# with strategy.scope():  
#  
########## GPU MEMORY PRE-ALLOCATION LIMITING ###########
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


print(sys.argv)

if sys.argv[-1] == "dummy":
    dummy = True
    sys.argv = sys.argv[:-1]

at = sys.argv[1]
model = sys.argv[2]
epoch = sys.argv[3] if len(sys.argv) == 4 else None


if "model" in model:
    #This is ment to be trained from scratch
    mt = ModelTrainer(model_type = model, dummy=dummy) 
 
elif len(model.split("-"))==2:
    #This is trained from the given model foldername
    e = Evaluator(at=at,epoch=epoch,model_folder=model,dummy=dummy)
    mt = e.mt



print(mt.model.summary()) 

# if dummy:
#      mt.dp.save_dummy_files(feature_type="raw",length_norm_method="pick",at=at)
#      mt.dp.save_dummy_files(feature_type="raw",length_norm_method="crop",at=at)
#      mt.dp.save_dummy_files(feature_type="fft",length_norm_method="pick",at=at)
#      mt.dp.save_dummy_files(feature_type="fft",length_norm_method="crop",at=at)

# print(f"Training {model}...")
# mt.run_training_process(at,length_norm_method, warm_start=False,verbose=1)



