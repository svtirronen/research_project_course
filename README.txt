############# TRAINING THE MODEL ###################
You may run the training by command:

python train.py <access-type> <dummy>

Where access type is either "LA" or "PA" and dummy is either 1 or 0.
1 means that smaller dummy data set should be generated and used
in training. This makes it possible to get something running even
in local machine. Dummy is optional parameter. If one leaves it
out, real data is used.

################# DATA LOCATION ###################
ASVspoof_root folder should be located in the parent folder of this folder.
Not included in git. One can train the model even without it using
dummy data. However, it needs to be in right place in order
to do real feature extraction and then run the training with that.

############### FEATURE EXTRACTION ################
DataProcessor class in process_data.py has a method 
create_feature_file. It only needs access type ("LA" or "PA") 
as input and saves the features in TFRecord format by default.
These files are then used when training the model with real data.
