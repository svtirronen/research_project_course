import numpy as np
import os
from datetime import datetime, timedelta
from model_trainer import ModelTrainer
import tensorflow as tf
import json
import eval_metrics as em
import pickle


class Evaluator(object):

    def __init__(self, at, epoch=None,model_folder=None, dummy=False, ensemble=False):
        '''
        INPUTS:
        If model folder is not given, uses the latest one

        at: str access type
        epoch: int
        use_latest: bool
        model_folder: str
        '''

        self.p_spoof = 0.05 #Prior of spoof
        self.at = at
        if ensemble:
            return
        else:
            assert at!=None, f"Invalid access type parameter: {at}"
            assert epoch!=None, f"Invalid epoch number: {epoch}"

        self.models_root = f"tf_models/{at}/"
        self.configs_root = f"model_configs/train/"
        self.stats_root = f"stats/"
        #Use only the dev file
        self.asv_score_file = f"scores/ASV/{at}/ASVspoof2019.{at}.asv.dev.gi.trl.scores.txt"
        
        #DEFINE THE FOLDER
        if type(model_folder)==str:
            #Check the input
            err_str = f"Invalid argument for model_folder: {model_folder}, {type(model_folder)}"
            assert type(model_folder)==str, err_str
            assert True==(model_folder in os.listdir(self.models_root)), err_str
            self.model_folder = model_folder
            print(f"Loading the model from folder {self.model_folder}")

        else:
            now = datetime.now()
            self.model_folder = self.find_closest_model(now) 
            print("Model folder was not given.")
            print(f"Loading the latest model from folder {self.model_folder}")

        #SET THE CM SCORE FILE PATH
        self.cm_score_file = f"scores/CM/{at}/{self.model_folder}.txt"



        #FIND THE CORRECT CONFIG FILE
        config = self.find_closest_config_file(self.model_folder)
        print(f"Using the closest config file: {config}")

        #INIT THE CORRECT KIND OF MODEL
        self.mt = ModelTrainer(config_file=config,dummy=dummy)
        #STORE THE MODEL AND DATA PROCESSOR
        self.model = self.mt.model
        self.dp = self.mt.dp

        #NEEDS TO TRAIN WITH ONE DUMMY EXAMPLE TO INITIALIZE EVERYTHING
        #otherwise the chekpoints are not loaded completely
        #it's a bug in tf source code, related to eager exec
        with open(self.configs_root+config, "r") as f:
            config_dict = json.load(f)
        #find data type
        for key in config_dict.keys():
            if "model" in key:
                self.feature_type = config_dict[key]["data_type"]
                y_dim = config_dict["data"][self.feature_type]["input_dim_y"]
                x_dim = config_dict["data"][self.feature_type]["input_dim_x"]
                break

        x = [[[1 for j in range(x_dim)] for i in range(y_dim)]]
        x = tf.constant(x)
        y = tf.constant([1])
        self.model.fit(x, y, epochs=1,verbose=False)

        #LOAD THE CORRECT WEIGHTS FOR THE MODEL
        epoch = str(epoch)
        while len(epoch)<4:
            epoch = "0"+epoch
        cp_path = f"{self.mt.cp_dir}{at}/{self.model_folder}/weights.{epoch}.ckpt"
        self.model.load_weights(cp_path)



    def find_closest_config_file(self,model_fn):
        '''
        Finds the config file from self.configs_root that is closest
        in time to the model_fn.

        INPUTS:
        model_fn: str of the folder name

        OUTPUTS:
        str: filename of the closest config.
        '''

        model_dt = datetime.strptime(model_fn, "%Y%m%d-%H%M%S")

        #FINDS THE CLOSEST MATCHING CONFIG FOR THE MODEL 
        dates = [datetime.strptime(fn.split(".")[0], "%Y%m%d-%H%M%S") for fn in os.listdir(self.configs_root)]
        closest_config_dt = min(dates, key=lambda x: abs(x - model_dt))

        #FINDS THE CLOSEST MATCHING MODEL FOR THE CONFIG
        closest_model_fn = self.find_closest_model(closest_config_dt)

        #IF THEY AGREE, IT CAN BE RETURNED
        assert closest_model_fn == model_fn, f"Unable to unambiguously define the closest config. Please check manually. Model folder: {closest_model_fn}, config time: {closest_config_dt}"
        closest_config_fn = closest_config_dt.strftime("%Y%m%d-%H%M%S")+".json"
        assert True == (closest_config_fn in os.listdir(self.configs_root)), f"Unable to find json file: {closest_config_fn}"
        
        return closest_config_fn


    def find_closest_model(self, dtime):
        '''
        Finds the model in the model folder that is closest to the given time.

        INPUTS:
        dtime: datetime.datetime

        OUTPUTS:
        str: name of the folder
        '''

        #FINDS THE ONE THAT IS CLOSEST TO NOW
        dates = [datetime.strptime(fn, "%Y%m%d-%H%M%S") for fn in os.listdir(self.models_root)]
        latest = min(dates, key=lambda x: abs(x - dtime))    
        return latest.strftime("%Y%m%d-%H%M%S")

    def find_closest_stats_folder(self, model_fn):
        '''
        Finds the folder in the stats folder that is closest to the given time.

        INPUTS:
        dtime: datetime.datetime

        OUTPUTS:
        str: name of the folder
        '''
        model_dt = datetime.strptime(model_fn, "%Y%m%d-%H%M%S")
        #FINDS THE ONE THAT IS CLOSEST TO NOW
        dates = [datetime.strptime(fn, "%Y%m%d-%H%M%S") for fn in os.listdir(self.stats_root)]
        latest = min(dates, key=lambda x: abs(x - model_dt))    
        return latest.strftime("%Y%m%d-%H%M%S")               
            

    def compute_cm_scores(self,length_norm_method=None, p_spoof=None, ensemble_ind=None):
        '''

        Computes the cm scores for the self.model
        '''
        if p_spoof==None: #GENERAL CASE
            closest_stats_folder = self.find_closest_stats_folder(self.model_folder)
            stats_folder = f"{self.stats_root}{closest_stats_folder}/"
            spoof, bonafide = self.dp.eval_pipeline(self.at, self.feature_type, length_norm_method, stats_folder)
            is_ensemble = False

        else:
            n_bona = p_spoof[1]
            arr = np.array(p_spoof[0])
            bonafide = arr[:n_bona].reshape(-1,1)
            spoof = arr[n_bona:].reshape(-1,1)
            is_ensemble = True
            

        p_spoof = self.p_spoof #prior for spoof
        p_bona = 1-p_spoof #prior for bonafide

        #FOR ACCURACY COMPUTING
        bona_correct = 0
        bona_fail = 0
        spoof_correct = 0
        spoof_fail = 0


        #FOR STORING PREDICTION PROBABILITIES
        p_spoof_list = []
        n_bona = 0
        with open(self.cm_score_file, "w") as f:
            #First bonafide
            for batch in bonafide:
                preds = self.model.predict(batch) if not is_ensemble else batch
                for pred in preds:
                    p_bona_x = pred[0] if not is_ensemble else 1-pred #p(bona|x)
                    p_spoof_x = pred[1] if not is_ensemble else pred   #p(spoof|x)  
                    p_spoof_list.append(p_spoof_x)
            
                    LLR = np.log(p_bona_x * p_spoof) - np.log(p_spoof_x * p_bona)#LLR = log(p(x|bona)) - log(p(x|spoof)) = log(p(bona|x)p(spoof)) - log(p(spoof|x)p(bona))
                    
                    s = f"- - bonafide {LLR}\n"
                    f.write(s)

                    c = np.argmax(pred) if not is_ensemble else int(round(p_spoof_x))
                    if c==0 :
                        bona_correct+=1
                    else: 
                        bona_fail+=1
                    n_bona +=1


            #Then spoof
            for batch in spoof:
                preds = self.model.predict(batch) if not is_ensemble else batch
                for pred in preds:
                    p_bona_x = pred[0] if not is_ensemble else 1-pred   #p(bona|x)
                    p_spoof_x = pred[1] if not is_ensemble else pred   #p(spoof|x)
                    p_spoof_list.append(p_spoof_x)


                    LLR = np.log(p_bona_x * p_spoof) - np.log(p_spoof_x * p_bona) #LLR = log(p(x|bona)) - log(p(x|spoof)) = log(p(bona|x)p(spoof)) - log(p(spoof|x)p(bona))
                    # print(pred)
                    s = f"- - spoof {LLR}\n"
                    f.write(s)

                    c = np.argmax(pred) if not is_ensemble else int(round(p_spoof_x))
                    if c==1 :
                        spoof_correct+=1
                    else: 
                        spoof_fail+=1
        #STORES THE PREDICTED PROBABILITIES FOR ENSEMBLE MODEL PURPOSES
        if not is_ensemble:
            modelkeys = [key for key in self.mt.config.keys() if "model" in key]
            assert len(modelkeys) == 1, f"Invalid model dictionary: {modelkeys}"
            mk = modelkeys[0]
            f_path = f"scores/p_spoof/{self.at}/{self.model_folder}_{mk}.pkl"
        else:
            mk = f"ensemble_{ensemble_ind}"
            f_path = f"scores/p_spoof/{self.at}/{mk}.pkl"
        with open(f_path, "wb") as f:
            pickle.dump((p_spoof_list, n_bona), f)
            
            

        #ALSO RETURNS THE CLASSIFICATION ACCURACIES
        bona_acc = bona_correct / (bona_correct + bona_fail)
        spoof_acc = spoof_correct / (spoof_correct + spoof_fail)
        total_acc = (bona_correct + spoof_correct) / (bona_correct + bona_fail+ spoof_correct + spoof_fail)
        return bona_acc, spoof_acc, total_acc

    def compute_tDCF(self):
        '''
        Computes the t-DCF score for the model assigned to self.model.
        Assumes that the cm score file is created for the model.
        '''

        # Fix tandem detection cost function (t-DCF) parameters
        cost_model = {
            'Pspoof': self.p_spoof,  # Prior probability of a spoofing attack
            'Ptar': (1 - self.p_spoof) * 0.99,  # Prior probability of target speaker
            'Pnon': (1 - self.p_spoof) * 0.01,  # Prior probability of nontarget speaker
            'Cmiss_asv': 1,  # Cost of ASV system falsely rejecting target speaker
            'Cfa_asv': 10,  # Cost of ASV system falsely accepting nontarget speaker
            'Cmiss_cm': 1,  # Cost of CM system falsely rejecting target speaker
            'Cfa_cm': 10,  # Cost of CM system falsely accepting spoof
        }

        # Load organizers' ASV scores
        asv_data = np.genfromtxt(self.asv_score_file, dtype=str)
        asv_sources = asv_data[:, 0]
        asv_keys = asv_data[:, 1]
        asv_scores = asv_data[:, 2].astype(np.float)

        # Load CM scores
        cm_data = np.genfromtxt(self.cm_score_file, dtype=str)
        cm_utt_id = cm_data[:, 0]
        cm_sources = cm_data[:, 1]
        cm_keys = cm_data[:, 2]
        cm_scores = cm_data[:, 3].astype(np.float)

        #Modify CM not to contain inf or -inf values. They appear if the model is certain about some result
        if(cm_scores[cm_scores!=float("inf")] != -float("inf")).any():
            real_max = cm_scores[cm_scores != float("inf")].max()
            real_min = cm_scores[cm_scores != -float("inf")].min()
            max_abs = min(20, max(abs(real_max), abs(real_min)))
        else:
            max_abs = 20
        cm_scores[cm_scores == float("inf")] = max_abs
        cm_scores[cm_scores == -float("inf")] = -max_abs
        

        # Extract target, nontarget, and spoof scores from the ASV scores
        tar_asv = asv_scores[asv_keys == 'target']
        non_asv = asv_scores[asv_keys == 'nontarget']
        spoof_asv = asv_scores[asv_keys == 'spoof']

        # Extract bona fide (real human) and spoof scores from the CM scores
        bona_cm = cm_scores[cm_keys == 'bonafide']
        spoof_cm = cm_scores[cm_keys == 'spoof']

        # EERs of the standalone systems and fix ASV operating point to EER threshold
        eer_asv, asv_threshold = em.compute_eer(tar_asv, non_asv)
        eer_cm = em.compute_eer(bona_cm, spoof_cm)[0]


        [Pfa_asv, Pmiss_asv, Pmiss_spoof_asv] = em.obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_threshold)


        # Compute t-DCF
        tDCF_curve, CM_thresholds = em.compute_tDCF(bona_cm, spoof_cm, Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, cost_model, False)

        # Minimum t-DCF
        min_tDCF_index = np.argmin(tDCF_curve)
        min_tDCF = tDCF_curve[min_tDCF_index]

        return min_tDCF, eer_cm


    def run(self,length_norm_method):
        ''' 
        Computes the cm scores and t-dcf

        OUTPUTS:
        min_tDCF: float
        '''
        accs = self.compute_cm_scores(length_norm_method)
        min_tDCF, eer_cm = self.compute_tDCF()

        #SAVE RESULTS TO A FILE
        fpath = f"scores/final/{self.at}/{self.model_folder}.txt"
        with open(fpath, "w") as f:
            f.write(f"min_tDCF: {min_tDCF}\neer(%): {eer_cm*100}\n")

            for label,acc in list(zip(["bonafide","spoof","total"],accs)):
                f.write(f"{label} accuracy: {acc}\n")
    
    def evaluate_ensemble(self,f_list):
        '''
        Just uses the pickled lists of p_sppf_x values within scores/p_spoof/
        Each of those files shoulf represent the predictions of different models over the same data set
        in the same order.
        '''

        p_spoof = []
        n_bona = []
        n_total = []
        for f_path in f_list:
            with open(f"scores/p_spoof/{self.at}/{f_path}.pkl", "rb") as f:
                obj = pickle.load(f)
                arr = np.array(obj[0])
                p_spoof.append(arr)
                n_bona.append(obj[1])
                n_total.append(arr.shape[0])
        assert len(np.unique(n_bona)) == 1, f"Amounts not matching."
        assert len(np.unique(n_total)) == 1, f"Amounts not matching."
        n_bona = n_bona[0]
        n = n_total[0]

        p_spoof = np.array(p_spoof)
        p_bona = 1-p_spoof

        votes_spoof = (p_spoof>0.5).sum(axis=0)
        votes_bona = (p_bona>0.5).sum(axis=0)


        tied = votes_bona==votes_spoof
        spoofs = votes_bona<votes_spoof
        bonas = votes_bona>votes_spoof

        max_p_bona = p_bona.max(axis=0)
        max_p_spoof = p_spoof.max(axis=0)
        bona_greater = max_p_bona > max_p_spoof

        res_p_spoof = []
        for i in range(n):
            if tied[i]:
                if bona_greater[i]:                 
                    p_spoof = 1-max_p_bona[i]
                else:
                    p_spoof = max_p_spoof[i]            
            elif spoofs[i]:
                p_spoof = max_p_spoof[i] 
            elif bonas[i]:
                p_spoof = 1-max_p_bona[i]
            else:
                raise ValueError("Invalid voting.")
            res_p_spoof.append(p_spoof)   

        #DEFINE THE INDEX OF THIS NEW ENSEMBLE
        prior_ensembles = [fn for fn in os.listdir(f"scores/final/{self.at}") if "ensemble" in fn]
        if len(prior_ensembles) == 0:
            ind = 1
        else:
            ind = 1 +np.max([int(en.split("_")[1].split(".")[0]) for en in prior_ensembles])
        f_name = f"ensemble_{ind}"

        #COMPUTE THE CM SCORES AND tDCF FOR THE ENSEMBLE MODEL
        self.asv_score_file = f"scores/ASV/{self.at}/ASVspoof2019.{self.at}.asv.dev.gi.trl.scores.txt"
        self.cm_score_file = f"scores/CM/{self.at}/{f_name}.txt"
        accs = self.compute_cm_scores(p_spoof = (res_p_spoof,n_bona), ensemble_ind=ind)
        min_tDCF, eer_cm = self.compute_tDCF()

        #SAVE RESULTS TO A FILE        
        fpath = f"scores/final/{self.at}/{f_name}.txt"
        with open(fpath, "w") as f:
            f.write(f"Ensemble model with the models: {f_list}\n\n")
            f.write(f"min_tDCF: {min_tDCF}\neer(%): {eer_cm*100}\n")

            for label,acc in list(zip(["bonafide","spoof","total"],accs)):
                f.write(f"{label} accuracy: {acc}\n")
                
