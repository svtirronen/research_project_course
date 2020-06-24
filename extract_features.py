from data_processor import DataProcessor
import json


#####################################
ft = "fft" #in (fft,raw)
shrink_eval = True

#####################################



with open("model_configs/main_config.json", "r") as f:
    config = json.load(f)["data"]["real"]

dp = DataProcessor(config,dummy=False)

#RUN THE GIVEN TYPE OF DATA EXTRACTION FOR ALL DATA SETS
sts = ["eval"]
ats = ["PA"] #LA, PA
lnts = ["pick"] #pick, crop
print(f"Starting feature extraction for ft:{ft}.")
for at in ats:
    for st in sts:
        for lnt in lnts:
            print(f"Extracting {at} {st} {lnt}...")
            dp.create_feature_file(ft,at,lnt,st,shrink_eval=shrink_eval)
            print("Done!")
