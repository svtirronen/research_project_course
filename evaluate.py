from evaluator import Evaluator
import tensorflow as tf
import numpy as np
import sys


at = sys.argv[1]
mt = sys.argv[2]


if mt=="ensemble":
    f_list = ["20200215-113101_model1","20200216-132958_model2", "20200216-132834_model3"]
    Evaluator(at, ensemble=True).evaluate_ensemble(f_list)


else:
    models = {
        "model1":{
            "folder":{"LA":None, "PA":None},
            "epoch":{"LA":None, "PA":None},
            "lnm": {"LA":"crop", "PA":None} #Length norm method
        },
        "model2":{
            "folder":{"LA":None, "PA":None},
            "epoch":{"LA":None, "PA":None},
            "lnm": {"LA":"crop", "PA":None} #Length norm method
        },
        "model3":{
            "folder":{"LA":None, "PA":None},
            "epoch":{"LA":None, "PA":None},
            "lnm": {"LA":"crop", "PA":None} #Length norm method
        },
        "model4":{
            "folder":{"LA":None, "PA":None},
            "epoch":{"LA":None, "PA":None},
            "lnm": {"LA":"crop", "PA":None} #Length norm method
        },
        "model5":{
            "folder":{"LA":None, "PA":None},
            "epoch":{"LA":None, "PA":None},
            "lnm": {"LA":"crop", "PA":None} #Length norm method
        },
        "model6":{
            "folder":{"LA":None, "PA":None},
            "epoch":{"LA":None, "PA":None},
            "lnm": {"LA":"crop", "PA":None} #Length norm method
        }
    }
    epoch = models[mt]["epoch"][at]
    model_folder = models[mt]["folder"][at]
    lnm = models[mt]["lnm"][at]

    print(f"Evaluating {mt}...")
    Evaluator(at, epoch=epoch, model_folder = model_folder).run(lnm)
    print("Done!")
