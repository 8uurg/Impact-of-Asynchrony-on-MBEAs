#  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
# 
# This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
# 
# Project leaders: Peter A.N. Bosman, Tanja Alderliesten
# Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
# Main code developer: Arthur Guijt

# Convert models from whatever version by loading & praying.
# Annoyingly enough: they have saved their models by pickling
# not using the xgboost model save function, likely because
# this function is a relatively new addition.
# In any case, we are only using these models for inference,
# so I do wonder whether doing this changes anything apart
# from reducing the number of warnings shown at the beginning.
import xgboost
import pickle
import shutil
from pathlib import Path

directory = Path("./nb_models_0.9/xgb_v0.9/")

print(f"directory: {directory}")
model_paths = directory.glob("*/xgb/*/surrogate_model.model")

for model_path in model_paths:
    print(f"Upgrading {model_path}")
    # corresponding paths
    backup_path = model_path.with_suffix(".model.bak")
    serialized_path = model_path.with_suffix(".ubj")
    # make backup if none exists already
    if not backup_path.exists():
        shutil.copy(model_path, backup_path)
    # load model from pickle (should incur a warning), may fail if the format has changed
    # and things are no longer backwards compatible.
    model = pickle.load(open(model_path, 'rb'))
    # write it using the new fancy binary ubj format
    model.save_model(serialized_path)
    # load it again, this time using a clean booster
    model = xgboost.core.Booster()
    model.load_model(serialized_path)
    # save the resulting model as a pickle again
    # (as the nasbench code assumes that format)
    pickle.dump(model, open(model_path, 'wb'))
