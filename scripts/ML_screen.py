# -*- coding: utf-8 -*-
import joblib
import glob
import pandas as pd
import numpy as np

# +
df = YOUR_DATAFRAME
feature_df = df['feature']
print(feature_df.shape)
model_path_list = YOUR_MODEL_PATH_LIST

model_list = []
for model_path in model_path_list:
    with open(model_path,'rb') as f:
        model_list.append(joblib.load(f))
# -

pred_y = np.array([ model.predict_proba(feature_df)[:,1] for model in model_list])
pred_y = pred_y.transpose()
pred_y_all = pred_y.sum(axis=1)
