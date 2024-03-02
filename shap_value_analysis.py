import joblib
import glob
import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import StratifiedKFold

data_df = YOUR_DATAFRAME
def Binarization(num):
    return 1 if num>10 else 0

model_path = YOUR_MODEL
print(model_path)

with open(model_path,'rb') as f:  
    model = joblib.load(f)

y = data_df['y']
x = data_df['x']

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(x)
shap.summary_plot(shap_values[1], x, )




