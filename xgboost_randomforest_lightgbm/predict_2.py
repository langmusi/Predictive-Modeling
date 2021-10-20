import pandas as pd

import numpy as np
from numpy import mean
from numpy import std
from numpy import NaN
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import GridSearchCV

import math
from xgboost import XGBRFRegressor
print(xgb.__version__)



# https://www.kaggle.com/shreyagopal/suicide-rate-prediction-with-machine-learning
#from sklearn.linear_model import LinearRegression
dat = "C:/Users/LIUM3478/OneDrive Corp/OneDrive - Atkins Ltd/Work_Atkins/Docker/hjulanalys/wheel_prediction_data.csv"
df = pd.read_csv(dat, encoding = 'ISO 8859-1', sep = ";", decimal=",")
df.head()

df.groupby(['Littera','VehicleOperatorName']).size().reset_index().rename(columns={0:'count'})

y = df[['km_till_OMS']].values
X = df[["LeftWheelDiameter", "Littera", "VehicleOperatorName",
        "TotalPerformanceSnapshot", "maxTotalPerformanceSnapshot"]]
# X["Littera_Operator"] = X.Littera + " " + X.VehicleOperatorName
# X.drop(["Littera", "VehicleOperatorName"], axis = 1, inplace=True)

def feature_generator (data, train = False):
    
    features_data = data
    
    # Create dummy variables with prefix 'Littera'
    features_data = pd.concat([features_data,
                               pd.get_dummies(features_data['Littera'], prefix = 'L')], 
                               axis=1)
    # VehicleOperatorName dummy
    features_data = pd.concat([features_data, 
                               pd.get_dummies(features_data['VehicleOperatorName'],
                                              prefix = 'V')], axis=1)
        
    # delete variables we are not going to use anymore
    del features_data['VehicleOperatorName']
    del features_data['Littera']
      
    return features_data     
    
# Generate features from training dataset
X = feature_generator(X)

# correlation of X
plt.figure(figsize=(12,10))
cor = X.corr()
sns.heatmap(cor)

# Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1234)

# set model
xgb_model = xgb.XGBRegressor()


# A parameter grid for XGBoost
params = {'learning_rate': [.03, 0.05, .07], #so called `eta` value,
          'min_child_weight': [4],
          'gamma': [0.5, 1, 1.5, 2, 5],
          'subsample': [0.7],
          'colsample_bytree': [1.0],
          'max_depth': [3, 4, 5]
        }


# set GridSearchCV parameters
xgb_model_grid = GridSearchCV(xgb_model, params, 
                              verbose = True, n_jobs = 2, cv = 2)

# use training data
xgb_model.fit(X_train, y_train)
print(xgb_model_grid.best_score_)
print(xgb_model_grid.best_params_)

y_pred = xgb_model.predict(X_test)
y_pred = pd.DataFrame(y_pred) 

xgb.plot_importance(xgb_model)

# eval
(np.nanmean((y_pred - y_test) ** 2))**0.5

def root_mean_squared_log_error(y_validations, y_predicted):
    if len(y_predicted) != len(y_validations):
        return 'error: mismatch in number of data points between x and y'
    y_predict_modified = [math.log(i) for i in y_predicted]
    y_validations_modified = [math.log(i) for i in y_validations]

    return mean_squared_error(y_validations_modified, y_predict_modified, squared=False)




