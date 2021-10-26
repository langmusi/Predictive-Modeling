
import pandas as pd
import numpy as np


from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import lightgbm as lgb
from lightgbm import LGBMRegressor

# https://www.kaggle.com/shreyagopal/suicide-rate-prediction-with-machine-learning
#from sklearn.linear_model import LinearRegression
dat = "C:/Users/LIUM3478/OneDrive Corp/OneDrive - Atkins Ltd/Work_Atkins/Docker/hjulanalys/wheel_prediction_data.csv"
df = pd.read_csv(dat, encoding = 'ISO 8859-1', sep = ";", decimal=",")
df.head()
df.columns.values

df.groupby(['Littera','VehicleOperatorName']).size().reset_index().rename(columns={0:'count'})

df[(df['km_till_OMS'].isnull()) & (df['counter'] ==1) & (df['ActionCode'] == 'OMS')]

# sorting df by ComponentUniqueID and ActionDate, then choosing km_till_oms that is nan
df.sort_values(['ComponentUniqueID', 'ActionDate'], ascending=[True, True], inplace=True)
df.shape
df_notnull = df[df['km_till_OMS'].notnull()]
df_notnull.shape
df_notnull = df_notnull.dropna(subset=['km_till_OMS'])
df_notnull.plot.scatter(x = 'ActionDate', y = 'km_till_OMS', s = 100);
###### 
y = df_notnull[['km_till_OMS']].values
X = df_notnull[["LeftWheelDiameter", "Littera", "VehicleOperatorName",
        "TotalPerformanceSnapshot"]]
# X["Littera_Operator"] = X.Littera + " " + X.VehicleOperatorName
# X.drop(["Littera", "VehicleOperatorName"], axis = 1, inplace=True)


############################### Light Gradient Boosting Decision Tree #####################
# converting object type to category for gradient boosting algorithms
def obj_to_cat(data):
    obj_feat = list(data.loc[:, data.dtypes == 'object'].columns.values)

    for feature in obj_feat:
        data[feature] = pd.Series(data[feature], dtype="category")

    return data

X = obj_to_cat(X)
    


# Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1234)
df_notnull.groupby(['Littera','VehicleOperatorName']).size().reset_index().rename(columns={0:'count'})


## sklearn
from lightgbm import LGBMRegressor # What was the problem at the beginning 
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

# model training （ Just as an ordinary model API To use , Set the parameters inside ）
gbm = LGBMRegressor(objective='regression', num_leaves=31, learning_rate=0.5, n_estimators=20)
gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='rmse', early_stopping_rounds=5)

gbm.best_iteration_
gbm.best_score_

y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
y_pred = pd.DataFrame(y_pred)
# eval
(np.nanmean((y_pred - y_test) ** 2))**0.5


############################ Linear Regression for groupped data ####################
### difficult to implement it here, easier in R
import statsmodels.api as sm 

def GroupRegress(data, yvar, xvars):
    Y = data[yvar]
    X = data[xvars]
    X['intercept'] = 1.
    result = sm.OLS(Y, X, missing='drop').fit()
    return result.params

def GroupRegress(data, yvar, xvars):
    regr = linear_model.LinearRegression()
    Y = data[yvar]
    X = data[xvars]
    regr.fit(X, Y)
    y_pred = regr.predict(test_df['LeftWheelDiameter'])
    y_pred = pd.DataFrame(y_pred)
    return y_pred


df_notnull_reg = df_notnull[["LeftWheelDiameter", "Littera", "VehicleOperatorName",
                        "TotalPerformanceSnapshot", "km_till_OMS"]]
train_df, test_df = train_test_split(df_notnull_reg, test_size = 0.2, random_state = 1234)
np.any(np.isfinite(train_df['km_till_OMS']),axis=0)
np.any(np.isnan(train_df['km_till_OMS']),axis=0)
np.where(np.isfinite(train_df['km_till_OMS']))
train_df.groupby(['Littera', 
                 'VehicleOperatorName']).apply(GroupRegress, 
                                              yvar='km_till_OMS', 
                                              xvars=['LeftWheelDiameter'])
    

from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

################################################################################
df.groupby(["Littera", "VehicleOperatorName"]).nunique()
df_small = df.loc[(df["Littera"] == "REGINA") & (df["VehicleOperatorName"] == "SJAB TÅG I BERGSLAGEN")]
df_small = df.loc[(df["Littera"] == "REGINA") & (df["VehicleOperatorName"] == "SJAB VÄNERTÅG")]

y = df_small[['km_till_OMS']].values
X = df_small[["LeftWheelDiameter", "TotalPerformanceSnapshot", "maxTotalPerformanceSnapshot"]]
       # "Littera", "VehicleOperatorName"]] 
X = feature_generator(X)
lightgbm_func(df_small)




