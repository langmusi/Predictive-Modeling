import pandas as pd

import pandas_profiling as pp
from sklearn.preprocessing import OneHotEncoder

class DataProcessor: 
    
    def __init__(self, data):
        self.data = data
            
    
    def eda(self):

        return pp.ProfileReport(self.data)

    
    def one_hot_encoding(self, one_hotify_these_categorical):
        

        res = pd.get_dummies(self.data, 
                             prefix=one_hotify_these_categorical, 
                             columns=one_hotify_these_categorical)
        print('\tData shape before one-hot-encoding:', str(self.data.shape) + '\n')
        print('\tData shape after one-hot-encoding:',str(res.shape))
       
        return res


    def dummy_generator (self, cat_convert = True):
    
        features_data = self.data
        
        # Create dummy variables with prefix 'Littera'
        features_data = pd.concat([features_data,
                                pd.get_dummies(features_data['Littera'], prefix = 'L')], 
                                axis=1)
        # VehicleOperatorName dummy
        features_data = pd.concat([features_data, 
                                pd.get_dummies(features_data['VehicleOperatorName'],
                                                prefix = 'V')], axis=1)

        if cat_convert == True:    
            # delete variables we are not going to use anymore
            del features_data['VehicleOperatorName']
            del features_data['Littera']
            
        return features_data    