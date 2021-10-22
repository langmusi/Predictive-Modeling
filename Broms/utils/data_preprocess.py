import pandas as pd

import pandas_profiling as pp
from sklearn.preprocessing import OneHotEncoder

class DataProcessor: 
    
    def __init__(self, data):
        self.data = data
            
    
    def eda(self):

        return pp.ProfileReport(self.data)


    def string_to_numeric(self, data, cols):
        data[cols] = data[cols].apply(pd.to_numeric)
        
        return data

    
    def one_hot_encoding(self, one_hotify_these_categorical):
        

        res = pd.get_dummies(self.data, 
                             prefix=one_hotify_these_categorical, 
                             columns=one_hotify_these_categorical)
        print('\tData shape before one-hot-encoding:', str(self.data.shape) + '\n')
        print('\tData shape after one-hot-encoding:',str(res.shape))
       
        return res


    def data_transform_wide_to_long(self, identity_col, value_col_list, new_col_name, value_col_name):
        df_trans = pd.melt(self.data, 
                        id_vars=identity_col, 
                        value_vars=value_col_list,
                        var_name=new_col_name, 
                        value_name=value_col_name)
        return df_trans