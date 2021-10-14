import pandas as pd

import pandas_profiling as pp
from sklearn.preprocessing import OneHotEncoder

class DataProcessor: 
    
    def __init__(self, data_name: str):
        self.data_name = data_name
    
    def eda(self):

        return pp.ProfileReport(self.data_name)


    def one_hot_encoding(self):
        columnsToEncode = ["VehicleOperatorName", "Littera"]

        res = pd.get_dummies(self.data_name, 
                       prefix=columnsToEncode, columns=columnsToEncode)
       
        return res