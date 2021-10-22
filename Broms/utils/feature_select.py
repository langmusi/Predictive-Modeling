import pandas as pd
from sklearn.preprocessing import StandardScaler

import seaborn as sns
import matplotlib.pyplot as plt

class FeatureSelect:

    def __init__(self, data):
        self.data = data

        
    def correlation_numeric_col(self, corr_method):
        # the function excludes non-numerical variables automatically
        return pd.DataFrame(self.data.corr(method = corr_method))

    
    def corr_standarised_num_col(self, corr_method):
        # the scaler object (model)
        standard_scaler = StandardScaler()
        # fit and transform the data
        scaled_data = standard_scaler.fit_transform(self.data) 
        scaled_data = pd.DataFrame(scaled_data, columns = self.data.columns)

        return pd.DataFrame(scaled_data.corr(method = corr_method))

    
    def heat_map(self):
        #always remember your magic function if using Jupyter
        sns.heatmap(self.data)
        
                                    