import pandas as pd
import numpy as np


class DataLoader:

    def __init__(self, data_dir):
        self.data_dir = data_dir
        


    def load_data(self):
        print(self.data_dir)
        df = pd.read_csv(self.data_dir, encoding = 'ISO 8859-1', sep = ";", decimal=",")

        return df

    # # converting object type to category for gradient boosting algorithms
    # def obj_to_cat(self):
    #     obj_feat = list(self.df.loc[:, self.df.dtypes == 'object'].columns.values)

    #     for feature in obj_feat:
    #         self.df[feature] = pd.Series(self.df[feature], dtype="category")

    #     return self.df

    