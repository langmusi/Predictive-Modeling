import pandas as pd
import numpy as np

class DataLoader:

    def __init__(self, data_dir: str):
        self.data_dir = data_dir


    def load_data(self):
        df = pd.read_csv(self, encoding = 'ISO 8859-1', sep = ";", decimal=",")

        return df


    # converting object type to category for gradient boosting algorithms
    def obj_to_cat(data):
        obj_feat = list(data.loc[:, data.dtypes == 'object'].columns.values)

        for feature in obj_feat:
            data[feature] = pd.Series(data[feature], dtype="category")

        return data