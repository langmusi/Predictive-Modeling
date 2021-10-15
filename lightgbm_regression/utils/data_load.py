import pandas as pd
import numpy as np


class DataLoader:
    """ not using __init__() here because I use the ouput form one method
    as input to another method
    """

    def read_data(self, data_dir):
       df = pd.read_csv(data_dir, encoding = 'ISO 8859-1', sep = ";", decimal=",")
       print('Data shape:', str(df.shape) + '\n')
       print('Data columns:', str(df.columns))
       
       return df


    
    def remove_col(self, df, column_name_list):
        res = df.drop(column_name_list, axis=1)
        print('Data shape before dropping columns:', str(df.shape) + '\n')
        print('Data shape after dropping columns:', str(res.shape))

        return res


    # # converting object type to category for gradient boosting algorithms
    # def obj_to_cat(self):
    #     obj_feat = list(self.df.loc[:, self.df.dtypes == 'object'].columns.values)

    #     for feature in obj_feat:
    #         self.df[feature] = pd.Series(self.df[feature], dtype="category")

    #     return self.df

    