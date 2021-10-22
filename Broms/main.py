
# from utils.data_load import DataLoader  --> works
# from utils.data_preprocess import DataProcessor  --> works
from utils import DataLoader, DataProcessor, FeatureSelect

data_dir = 'BrakeData.csv'
data_loader = DataLoader()
df = data_loader.read_data(data_dir)
df = data_loader.ImportAndCleanSQL(dataquery="select * from BrakeData")
df.columns.values

columns_to_remove = ['PostID']
df = data_loader.remove_col(df, column_name_list=columns_to_remove)

data_loader.correlation_numeric_col(df, corr_method="pearson")
data_loader.correlation_numeric_col(df, corr_method="kendall")
data_loader.correlation_numeric_col(df, corr_method="spearman")

cols_list = ["TotalPerformanceSnapshot", "BrskThickness1", 
             "BrskThickness2", "BrskThickness3", "BrskThickness4"]
feature_selector = FeatureSelect(data = df[cols_list])

feature_selector.corr_standarised_num_col(corr_method="pearson")
feature_selector.heat_map()


# wide dataframe to long
# melt
import pandas as pd
df_changedate = pd.melt(df, 
                        id_vars=['PostID'], 
                        value_vars=['BrskChangeDate1', 'BrskChangeDate2', 'BrskChangeDate3', 'BrskChangeDate4'],
                        var_name='Brakes', value_name='ChangeDate')

df.join(df_changedate, on='PostID', how='right')

dat_prep = DataProcessor(df)
dat_prep.eda()

one_hot_enc = DataProcessor(df)
one_hotify_these_categorical = ["VehicleOperatorName", "Littera"]
one_hot_enc.one_hot_encoding(one_hotify_these_categorical)




