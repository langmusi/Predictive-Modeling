
# from utils.data_load import DataLoader  --> works
# from utils.data_preprocess import DataProcessor  --> works
from utils import DataLoader, DataProcessor, FeatureSelect

data_dir = 'C:/Users/LIUM3478/OneDrive Corp/OneDrive - Atkins Ltd/Work_Atkins/Docker/hjulanalys/wheel_prediction_data.csv'
data_loader = DataLoader()
df = data_loader.read_data(data_dir)

columns_to_remove = ['seq', 'counter', 'ref', 'TotalPerformanceADJ',
                    'nextOMSPerf', 'diamDiff', 'perfDiff']
df = data_loader.remove_col(df, column_name_list=columns_to_remove)

data_loader.correlation_numeric_col(df, corr_method="pearson")
data_loader.correlation_numeric_col(df, corr_method="kendall")
data_loader.correlation_numeric_col(df, corr_method="spearman")

cols_list = ["TotalPerformanceSnapshot", "LeftWheelDiameter", 
             "maxTotalPerformanceSnapshot", "km_till_OMS"]
feature_selector = FeatureSelect(data = df[cols_list])

feature_selector.corr_standarised_num_col(corr_method="pearson")
feature_selector.heat_map()




dat_prep = DataProcessor(df)
dat_prep.eda()

one_hot_enc = DataProcessor(df)
one_hotify_these_categorical = ["VehicleOperatorName", "Littera"]
one_hot_enc.one_hot_encoding(one_hotify_these_categorical)




