from utils import DataLoader, DataProcessor, FeatureSelect

data_dir = 'BrakeData.csv'
data_loader = DataLoader()
df = data_loader.read_data(data_dir)
#df = data_loader.ImportAndCleanSQL(dataquery="select * from BrakeData")
df.columns.values

columns_to_remove = ['PostID', 'BrskLatheDate1', 'BrskLatheDate2',
                    'BrskLatheDate3', 'BrskLatheDate4',
                    'ReportingDateTime', 'DataSavedInDBDateTime']
df = data_loader.remove_col(df, column_name_list=columns_to_remove)

# create a new column for broms 1 and 2 thickness

cols = ['BrskThickness1', 'BrskThickness2', 'BrskThickness3', 'BrskThickness4']

data_prep = DataProcessor(df)
df_1 = data_prep.string_to_numeric(df, cols=cols)
(df_1['BrskThickness1'] + df_1['BrskThickness2'])/2
