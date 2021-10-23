from utils import DataLoader, DataProcessor, FeatureSelect, my_plot

data_dir = 'brake_data_azure.csv'
data_loader = DataLoader()
df = data_loader.read_data(data_dir)
#df = data_loader.ImportAndCleanSQL(dataquery="select * from BrakeData")
df.columns.values

################### plot ##########################
brake_plot = my_plot(df)

# checking if brake 1, 2 have the same change dates.
brake_plot.scatter_plot(col_x='BrskChangeDate1',
                        col_y='BrskChangeDate2',
                        data=df, group=False)

# checking if brake 3, 4 have the same change dates.
brake_plot.scatter_plot(col_x='BrskChangeDate3',
                        col_y='BrskChangeDate4',
                        data=df, group=False)

# checking the pattern of brake thickness vs. performance km.
brake_plot.scatter_plot(col_x='BrskThickness1',
                        col_y='TotalPerformanceSnapshot',
                        data=df, group=False)

brake_plot.scatter_plot(col_x='BrskThickness4',
                        col_y='TotalPerformanceSnapshot',
                        data=df, group=False)

# action data == brake change date ?
brake_plot.scatter_plot(col_x='ActionDate',
                        col_y='BrskChangeDate1',
                        data=df, group=False)

brake_plot.scatter_plot('ActionDate',
                        'BrskThickness1', df,
                        'ComponentUniqueID', group=True)

args = df['ComponentID']
brake_plot.scatter_plot('ComponentLocation',
                        'BrskThickness1', df,
                        *args, group=True)

args = df['ComponentID'].astype('category')
brake_plot.count_chart('ComponentLocation', df, group=False)
import seaborn as sns
import matplotlib.pyplot as plt
ax = sns.countplot(x='ComponentLocation', hue='ComponentParentLocation', 
                   data=df)
plt.show()
############################################
df[df['BrskChangeDate1'].isnull() & df['BrskChangeDate2'].notnull()]
df[df['BrskChangeDate2'].isnull() & df['BrskChangeDate1'].notnull()]


columns_to_remove = ['PostID', 'BrskLatheDate1', 'BrskLatheDate2',
                    'BrskLatheDate3', 'BrskLatheDate4',
                    'ReportingDateTime', 'DataSavedInDBDateTime']
df = data_loader.remove_col(df, column_name_list=columns_to_remove)

# create a new column for broms 1 and 2 thickness

cols = ['BrskThickness1', 'BrskThickness2', 'BrskThickness3', 'BrskThickness4']

data_prep = DataProcessor(df)
df_1 = data_prep.string_to_numeric(df, cols=cols)
(df_1['BrskThickness1'] + df_1['BrskThickness2'])/2
