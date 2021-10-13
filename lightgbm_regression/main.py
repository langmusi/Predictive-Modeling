
# from utils.data_load import DataLoader  --> works
# from utils.data_preprocess import DataProcessor  --> works
from utils import DataLoader, DataProcessor

data_dir = "C:/Users/LIUM3478/OneDrive Corp/OneDrive - Atkins Ltd/Work_Atkins/Docker/hjulanalys/wheel_prediction_data.csv"
data_loader = DataLoader(data_dir)
df = data_loader.load_data()
df.shape
data_load.load_data(data_path=)
