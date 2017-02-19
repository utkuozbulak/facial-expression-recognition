from src.pre_processing.extract_landscape import get_facial_vectors
from src.extract_data.get_data_from_csv import GetDataFromCSV

x_train_data_facial_vectors =  get_facial_vectors(only_test_data=True, load_cached=True)
_, y_train_data = GetDataFromCSV.get_test_data()
print("HERE")
