import pandas as pd

def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data.dropna(inplace=True)
    # その他の前処理
    return data
