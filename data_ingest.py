
import pandas as pd
def data_load(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name)
    print(df.head())
    return df












