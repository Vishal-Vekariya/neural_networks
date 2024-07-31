import pandas as pd
data_path = "data/raw/Admission.csv"


def load_and_preprocess_data(data_path):
    
    # Import the data from 'credit.csv'
    data = pd.read_csv(data_path)
    
    data['Admit_Chance']=(data['Admit_Chance'] >=0.8).astype(int)
    data = data.drop(['Serial_No'], axis=1)
    
    return data

if __name__ == "__main__":
    data = load_and_preprocess_data(data_path)
    print(data.head())