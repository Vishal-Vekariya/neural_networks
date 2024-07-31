import pandas as pd

def create_dummy_vars(data):
    # Create dummy variables for all 'object' type variables except 'Loan_Status'
    data = pd.get_dummies(data, columns=['University_Rating','Research']).astype(int)
    

    # Separate the input features and target variable
    x = data.drop(['Admit_Chance'], axis=1)
    y = data['Admit_Chance']

    return x, y