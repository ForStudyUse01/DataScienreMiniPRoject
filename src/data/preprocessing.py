import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """Load and perform initial data preprocessing."""
    df = pd.read_csv(file_path)
    df = df.dropna()
    df['Attrition'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
    return df

def preprocess_data(df):
    """Perform data preprocessing including one-hot encoding and scaling."""
    # One-hot encoding
    df = pd.get_dummies(df, drop_first=True)
    
    # Normalizing numerical columns
    scaler = StandardScaler()
    numerical_cols = ['MonthlyIncome', 'DistanceFromHome', 'Age']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    # Converting to Indian metrics
    df['MonthlyIncome'] = df['MonthlyIncome'] * 85.80  # USD to INR
    df['DistanceFromHome'] = df['DistanceFromHome'].apply(lambda x: x * 1.5)  # Miles to KM
    
    return df, scaler 