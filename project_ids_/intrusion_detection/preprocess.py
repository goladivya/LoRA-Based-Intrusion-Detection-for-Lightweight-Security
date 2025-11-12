# preprocess.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_preprocess(csv_path, label_col='label', drop_cols=None, test_size=0.2, random_state=42):
    """
    Generic loader:
      - reads CSV
      - drops columns in drop_cols (if any)
      - encodes label_col to binary {0,1}
      - fills missing values
      - returns X_train, X_test, y_train, y_test, scaler
    """
    df = pd.read_csv(csv_path)
    if drop_cols:
        df = df.drop(columns=drop_cols, errors='ignore')
    # make label binary (assumes benign/normal vs attack)
    lbl = df[label_col].astype(str)
    # common label variants: 'Benign','Normal','attack','DoS', etc.
    # We'll treat anything not equal to 'Benign'/'Normal' as attack.
    benign_vals = set(['benign','Benign','normal','Normal','0','0.0','NULL'])
    y = (~lbl.isin(benign_vals)).astype(int).values
    X = df.drop(columns=[label_col]).select_dtypes(include=[np.number])  # keep numeric features
    X = X.fillna(0.0)
    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=test_size, random_state=random_state, stratify=y)
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler, list(X.columns)
