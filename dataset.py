import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)
    # data = data.drop(columns=['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Attack Name'])
    data = data.drop(columns=['Flow ID', 'Src IP', 'Src Port', 'Timestamp', 'Attack Name'])


    # Define features and labels
    X = data.drop(columns=['Label'])
    y = data['Label'].values

    # Encode categorical variable
    protocol_encoder = OneHotEncoder()
    protocol_encoded = protocol_encoder.fit_transform(X[['Protocol']]).toarray()
    DST_IP_encoder = OneHotEncoder()
    DST_IP_encoded = DST_IP_encoder.fit_transform(X[['Dst IP']]).toarray()
    DST_PORT_encoder = OneHotEncoder()
    DST_PORT_encoded = DST_PORT_encoder.fit_transform(X[['Dst Port']]).toarray()

    X_numeric = X.drop(columns=['Protocol'])
    X_numeric = X_numeric.drop(columns=['Dst Port'])
    X_numeric = X_numeric.drop(columns=['Dst IP']).values
    scaler = StandardScaler()
    X_numeric_scaled = scaler.fit_transform(X_numeric)


    # Combine preprocessed features
    X_processed = np.hstack([X_numeric_scaled, protocol_encoded,DST_IP_encoded,DST_PORT_encoded])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
