import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import ipaddress

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def load_and_preprocess_data(filepath):
    # Load dataset
    data = pd.read_csv(filepath)

    # Drop irrelevant columns
    drop_columns = [
        'Flow ID', 'Timestamp', 'Attack Name', 'Fwd PSH Flags', 'Bwd PSH Flags',
        'Fwd URG Flags', 'Bwd URG Flags', 'FIN Flag Count', 'SYN Flag Count',
        'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count',
        'CWR Flag Count', 'ECE Flag Count', 'Fwd Bytes/Bulk Avg',
        'Fwd Packet/Bulk Avg', 'Fwd Bulk Rate Avg', 'Bwd Bytes/Bulk Avg',
        'Bwd Packet/Bulk Avg', 'Bwd Bulk Rate Avg'
    ]
    data = data.drop(columns=drop_columns, errors='ignore')

    # Combine IP and Port into a single feature
    data['Src_IP_Port'] = data['Src IP'] + ':' + data['Src Port'].astype(str)
    data['Dst_IP_Port'] = data['Dst IP'] + ':' + data['Dst Port'].astype(str)

    # Convert IP:Port to integer hash values
    data['Src_IP_Port_Int'] = data['Src_IP_Port'].apply(lambda x: hash(x) % (10 ** 9))
    data['Dst_IP_Port_Int'] = data['Dst_IP_Port'].apply(lambda x: hash(x) % (10 ** 9))

    # Drop original IP and Port columns
    data = data.drop(columns=['Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Src_IP_Port', 'Dst_IP_Port'], errors='ignore')

    # Define features and labels
    features = data.drop(columns=['Label'], errors='ignore')
    labels = data['Label'].values

    # Apply MinMax scaling
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=1997)

    return X_train, X_test, y_train, y_test