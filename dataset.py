import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
    data = data.drop(columns=[
        'Flow ID', 'Src Port', 'Dst Port', 'Timestamp', 'Attack Name',
        'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
        'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count',
        'ACK Flag Count', 'URG Flag Count', 'CWR Flag Count', 'ECE Flag Count',
        'Fwd Bytes/Bulk Avg', 'Fwd Packet/Bulk Avg', 'Fwd Bulk Rate Avg',
        'Bwd Bytes/Bulk Avg', 'Bwd Packet/Bulk Avg', 'Bwd Bulk Rate Avg'
    ], errors='ignore')

    # Convert IPs to integer representations
    data['Src_IP_Int'] = data['Src IP'].apply(lambda x: int(ipaddress.IPv4Address(x)))
    data['Dst_IP_Int'] = data['Dst IP'].apply(lambda x: int(ipaddress.IPv4Address(x)))

    # Drop original IP columns
    data = data.drop(columns=['Src IP', 'Dst IP'], errors='ignore')

    # Define features and labels
    features = data.drop(columns=['Label'], errors='ignore')
    labels = data['Label'].values

    # Apply MinMax scaling
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test