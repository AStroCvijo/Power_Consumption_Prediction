import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# Function for preprocessing data
def data_preprocess(data_path):
    
    # Load the data
    data = pd.read_csv("data/powerconsumption.csv")

    # Drop the Datetime column
    data = data.drop(columns=['Datetime'])

    # Normalize the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # Convert the scaled data back to a DataFrame
    data = pd.DataFrame(scaled_data, columns=data.columns)

    return data

# Function for extracting and preprocessing sequences and targets from data
def create_sequences(data, seq_length, prediction_step, prediction_target):
    
    # Initialize empty sequences and targets lists
    sequences = []
    targets = []

    # Extract the sequences and targets from data
    for i in range(len(data) - seq_length - prediction_step):
        seq = data.iloc[i:i+seq_length].values  # Convert each sequence to a NumPy array
        target = data[prediction_target].iloc[i + prediction_step + seq_length - 1]
        sequences.append(seq)
        targets.append(target)

    # Convert sequences and targets to numpy arrays for faster conversio to torch tensors
    sequences = np.array(sequences)
    targets = np.array(targets)

    # Convert sequences and targets to torch tensors
    sequences = torch.tensor(sequences, dtype = torch.float32)
    targets = torch.tensor(targets, dtype = torch.float32)

    # Reshape the sequences to match LSTM input (samples, timesteps, features)
    sequences = sequences.reshape((sequences.shape[0], seq_length, sequences.shape[2]))
        
    return sequences, targets

# Function for creating data loaders
def create_data_loaders(sequences, targets):

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(sequences, targets, test_size=0.2, random_state=42)

    # Create DataLoader for batching
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader