# This file traines a MLP oracle model on the provided GB1 dataset. It includes data loading, 
# model training, and saving the trained model parameters.

from models import mlp_oracle
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import mavenn


def load_gb1_data(batch_size=64):
    """Load GB1 dataset from mavenn library. Extract x, y. One-hot-encode the sequences. Returns a PyTorch DataLoader."""
    dataset = mavenn.load_example_dataset('gb1')
    x = dataset['x']  # Sequences
    print(f"Loaded {x.shape[0]} sequences of length {x.shape[1]}.")
    y = dataset['y']  # Fitness values
    # One-hot-encode the sequences
    x_one_hot = np.zeros((x.shape[0], x.shape[1], 20))  # Assuming 20 amino acidsS
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            aa_index = x[i, j]
            if aa_index >= 0:  # Valid amino acid index
                x_one_hot[i, j, aa_index] = 1
    x_one_hot = x_one_hot.reshape(x.shape[0], -1)  # Flatten the one-hot encoding
    # Create a PyTorch DataLoader
    dataset = torch.utils.data.TensorDataset(torch.tensor(x_one_hot, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader

def train_oracle_model(model_save_path):
    """Train the MLP oracle model on the GB1 dataset and save the trained model parameters."""
    # Load data
    train_loader = load_gb1_data(batch_size=64)
    print("Data loaded successfully.")

    # Initialize model
    input_size = 20 * 4  # Assuming sequences of length 4 and 20 amino acids
    output_size = 1  # Predicting fitness value
    model = mlp_oracle.MLPOracle(input_size, output_size)
    
    # Fit normalization parameters
    print("Fitting normalization parameters...")
    model.fit_normalization(train_loader)
    
    # Train the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train_model(train_loader, device=device)
    
    # Save the trained model parameters
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save_model(model_save_path)

if __name__ == "__main__":
    train_loader = load_gb1_data(batch_size=64)
    print("Data loaded successfully.")
    print(train_loader.dataset[0])  # Print the first data point to verify loading
    train_oracle_model(model_save_path="models/oracle_mlp.pth")
