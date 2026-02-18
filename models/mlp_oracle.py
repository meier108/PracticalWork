import torch
from torch.utils.data import DataLoader, random_split

import torch.nn as nn
import torch.optim as optim


class MLPOracle(nn.Module):
    def __init__(self, input_size, output_size=1, hidden_size=256, dropout_rate=0.3):
        super(MLPOracle, self).__init__()
        # Reduced model capacity + dropout for regularization
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout_rate)
        
        # Gaussian normalization parameters
        self.register_buffer('mean', torch.zeros(input_size))
        self.register_buffer('std', torch.ones(input_size))
    
    def forward(self, x):
        x = (x - self.mean) / (self.std + 1e-8)
        x = self.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
    def train_model(self, train_loader, val_loader=None, device='cpu', epochs=100, 
                    weight_decay=1e-4, patience=10):
        self.train()
        # Weight decay (L2 regularization) in optimizer
        optimizer = optim.Adam(self.parameters(), weight_decay=weight_decay)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        best_state = None
        epochs_without_improvement = 0
        
        for epoch in range(epochs):
            # Training
            self.train()
            total_loss = 0
            for inputs, targets in train_loader:
                inputs = inputs.float().to(device)
                targets = targets.float().to(device)
                
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, targets.unsqueeze(-1))
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            train_loss = total_loss / len(train_loader)
            
            # Validation
            if val_loader is not None:
                self.eval()
                val_loss = 0
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs = inputs.float().to(device)
                        targets = targets.float().to(device)
                        outputs = self(inputs)
                        val_loss += criterion(outputs, targets.unsqueeze(-1)).item()
                val_loss /= len(val_loader)
                
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in self.state_dict().items()}
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}")
        
        # Restore best model if validation was used
        if best_state is not None:
            self.load_state_dict(best_state)
            print(f"Restored best model with val loss: {best_val_loss:.4f}")
    
    def inference(self, x, device='cpu'):
        self.eval()
        with torch.no_grad():
            #x = x.to(device)
            predictions = self(x)
        return predictions
    
    def fit_normalization(self, data_loader):
        """Fit Gaussian normalization parameters"""
        all_data = []
        for batch in data_loader:
            all_data.append(batch[0])
        all_data = torch.cat(all_data, dim=0).float()  # Ensure float32
        self.mean = all_data.mean(dim=0)
        self.std = all_data.std(dim=0)

    def save_model(self, path):
        """Save the trained model parameters to a file."""
        torch.save(self.state_dict(), path)

    def load_model(self, path, device='cpu'):
        """Load model parameters from a file."""
        self.load_state_dict(torch.load(path, map_location=device))
        self.to(device)
    