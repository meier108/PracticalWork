"""
LSTM Model for Sequence Prediction (PyTorch)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMSequencePredictor(nn.Module):
    """LSTM model for predicting binding scores from one-hot encoded sequences."""
    
    def __init__(self, input_size=4, lstm_units=100, dense_units=4, dropout_rate=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=lstm_units, batch_first=True)
        self.fc1 = nn.Linear(lstm_units, dense_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(dense_units, 1)
        self.input_size = input_size
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = torch.relu(self.fc1(lstm_out[:, -1, :]))
        out = self.dropout(out)
        return self.fc2(out)

    def predict(self, X):
        """Predict binding scores for given one-hot encoded sequences."""
        self.eval()
        flat = np.asarray(X).reshape(-1)
        seq_len = flat.shape[0] // self.input_size
        lstm_input = flat.reshape(1, seq_len, self.input_size).astype(np.float32)
        with torch.no_grad():
            X_t = torch.tensor(lstm_input, dtype=torch.float32).to(get_device())
            return self(X_t).cpu().numpy().flatten()[0]

def get_device():
    """Get available device (CUDA if available, else CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, X_train, y_train, X_val=None, y_val=None,
          epochs=20, batch_size=32, lr=0.001, verbose=True):
    """
    Train the model.
    
    Returns:
        dict with 'loss', 'val_loss' history
    """
    device = get_device()
    model.to(device)
    
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    history = {'loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        history['loss'].append(avg_loss)
        
        # Validation
        val_loss = None
        if X_val is not None and y_val is not None:
            val_loss = evaluate(model, X_val, y_val)
            history['val_loss'].append(val_loss)
        
        if verbose:
            msg = f'Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f}'
            if val_loss is not None:
                msg += f' - val_loss: {val_loss:.4f}'
            print(msg)
    
    return history


def evaluate(model, X, y):
    """Evaluate the model. Returns MSE loss."""
    device = get_device()
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        y_t = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(device)
        loss = nn.MSELoss()(model(X_t), y_t).item()
    return loss


def save_model(model, filepath='lstm_model.pt'):
    """Save model to disk."""
    torch.save(model.state_dict(), filepath)


def load_model(filepath='lstm_model.pt', input_size=4, lstm_units=100, 
               dense_units=4, dropout_rate=0.2):
    """Load model from disk."""
    model = LSTMSequencePredictor(input_size, lstm_units, dense_units, dropout_rate)
    model.load_state_dict(torch.load(filepath, map_location=get_device()))
    model.to(get_device())
    model.eval()
    return model
