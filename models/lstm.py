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


def load_model(filepath='lstm_model.pt', input_size=4, lstm_units=128, 
               dense_units=32, dropout_rate=0.2):
    """Load model from disk."""
    model = LSTMSequencePredictor(input_size, lstm_units, dense_units, dropout_rate)
    model.load_state_dict(torch.load(filepath, map_location=get_device()))
    model.to(get_device())
    model.eval()
    return model


def train_with_ranking(model, X_train, y_train, X_val=None, y_val=None,
                       epochs=20, batch_size=32, lr=0.001, 
                       ranking_weight=0.5, margin=0.1, verbose=True):
    """
    Train with combined MSE + Margin Ranking Loss.
    
    The ranking loss ensures the model learns relative ordering of sequences,
    which is crucial for optimization even when absolute predictions are poor.
    
    Args:
        ranking_weight: Weight of ranking loss vs MSE (0-1). Higher = more ranking focus.
        margin: Minimum margin for ranking loss.
    """
    device = get_device()
    model.to(device)
    
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_criterion = nn.MSELoss()
    ranking_criterion = nn.MarginRankingLoss(margin=margin)
    
    history = {'loss': [], 'val_loss': [], 'mse': [], 'rank': []}
    
    for epoch in range(epochs):
        model.train()
        epoch_mse, epoch_rank = 0.0, 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            
            preds = model(X_batch)
            
            # MSE Loss
            mse_loss = mse_criterion(preds, y_batch)
            
            # Ranking Loss: Sample pairs and ensure correct ordering
            if len(preds) > 1:
                # Create pairs for ranking
                n = len(preds)
                idx1 = torch.randint(0, n, (min(n, 64),), device=device)
                idx2 = torch.randint(0, n, (min(n, 64),), device=device)
                
                pred1, pred2 = preds[idx1], preds[idx2]
                true1, true2 = y_batch[idx1], y_batch[idx2]
                
                # Target: +1 if true1 > true2, -1 otherwise
                target = torch.sign(true1 - true2)
                # Avoid 0 targets
                target[target == 0] = 1
                
                rank_loss = ranking_criterion(pred1, pred2, target)
            else:
                rank_loss = torch.tensor(0.0, device=device)
            
            # Combined loss
            total_loss = (1 - ranking_weight) * mse_loss + ranking_weight * rank_loss
            total_loss.backward()
            optimizer.step()
            
            epoch_mse += mse_loss.item()
            epoch_rank += rank_loss.item()
        
        avg_mse = epoch_mse / len(train_loader)
        avg_rank = epoch_rank / len(train_loader)
        total = (1 - ranking_weight) * avg_mse + ranking_weight * avg_rank
        
        history['loss'].append(total)
        history['mse'].append(avg_mse)
        history['rank'].append(avg_rank)
        
        # Validation
        val_loss = None
        if X_val is not None and y_val is not None:
            val_loss = evaluate(model, X_val, y_val)
            history['val_loss'].append(val_loss)
        
        if verbose:
            msg = f'Epoch {epoch+1}/{epochs} - mse: {avg_mse:.4f} - rank: {avg_rank:.4f}'
            if val_loss is not None:
                msg += f' - val_mse: {val_loss:.4f}'
            print(msg)
    
    return history
