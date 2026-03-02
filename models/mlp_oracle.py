import torch
from torch.utils.data import DataLoader, random_split

import torch.nn as nn
import torch.optim as optim


class MLPOracle(nn.Module):
    def __init__(self, input_size, output_size=1, hidden_size=2048, dropout_rate=0.3):
        super(MLPOracle, self).__init__()
        # Two hidden layers + dropout for regularization
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.leaky_relu = nn.LeakyReLU(0.3)
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
    
    def _sample_proposal_distribution(self, batch_size, device):
        """
        Sample from proposal distribution μ(x).
        For discrete tasks, samples uniform random one-hot encoded sequences.
        """
        # Generate uniform random samples in the input space
        # Assuming one-hot encoding, sample random indices and create one-hot vectors
        input_size = self.mean.shape[0]
        samples = torch.rand(batch_size, input_size, device=device)
        return samples
    
    def _generate_adversarial_samples(self, x_init, adv_steps=50, adv_lr=0.01):
        """
        Generate adversarial samples via gradient ascent to find points where
        the model is currently overestimating.
        
        Args:
            x_init: Initial samples to start gradient ascent from (batch of training data)
            adv_steps: Number of gradient ascent steps (default: 50)
            adv_lr: Learning rate for gradient ascent
            
        Returns:
            x_adv: Adversarial samples that maximize model predictions
        """
        # Create a copy of inputs that requires gradients
        x_adv = x_init.clone().detach().requires_grad_(True)
        
        # Run gradient ascent to maximize model predictions
        for _ in range(adv_steps):
            # Forward pass
            pred = self(x_adv)
            
            # Compute gradient of prediction w.r.t. input
            grad = torch.autograd.grad(pred.sum(), x_adv, create_graph=False)[0]
            
            # Gradient ascent step (maximize prediction)
            with torch.no_grad():
                x_adv = x_adv + adv_lr * grad
                # Clamp to valid input range [0, 1] for normalized inputs
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
            
            x_adv = x_adv.requires_grad_(True)
        
        return x_adv.detach()
    
    def train_model(self, train_loader, val_loader=None, device='cpu', epochs=100, 
                    weight_decay=1e-4, patience=10, alpha=0.1, tau=2.0, use_com=True,
                    adv_steps=50, adv_lr=0.01):
        """
        Train the model with optional COM (Conservative Objective Model) regularization.
        
        COM Loss = MSE(f_θ(x_0), y) - α·E_{x_0~D}[f_θ(x_0)] + α·E_{x~μ(x)}[f_θ(x)]
        
        Uses adversarial sample generation via gradient ascent to find points where
        the model overestimates, instead of random proposal sampling.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data (optional)
            device: Device to train on
            epochs: Number of training epochs
            weight_decay: L2 regularization weight
            patience: Early stopping patience
            alpha: Lagrange multiplier for COM regularization (trades off conservatism for accuracy)
            tau: Constraint threshold for discrete tasks (default 2.0 for TFBind8)
            use_com: Whether to use COM regularization
            adv_steps: Number of gradient ascent steps for adversarial generation (default: 50)
            adv_lr: Learning rate for adversarial gradient ascent
        """
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
            total_mse = 0
            total_com = 0
            
            for inputs, targets in train_loader:
                inputs = inputs.float().to(device)
                targets = targets.float().to(device)
                
                optimizer.zero_grad()
                outputs = self(inputs)
                
                # MSE loss term
                mse_loss = criterion(outputs, targets.unsqueeze(-1))
                
                if use_com:
                    # Inner loop: Generate adversarial samples via gradient ascent
                    # Mine for points where the model is currently overestimating
                    adv_samples = self._generate_adversarial_samples(
                        inputs, adv_steps=adv_steps, adv_lr=adv_lr
                    )
                    
                    # COM regularization: -α·E[f(x_train)] + α·E[f(x_adv)]
                    # Encourages lower predictions on adversarial samples, higher on training data
                    
                    # E[f(x_train)] - expectation over training batch
                    train_expectation = outputs.mean()
                    
                    # Forward pass on adversarial samples
                    adv_outputs = self(adv_samples)
                    
                    # E[f(x_adv)] - expectation over adversarial samples
                    adv_expectation = adv_outputs.mean()
                    
                    # COM regularization term with tau constraint
                    # Penalize when adv_expectation - train_expectation > tau
                    com_gap = adv_expectation - train_expectation
                    com_loss = alpha * torch.clamp(com_gap + tau, min=0)
                    
                    loss = mse_loss + com_loss
                    total_com += com_loss.item()
                else:
                    loss = mse_loss
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_mse += mse_loss.item()
            
            train_loss = total_loss / len(train_loader)
            train_mse = total_mse / len(train_loader)
            
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
                
                if use_com:
                    com_avg = total_com / len(train_loader)
                    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f} (MSE: {train_mse:.4f}, COM: {com_avg:.4f}), Val Loss: {val_loss:.4f}")
                else:
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
                if use_com:
                    com_avg = total_com / len(train_loader)
                    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f} (MSE: {train_mse:.4f}, COM: {com_avg:.4f})")
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
    