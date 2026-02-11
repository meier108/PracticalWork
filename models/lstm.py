"""
LSTM Model for Sequence Prediction

This module provides functionality to train and use an LSTM model
for predicting binding scores from DNA sequences.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore


class LSTMSequencePredictor:
    """
    LSTM model for predicting binding scores from one-hot encoded DNA sequences.
    
    Attributes:
        model: The Keras Sequential model
        sequence_length: Length of input sequences
        seed: Random seed for reproducibility
    """
    
    def __init__(self, lstm_units=100, dense_units=4, dropout_rate=0.2, seed=42):
        """
        Initialize the LSTM model architecture.
        
        Args:
            lstm_units: Number of units in the LSTM layer (default: 100)
            dense_units: Number of units in the first dense layer (default: 4)
            dropout_rate: Dropout rate for regularization (default: 0.2)
            seed: Random seed for reproducibility (default: 42)
        """
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.seed = seed
        self.model = None
        self.sequence_length = None
        
    def build_model(self, input_shape):
        """
        Build the LSTM model architecture.
        
        Args:
            input_shape: Tuple of (timesteps, features) for the input layer
        """
        self.model = Sequential()
        self.model.add(LSTM(self.lstm_units, input_shape=input_shape))
        self.model.add(Dense(self.dense_units, activation='relu'))
        self.model.add(Dropout(self.dropout_rate))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
    def prepare_data(self, x, y, test_size=0.1):
        """
        Prepare data by one-hot encoding and splitting into train/validation sets.
        
        Args:
            x: Input sequences as numpy array (samples, sequence_length)
            y: Target binding scores as numpy array
            test_size: Fraction of data to use for validation (default: 0.1)
            
        Returns:
            Tuple of (X_train, X_val, y_train, y_val) with one-hot encoded inputs
        """
        # One-hot encode sequences (samples, timesteps, features=4)
        x_one_hot = np.array([tf.one_hot(seq, depth=4).numpy() for seq in x])
        self.sequence_length = x_one_hot.shape[1]
        
        print('One-hot encoded shape (3D):', x_one_hot.shape)
        
        # Split into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            x_one_hot, y, test_size=test_size, random_state=self.seed
        )
        
        print('Training samples:', X_train.shape[0])
        print('Validation samples:', X_val.shape[0])
        print('Input shape (timesteps, features):', X_train.shape[1:])
        
        return X_train, X_val, y_train, y_val
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=20, batch_size=32, verbose=1):
        """
        Train the LSTM model.
        
        Args:
            X_train: Training input sequences (one-hot encoded)
            y_train: Training target scores
            X_val: Validation input sequences (optional)
            y_val: Validation target scores (optional)
            epochs: Number of training epochs (default: 20)
            batch_size: Batch size for training (default: 32)
            verbose: Verbosity mode (default: 1)
            
        Returns:
            Training history object
        """
        if self.model is None:
            self.build_model(input_shape=X_train.shape[1:])
            self.model.summary()
        
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            verbose=verbose
        )
        
        return history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test input sequences (one-hot encoded)
            y_test: Test target scores
            
        Returns:
            Test loss and metrics
        """
        test_loss = self.model.evaluate(X_test, y_test)
        print('Test Loss:', test_loss)
        return test_loss
    
    def predict(self, X):
        """
        Predict binding scores for input sequences.
        
        Args:
            X: Input sequences (one-hot encoded)
            
        Returns:
            Predicted binding scores
        """
        return self.model.predict(X)
    
    def score_sequence(self, flattened_mutation):
        """
        Score a single one-hot encoded sequence.
        
        Args:
            flattened_mutation: Flattened one-hot encoded sequence
            
        Returns:
            Predicted binding score
        """
        flat = np.asarray(flattened_mutation).reshape(-1)
        sequence_length = flat.shape[0] // 4
        lstm_input = flat.reshape(1, sequence_length, 4)
        predicted_score = self.model.predict(lstm_input, verbose=0)
        return predicted_score
    
    def save_model(self, filepath='lstm_sequence_model.h5'):
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model (default: 'lstm_sequence_model.h5')
        """
        if self.model is not None:
            self.model.save(filepath)
            print(f'Model saved to {filepath}')
        else:
            print('No model to save. Train the model first.')
    
    def load_model(self, filepath='lstm_sequence_model.h5'):
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model (default: 'lstm_sequence_model.h5')
        """
        self.model = load_model(filepath)
        print(f'Model loaded from {filepath}')


def train_lstm_on_dataframe(df, target_column='binding_scores', 
                            lstm_units=100, dense_units=4, dropout_rate=0.2,
                            epochs=20, batch_size=32, test_size=0.1,
                            seed=42, save_path=None):
    """
    Convenience function to train an LSTM model directly from a dataframe.
    
    Args:
        df: DataFrame with sequence data and target scores
        target_column: Name of the target column (default: 'binding_scores')
        lstm_units: Number of LSTM units (default: 100)
        dense_units: Number of dense layer units (default: 4)
        dropout_rate: Dropout rate (default: 0.2)
        epochs: Number of training epochs (default: 20)
        batch_size: Training batch size (default: 32)
        test_size: Validation split fraction (default: 0.1)
        seed: Random seed (default: 42)
        save_path: Path to save the trained model (optional)
        
    Returns:
        Tuple of (trained_model, history, X_val, y_val)
    """
    # Extract features (all columns except last 5: target + split columns)
    x = df.iloc[:, :-5].values
    y = df[target_column].values
    
    print('Shape of x:', x.shape)
    print('First row of x:', x[0])
    
    # Initialize and prepare model
    lstm_predictor = LSTMSequencePredictor(
        lstm_units=lstm_units,
        dense_units=dense_units,
        dropout_rate=dropout_rate,
        seed=seed
    )
    
    # Prepare data
    X_train, X_val, y_train, y_val = lstm_predictor.prepare_data(x, y, test_size=test_size)
    
    # Train model
    history = lstm_predictor.train(
        X_train, y_train, X_val, y_val,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Evaluate model
    lstm_predictor.evaluate(X_val, y_val)
    
    # Save model if path provided
    if save_path:
        lstm_predictor.save_model(save_path)
    
    return lstm_predictor, history, X_val, y_val
