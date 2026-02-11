"""
Random Forest Model for Binding Score Prediction

This module provides functionality to train, save, load, and use
a Random Forest model for predicting binding scores from DNA sequences.
"""

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr


class RandomForestPredictor:
    """
    Random Forest model for predicting binding scores from one-hot encoded DNA sequences.
    
    Attributes:
        model: The trained Random Forest model
        best_params: Best hyperparameters from grid search
        seed: Random seed for reproducibility
    """
    
    def __init__(self, seed=42):
        """
        Initialize the Random Forest predictor.
        
        Args:
            seed: Random seed for reproducibility (default: 42)
        """
        self.model = None
        self.best_params = None
        self.seed = seed
    
    def train_with_gridsearch(self, X_train, y_train, X_val, y_val,
                              param_grid=None, cv=3, n_jobs=-1, verbose=1):
        """
        Train a Random Forest model using GridSearchCV for hyperparameter tuning.
        
        Args:
            X_train: Training input sequences (one-hot encoded)
            y_train: Training target scores
            X_val: Validation input sequences
            y_val: Validation target scores
            param_grid: Dictionary of hyperparameters to search (optional)
            cv: Number of cross-validation folds (default: 3)
            n_jobs: Number of parallel jobs (default: -1 for all cores)
            verbose: Verbosity level (default: 1)
            
        Returns:
            Dictionary with results including best params and scores
        """
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 300, 500],
                'max_depth': [5, 10, 15, None],
                'min_samples_leaf': [1, 3, 5, 10],
                'min_samples_split': [2, 5, 10],
                'max_features': ['sqrt', 'log2', 0.5]
            }
        
        rf_search = GridSearchCV(
            RandomForestRegressor(random_state=self.seed),
            param_grid,
            cv=cv,
            scoring='r2',
            n_jobs=n_jobs,
            verbose=verbose
        )
        
        print("Starting GridSearchCV...")
        rf_search.fit(X_train, y_train)
        
        # Store the best model and parameters
        self.model = rf_search.best_estimator_
        self.best_params = rf_search.best_params_
        
        # Calculate validation metrics
        y_pred_val = self.model.predict(X_val)
        r2_val = r2_score(y_val, y_pred_val)
        mse_val = mean_squared_error(y_val, y_pred_val)
        
        results = {
            'best_params': self.best_params,
            'best_cv_r2': rf_search.best_score_,
            'val_r2': r2_val,
            'val_mse': mse_val
        }
        
        print(f'Best params: {self.best_params}')
        print(f'Best CV R2: {rf_search.best_score_}')
        print(f'Validation R2: {r2_val}')
        print(f'Validation MSE: {mse_val}')
        
        return results
    
    def train(self, X_train, y_train, n_estimators=500, max_depth=15,
              min_samples_leaf=3, min_samples_split=10, max_features='sqrt'):
        """
        Train a Random Forest model with specified hyperparameters.
        
        Args:
            X_train: Training input sequences
            y_train: Training target scores
            n_estimators: Number of trees (default: 500)
            max_depth: Maximum depth of trees (default: 15)
            min_samples_leaf: Minimum samples per leaf (default: 3)
            min_samples_split: Minimum samples to split node (default: 10)
            max_features: Number of features to consider (default: 'sqrt')
        """
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            max_features=max_features,
            random_state=self.seed,
            n_jobs=-1
        )
        
        print("Training Random Forest model...")
        self.model.fit(X_train, y_train)
        print("Training completed.")
    
    def evaluate(self, X_val, y_val):
        """
        Evaluate the model on validation data.
        
        Args:
            X_val: Validation input sequences
            y_val: Validation target scores
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            print("No model trained. Train the model first.")
            return None
        
        y_pred = self.model.predict(X_val)
        r2 = r2_score(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)
        spearman_corr, _ = spearmanr(y_val, y_pred)
        
        metrics = {
            'r2': r2,
            'mse': mse,
            'spearman_correlation': spearman_corr
        }
        
        print(f'R2 Score: {r2}')
        print(f'MSE: {mse}')
        print(f'Spearman Correlation: {spearman_corr}')
        
        return metrics
    
    
    def predict(self, flattened_sequence):
        """
        Predict binding score for a single flattened one-hot encoded sequence.
        
        Args:
            flattened_sequence: Flattened one-hot encoded sequence
            
        Returns:
            Predicted binding score
        """
        if self.model is None:
            print("No model available. Train or load a model first.")
            return None
        
        # Reshape for prediction: flatten has shape (sequence_length * 4,)
        # We need to reshape back to (-1, sequence_length * 4) for sklearn
        flat_array = np.asarray(flattened_sequence).reshape(1, -1)
        predicted_score = self.model.predict(flat_array)
        
        return predicted_score[0]
    
    def save_model(self, filepath='rf_model.pkl'):
        """
        Save the trained model to disk using joblib.
        
        Args:
            filepath: Path to save the model (default: 'rf_model.pkl')
        """
        if self.model is None:
            print("No model to save. Train the model first.")
            return
        
        joblib.dump(self.model, filepath)
        
        # Also save best params if available
        if self.best_params:
            params_filepath = filepath.replace('.pkl', '_params.pkl')
            joblib.dump(self.best_params, params_filepath)
            print(f'Model saved to {filepath}')
            print(f'Best params saved to {params_filepath}')
        else:
            print(f'Model saved to {filepath}')
    
    def load_model(self, filepath='rf_model.pkl'):
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model (default: 'rf_model.pkl')
        """
        try:
            self.model = joblib.load(filepath)
            print(f'Model loaded from {filepath}')
            
            # Try to load best params if available
            params_filepath = filepath.replace('.pkl', '_params.pkl')
            try:
                self.best_params = joblib.load(params_filepath)
                print(f'Best params loaded from {params_filepath}')
            except FileNotFoundError:
                print(f'No params file found at {params_filepath}')
        except FileNotFoundError:
            print(f'Model file not found at {filepath}')
            