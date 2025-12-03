"""
Data preprocessing and sequence preparation for weather forecasting
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List
import torch
from torch.utils.data import Dataset, DataLoader


class WeatherDataset(Dataset):
    """PyTorch Dataset for weather time series data"""

    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class WeatherDataPreprocessor:
    """Handles all data preprocessing for weather forecasting"""

    def __init__(self, sequence_length: int = 30, forecast_horizon: int = 7):
        """
        Initialize preprocessor

        Args:
            sequence_length: Number of past days to use for prediction
            forecast_horizon: Number of future days to predict
        """
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.target_columns = None

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load weather data from CSV file"""
        df = pd.read_csv(filepath)

        # Try to parse date column if it exists
        date_columns = ['date', 'Date', 'datetime', 'DateTime', 'time', 'Time']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                df = df.sort_values(by=col)
                break

        return df

    def prepare_features(self, df: pd.DataFrame,
                        target_cols: List[str] = None,
                        exclude_cols: List[str] = None) -> pd.DataFrame:
        """
        Prepare features by handling missing values and selecting relevant columns

        Args:
            df: Input dataframe
            target_cols: List of target column names to predict
            exclude_cols: List of columns to exclude (e.g., dates, IDs)

        Returns:
            Processed dataframe
        """
        df = df.copy()

        # Identify and exclude non-numeric columns
        if exclude_cols is None:
            exclude_cols = []

        # Add datetime columns to exclude list
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]) or \
               pd.api.types.is_object_dtype(df[col]):
                if col not in exclude_cols:
                    exclude_cols.append(col)

        # Get numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Remove excluded columns
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]

        # Handle missing values
        df[feature_cols] = df[feature_cols].fillna(method='ffill').fillna(method='bfill')

        # If target columns not specified, use all numeric columns
        if target_cols is None:
            # Common weather targets
            possible_targets = ['temperature', 'temp', 'Temperature', 'Temp',
                              'humidity', 'Humidity', 'pressure', 'Pressure']
            target_cols = [col for col in possible_targets if col in feature_cols]

            # If no common targets found, use first numeric column
            if not target_cols:
                target_cols = [feature_cols[0]]

        self.feature_columns = feature_cols
        self.target_columns = target_cols

        print(f"Selected features: {feature_cols}")
        print(f"Target columns: {target_cols}")

        return df[feature_cols]

    def create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series forecasting

        Args:
            data: Normalized data array

        Returns:
            X: Input sequences of shape (samples, sequence_length, features)
            y: Target values of shape (samples, forecast_horizon, targets)
        """
        X, y = [], []

        for i in range(len(data) - self.sequence_length - self.forecast_horizon + 1):
            # Input sequence
            X.append(data[i:i + self.sequence_length])

            # Target sequence (only target columns)
            target_idx = [self.feature_columns.index(col) for col in self.target_columns]
            y.append(data[i + self.sequence_length:
                         i + self.sequence_length + self.forecast_horizon,
                         target_idx])

        return np.array(X), np.array(y)

    def prepare_data(self, df: pd.DataFrame,
                     train_ratio: float = 0.7,
                     val_ratio: float = 0.15) -> Tuple:
        """
        Complete data preparation pipeline

        Args:
            df: Preprocessed dataframe with features
            train_ratio: Ratio of training data
            val_ratio: Ratio of validation data

        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        # Convert to numpy array
        data = df.values

        # Normalize data
        data_normalized = self.scaler.fit_transform(data)

        # Create sequences
        X, y = self.create_sequences(data_normalized)

        # Split data
        n = len(X)
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)

        X_train = X[:train_size]
        y_train = y[:train_size]

        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]

        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]

        print(f"\nData split:")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Input shape: {X_train.shape}")
        print(f"Output shape: {y_train.shape}")

        return X_train, y_train, X_val, y_val, X_test, y_test

    def create_dataloaders(self, X_train, y_train, X_val, y_val,
                          batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
        """Create PyTorch DataLoaders"""
        train_dataset = WeatherDataset(X_train, y_train)
        val_dataset = WeatherDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                 shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                               shuffle=False)

        return train_loader, val_loader

    def inverse_transform_targets(self, predictions: np.ndarray) -> np.ndarray:
        """
        Inverse transform predictions back to original scale

        Args:
            predictions: Normalized predictions

        Returns:
            Predictions in original scale
        """
        # Create a full array with all features (zeros for non-target features)
        n_features = len(self.feature_columns)
        full_array = np.zeros((predictions.shape[0], predictions.shape[1], n_features))

        # Fill in target columns
        target_idx = [self.feature_columns.index(col) for col in self.target_columns]
        for i, idx in enumerate(target_idx):
            full_array[:, :, idx] = predictions[:, :, i]

        # Reshape for inverse transform
        original_shape = full_array.shape
        reshaped = full_array.reshape(-1, n_features)

        # Inverse transform
        inverse = self.scaler.inverse_transform(reshaped)

        # Reshape back and extract target columns
        inverse = inverse.reshape(original_shape)
        result = inverse[:, :, target_idx]

        return result
