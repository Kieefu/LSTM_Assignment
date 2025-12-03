"""
Weather forecasting functions using trained LSTM model
"""
import numpy as np
import torch
import pandas as pd
from typing import List, Tuple, Optional
from datetime import datetime, timedelta


class WeatherForecaster:
    """Class for making weather forecasts using trained LSTM model"""

    def __init__(self, model, preprocessor, device='cpu'):
        """
        Initialize forecaster

        Args:
            model: Trained LSTM model
            preprocessor: Data preprocessor with fitted scaler
            device: Device to run predictions on
        """
        self.model = model.to(device)
        self.model.eval()
        self.preprocessor = preprocessor
        self.device = device

    def forecast_from_sequence(self, input_sequence: np.ndarray,
                               return_original_scale: bool = True) -> np.ndarray:
        """
        Make forecast from a single input sequence

        Args:
            input_sequence: Input sequence of shape (sequence_length, features)
            return_original_scale: Whether to inverse transform predictions

        Returns:
            Forecast of shape (forecast_horizon, n_targets)
        """
        # Reshape to (1, sequence_length, features)
        if input_sequence.ndim == 2:
            input_sequence = input_sequence.reshape(1, *input_sequence.shape)

        # Convert to tensor
        input_tensor = torch.FloatTensor(input_sequence).to(self.device)

        # Make prediction
        with torch.no_grad():
            prediction = self.model(input_tensor)

        # Convert to numpy
        prediction = prediction.cpu().numpy()

        # Inverse transform if requested
        if return_original_scale:
            prediction = self.preprocessor.inverse_transform_targets(prediction)

        return prediction[0]  # Remove batch dimension

    def multi_step_forecast(self, initial_sequence: np.ndarray,
                            n_steps: int,
                            return_original_scale: bool = True) -> np.ndarray:
        """
        Make multi-step forecast by iteratively predicting

        Args:
            initial_sequence: Initial sequence (sequence_length, features)
            n_steps: Number of forecast steps
            return_original_scale: Whether to inverse transform

        Returns:
            Multi-step forecast
        """
        forecasts = []
        current_sequence = initial_sequence.copy()

        for _ in range(n_steps):
            # Make prediction
            forecast = self.forecast_from_sequence(
                current_sequence,
                return_original_scale=False
            )

            # Take first forecast step
            next_step = forecast[0]  # (n_targets,)

            forecasts.append(next_step)

            # Update sequence for next iteration
            # This assumes we're updating with the predicted targets
            # You might need to adjust based on your feature set
            current_sequence = np.roll(current_sequence, -1, axis=0)

            # Update the last time step with predicted values
            # Only update the target columns
            target_idx = [self.preprocessor.feature_columns.index(col)
                         for col in self.preprocessor.target_columns]
            current_sequence[-1, target_idx] = next_step

        forecasts = np.array(forecasts)

        # Inverse transform if requested
        if return_original_scale:
            # Reshape for inverse transform
            forecasts = forecasts.reshape(1, -1, len(self.preprocessor.target_columns))
            forecasts = self.preprocessor.inverse_transform_targets(forecasts)
            forecasts = forecasts[0]

        return forecasts

    def batch_forecast(self, sequences: np.ndarray,
                       return_original_scale: bool = True) -> np.ndarray:
        """
        Make forecasts for batch of sequences

        Args:
            sequences: Input sequences of shape (batch, sequence_length, features)
            return_original_scale: Whether to inverse transform

        Returns:
            Forecasts of shape (batch, forecast_horizon, n_targets)
        """
        # Convert to tensor
        input_tensor = torch.FloatTensor(sequences).to(self.device)

        # Make predictions
        with torch.no_grad():
            predictions = self.model(input_tensor)

        # Convert to numpy
        predictions = predictions.cpu().numpy()

        # Inverse transform if requested
        if return_original_scale:
            predictions = self.preprocessor.inverse_transform_targets(predictions)

        return predictions

    def forecast_with_dates(self, input_sequence: np.ndarray,
                           start_date: datetime,
                           return_original_scale: bool = True) -> pd.DataFrame:
        """
        Make forecast and return as DataFrame with dates

        Args:
            input_sequence: Input sequence
            start_date: Start date for forecast
            return_original_scale: Whether to inverse transform

        Returns:
            DataFrame with forecast and dates
        """
        # Make forecast
        forecast = self.forecast_from_sequence(
            input_sequence,
            return_original_scale=return_original_scale
        )

        # Create dates
        dates = [start_date + timedelta(days=i) for i in range(len(forecast))]

        # Create DataFrame
        df = pd.DataFrame(
            forecast,
            columns=self.preprocessor.target_columns,
            index=dates
        )
        df.index.name = 'Date'

        return df

    def get_forecast_intervals(self, input_sequence: np.ndarray,
                              n_samples: int = 100,
                              dropout_rate: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate forecast uncertainty using Monte Carlo dropout

        Args:
            input_sequence: Input sequence
            n_samples: Number of MC samples
            dropout_rate: Dropout rate for MC dropout

        Returns:
            Tuple of (mean_forecast, lower_bound, upper_bound)
        """
        # Enable dropout during inference
        self.model.train()

        forecasts = []

        # Reshape input
        if input_sequence.ndim == 2:
            input_sequence = input_sequence.reshape(1, *input_sequence.shape)

        input_tensor = torch.FloatTensor(input_sequence).to(self.device)

        # Make multiple predictions with dropout
        for _ in range(n_samples):
            with torch.no_grad():
                prediction = self.model(input_tensor)
            forecasts.append(prediction.cpu().numpy())

        forecasts = np.array(forecasts).squeeze(1)  # (n_samples, horizon, targets)

        # Calculate statistics
        mean_forecast = np.mean(forecasts, axis=0)
        std_forecast = np.std(forecasts, axis=0)

        # 95% confidence interval
        lower_bound = mean_forecast - 1.96 * std_forecast
        upper_bound = mean_forecast + 1.96 * std_forecast

        # Inverse transform
        mean_forecast = self.preprocessor.inverse_transform_targets(
            mean_forecast.reshape(1, *mean_forecast.shape)
        )[0]
        lower_bound = self.preprocessor.inverse_transform_targets(
            lower_bound.reshape(1, *lower_bound.shape)
        )[0]
        upper_bound = self.preprocessor.inverse_transform_targets(
            upper_bound.reshape(1, *upper_bound.shape)
        )[0]

        # Set model back to eval mode
        self.model.eval()

        return mean_forecast, lower_bound, upper_bound


def create_forecast_report(forecaster: WeatherForecaster,
                          input_sequence: np.ndarray,
                          start_date: datetime,
                          actual_values: Optional[np.ndarray] = None) -> pd.DataFrame:
    """
    Create comprehensive forecast report

    Args:
        forecaster: WeatherForecaster instance
        input_sequence: Input sequence for forecast
        start_date: Start date of forecast
        actual_values: Actual values if available for comparison

    Returns:
        DataFrame with forecast report
    """
    # Get forecast
    forecast_df = forecaster.forecast_with_dates(
        input_sequence,
        start_date,
        return_original_scale=True
    )

    # Get uncertainty estimates
    mean_forecast, lower_bound, upper_bound = forecaster.get_forecast_intervals(
        input_sequence
    )

    # Add confidence intervals to DataFrame
    for i, col in enumerate(forecaster.preprocessor.target_columns):
        forecast_df[f'{col}_lower_95'] = lower_bound[:, i]
        forecast_df[f'{col}_upper_95'] = upper_bound[:, i]

    # Add actual values if provided
    if actual_values is not None:
        for i, col in enumerate(forecaster.preprocessor.target_columns):
            forecast_df[f'{col}_actual'] = actual_values[:, i]

            # Calculate error
            error = actual_values[:, i] - forecast_df[col].values
            forecast_df[f'{col}_error'] = error
            forecast_df[f'{col}_abs_error'] = np.abs(error)

    return forecast_df


def compare_forecasts(forecaster: WeatherForecaster,
                     test_sequences: np.ndarray,
                     test_targets: np.ndarray,
                     n_examples: int = 5) -> List[pd.DataFrame]:
    """
    Compare forecasts with actual values for multiple examples

    Args:
        forecaster: WeatherForecaster instance
        test_sequences: Test input sequences
        test_targets: Actual target values
        n_examples: Number of examples to compare

    Returns:
        List of comparison DataFrames
    """
    comparisons = []

    for i in range(min(n_examples, len(test_sequences))):
        # Make forecast
        forecast = forecaster.forecast_from_sequence(
            test_sequences[i],
            return_original_scale=True
        )

        # Create comparison DataFrame
        data = {}
        for j, col in enumerate(forecaster.preprocessor.target_columns):
            data[f'{col}_forecast'] = forecast[:, j]
            data[f'{col}_actual'] = test_targets[i, :, j]
            data[f'{col}_error'] = test_targets[i, :, j] - forecast[:, j]

        df = pd.DataFrame(data)
        df['forecast_day'] = range(1, len(df) + 1)
        comparisons.append(df)

    return comparisons
