"""
Evaluation metrics and analysis for weather forecasting model
"""
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Tuple
import pandas as pd


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate various regression metrics

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Dictionary of metrics
    """
    # Flatten arrays for overall metrics
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # Calculate metrics
    mse = mean_squared_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    r2 = r2_score(y_true_flat, y_pred_flat)

    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true_flat - y_pred_flat) / (y_true_flat + 1e-8))) * 100

    # Explained Variance Score
    explained_var = 1 - np.var(y_true_flat - y_pred_flat) / np.var(y_true_flat)

    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape,
        'Explained_Variance': explained_var
    }

    return metrics


def calculate_metrics_per_horizon(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """
    Calculate metrics for each forecast horizon

    Args:
        y_true: True values of shape (samples, horizon, features)
        y_pred: Predicted values of shape (samples, horizon, features)

    Returns:
        DataFrame with metrics for each horizon
    """
    horizon = y_true.shape[1]
    n_features = y_true.shape[2]

    results = []

    for h in range(horizon):
        for f in range(n_features):
            y_t = y_true[:, h, f]
            y_p = y_pred[:, h, f]

            mse = mean_squared_error(y_t, y_p)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_t, y_p)
            r2 = r2_score(y_t, y_p)
            mape = np.mean(np.abs((y_t - y_p) / (y_t + 1e-8))) * 100

            results.append({
                'Horizon': h + 1,
                'Feature': f,
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'MAPE': mape
            })

    return pd.DataFrame(results)


def evaluate_model(model, test_loader, device='cpu') -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate model on test set

    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run evaluation on

    Returns:
        Tuple of (predictions, targets)
    """
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)

            # Get predictions
            output = model(data)

            all_predictions.append(output.cpu().numpy())
            all_targets.append(target.numpy())

    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    return predictions, targets


def print_evaluation_results(metrics: Dict[str, float], title: str = "Evaluation Results"):
    """Pretty print evaluation metrics"""
    print("\n" + "=" * 60)
    print(f"{title:^60}")
    print("=" * 60)

    for metric_name, value in metrics.items():
        print(f"{metric_name:.<30} {value:>15.6f}")

    print("=" * 60 + "\n")


def analyze_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Analyze residuals (errors)

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Dictionary with residual statistics
    """
    residuals = y_true.flatten() - y_pred.flatten()

    stats = {
        'Mean_Residual': np.mean(residuals),
        'Std_Residual': np.std(residuals),
        'Min_Residual': np.min(residuals),
        'Max_Residual': np.max(residuals),
        'Median_Residual': np.median(residuals),
        '25th_Percentile': np.percentile(residuals, 25),
        '75th_Percentile': np.percentile(residuals, 75)
    }

    return stats


def forecast_accuracy_by_horizon(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """
    Calculate forecast accuracy degradation over horizon

    Args:
        y_true: True values of shape (samples, horizon, features)
        y_pred: Predicted values of shape (samples, horizon, features)

    Returns:
        DataFrame with accuracy metrics per horizon step
    """
    horizon = y_true.shape[1]
    n_features = y_true.shape[2]

    horizon_metrics = []

    for h in range(horizon):
        # Average across all features for this horizon
        y_t = y_true[:, h, :].flatten()
        y_p = y_pred[:, h, :].flatten()

        rmse = np.sqrt(mean_squared_error(y_t, y_p))
        mae = mean_absolute_error(y_t, y_p)
        mape = np.mean(np.abs((y_t - y_p) / (y_t + 1e-8))) * 100

        horizon_metrics.append({
            'Forecast_Day': h + 1,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        })

    df = pd.DataFrame(horizon_metrics)
    return df


class ModelEvaluator:
    """Comprehensive model evaluation"""

    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.results = {}

    def evaluate(self, test_loader, preprocessor=None, save_results=True):
        """
        Comprehensive evaluation

        Args:
            test_loader: Test data loader
            preprocessor: Data preprocessor for inverse transform
            save_results: Save results to file

        Returns:
            Dictionary with all evaluation results
        """
        print("\n" + "=" * 60)
        print("Starting Model Evaluation")
        print("=" * 60)

        # Get predictions
        print("\nGenerating predictions...")
        predictions, targets = evaluate_model(self.model, test_loader, self.device)

        # Inverse transform if preprocessor provided
        if preprocessor is not None:
            print("Inverse transforming predictions to original scale...")
            predictions = preprocessor.inverse_transform_targets(predictions)
            targets = preprocessor.inverse_transform_targets(targets)

        # Overall metrics
        print("\nCalculating overall metrics...")
        overall_metrics = calculate_metrics(targets, predictions)
        self.results['overall_metrics'] = overall_metrics
        print_evaluation_results(overall_metrics, "Overall Performance")

        # Per-horizon metrics
        print("Calculating per-horizon metrics...")
        horizon_metrics = calculate_metrics_per_horizon(targets, predictions)
        self.results['horizon_metrics'] = horizon_metrics

        # Forecast accuracy by horizon
        accuracy_by_horizon = forecast_accuracy_by_horizon(targets, predictions)
        self.results['accuracy_by_horizon'] = accuracy_by_horizon

        print("\nForecast Accuracy by Horizon:")
        print(accuracy_by_horizon.to_string(index=False))

        # Residual analysis
        print("\nAnalyzing residuals...")
        residual_stats = analyze_residuals(targets, predictions)
        self.results['residual_stats'] = residual_stats
        print_evaluation_results(residual_stats, "Residual Statistics")

        # Store predictions and targets
        self.results['predictions'] = predictions
        self.results['targets'] = targets

        # Save results
        if save_results:
            self.save_results()

        return self.results

    def save_results(self, filepath='results/evaluation_results.npz'):
        """Save evaluation results"""
        import os
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

        np.savez(
            filepath,
            predictions=self.results['predictions'],
            targets=self.results['targets'],
            overall_metrics=self.results['overall_metrics'],
            residual_stats=self.results['residual_stats']
        )

        # Save DataFrames
        if 'horizon_metrics' in self.results:
            self.results['horizon_metrics'].to_csv(
                filepath.replace('.npz', '_horizon_metrics.csv'),
                index=False
            )

        if 'accuracy_by_horizon' in self.results:
            self.results['accuracy_by_horizon'].to_csv(
                filepath.replace('.npz', '_accuracy_by_horizon.csv'),
                index=False
            )

        print(f"\nResults saved to {filepath}")

    def get_best_and_worst_predictions(self, n=5):
        """Get best and worst predictions"""
        predictions = self.results['predictions']
        targets = self.results['targets']

        # Calculate error for each sample
        errors = np.mean(np.abs(predictions - targets), axis=(1, 2))

        # Get indices
        best_indices = np.argsort(errors)[:n]
        worst_indices = np.argsort(errors)[-n:]

        return {
            'best_indices': best_indices,
            'worst_indices': worst_indices,
            'best_errors': errors[best_indices],
            'worst_errors': errors[worst_indices]
        }
