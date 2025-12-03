"""
Visualization functions for weather forecasting results
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Optional
import os


# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def plot_training_history(history: dict, save_path: Optional[str] = None):
    """
    Plot training and validation loss

    Args:
        history: Training history dictionary
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot losses
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss (MSE)', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Plot learning rate
    ax2.plot(epochs, history['learning_rates'], 'g-', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Learning Rate', fontsize=12)
    ax2.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")

    plt.show()


def plot_predictions_vs_actual(predictions: np.ndarray, targets: np.ndarray,
                               feature_names: List[str],
                               n_samples: int = 100,
                               save_path: Optional[str] = None):
    """
    Plot predictions vs actual values

    Args:
        predictions: Predicted values (samples, horizon, features)
        targets: Actual values (samples, horizon, features)
        feature_names: Names of features
        n_samples: Number of samples to plot
        save_path: Path to save figure
    """
    n_features = predictions.shape[2]
    n_samples = min(n_samples, predictions.shape[0])

    # Flatten for scatter plot
    pred_flat = predictions[:n_samples].reshape(-1, n_features)
    target_flat = targets[:n_samples].reshape(-1, n_features)

    # Create subplots
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))

    if n_features == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_features > 1 else [axes]

    for i, ax in enumerate(axes[:n_features]):
        # Scatter plot
        ax.scatter(target_flat[:, i], pred_flat[:, i],
                  alpha=0.5, s=20, edgecolors='k', linewidths=0.5)

        # Perfect prediction line
        min_val = min(target_flat[:, i].min(), pred_flat[:, i].min())
        max_val = max(target_flat[:, i].max(), pred_flat[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val],
               'r--', linewidth=2, label='Perfect Prediction')

        ax.set_xlabel(f'Actual {feature_names[i]}', fontsize=11)
        ax.set_ylabel(f'Predicted {feature_names[i]}', fontsize=11)
        ax.set_title(f'{feature_names[i]} - Predictions vs Actual',
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    # Remove empty subplots
    for i in range(n_features, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Predictions vs actual plot saved to {save_path}")

    plt.show()


def plot_forecast_sequence(forecast: np.ndarray,
                          actual: Optional[np.ndarray] = None,
                          feature_names: List[str] = None,
                          confidence_intervals: Optional[tuple] = None,
                          save_path: Optional[str] = None):
    """
    Plot forecast sequence with optional confidence intervals

    Args:
        forecast: Forecast values (horizon, features)
        actual: Actual values if available
        feature_names: Names of features
        confidence_intervals: Tuple of (lower_bound, upper_bound)
        save_path: Path to save figure
    """
    horizon = forecast.shape[0]
    n_features = forecast.shape[1]

    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(n_features)]

    # Create subplots
    n_cols = min(2, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 5 * n_rows))

    if n_features == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_features > 1 else [axes]

    days = np.arange(1, horizon + 1)

    for i, ax in enumerate(axes[:n_features]):
        # Plot forecast
        ax.plot(days, forecast[:, i], 'b-', linewidth=2,
               marker='o', markersize=6, label='Forecast')

        # Plot actual if available
        if actual is not None:
            ax.plot(days, actual[:, i], 'g-', linewidth=2,
                   marker='s', markersize=6, label='Actual')

        # Plot confidence intervals
        if confidence_intervals is not None:
            lower, upper = confidence_intervals
            ax.fill_between(days, lower[:, i], upper[:, i],
                           alpha=0.2, color='blue', label='95% CI')

        ax.set_xlabel('Forecast Day', fontsize=11)
        ax.set_ylabel(feature_names[i], fontsize=11)
        ax.set_title(f'{feature_names[i]} Forecast',
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(days)

    # Remove empty subplots
    for i in range(n_features, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Forecast sequence plot saved to {save_path}")

    plt.show()


def plot_residuals(predictions: np.ndarray, targets: np.ndarray,
                  feature_names: List[str],
                  save_path: Optional[str] = None):
    """
    Plot residual analysis

    Args:
        predictions: Predicted values
        targets: Actual values
        feature_names: Names of features
        save_path: Path to save figure
    """
    n_features = predictions.shape[2]

    # Calculate residuals
    residuals = targets - predictions

    # Flatten
    residuals_flat = residuals.reshape(-1, n_features)
    predictions_flat = predictions.reshape(-1, n_features)

    # Create figure
    fig, axes = plt.subplots(n_features, 2, figsize=(15, 5 * n_features))

    if n_features == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_features):
        # Residual plot
        axes[i, 0].scatter(predictions_flat[:, i], residuals_flat[:, i],
                          alpha=0.5, s=20, edgecolors='k', linewidths=0.5)
        axes[i, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[i, 0].set_xlabel(f'Predicted {feature_names[i]}', fontsize=11)
        axes[i, 0].set_ylabel('Residuals', fontsize=11)
        axes[i, 0].set_title(f'{feature_names[i]} - Residual Plot',
                            fontsize=12, fontweight='bold')
        axes[i, 0].grid(True, alpha=0.3)

        # Histogram of residuals
        axes[i, 1].hist(residuals_flat[:, i], bins=50, edgecolor='black', alpha=0.7)
        axes[i, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[i, 1].set_xlabel('Residual Value', fontsize=11)
        axes[i, 1].set_ylabel('Frequency', fontsize=11)
        axes[i, 1].set_title(f'{feature_names[i]} - Residual Distribution',
                            fontsize=12, fontweight='bold')
        axes[i, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Residual plot saved to {save_path}")

    plt.show()


def plot_metrics_by_horizon(horizon_metrics: pd.DataFrame,
                           save_path: Optional[str] = None):
    """
    Plot metrics degradation over forecast horizon

    Args:
        horizon_metrics: DataFrame with metrics per horizon
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    metrics = ['RMSE', 'MAE', 'MAPE']

    for ax, metric in zip(axes, metrics):
        for feature in horizon_metrics['Feature'].unique():
            feature_data = horizon_metrics[horizon_metrics['Feature'] == feature]
            ax.plot(feature_data['Horizon'], feature_data[metric],
                   marker='o', linewidth=2, markersize=8,
                   label=f'Feature {feature}')

        ax.set_xlabel('Forecast Horizon (days)', fontsize=11)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(f'{metric} by Forecast Horizon',
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics by horizon plot saved to {save_path}")

    plt.show()


def plot_time_series_comparison(predictions: np.ndarray,
                                targets: np.ndarray,
                                feature_names: List[str],
                                n_sequences: int = 3,
                                save_path: Optional[str] = None):
    """
    Plot time series comparison for multiple sequences

    Args:
        predictions: Predicted sequences
        targets: Actual sequences
        feature_names: Names of features
        n_sequences: Number of sequences to plot
        save_path: Path to save figure
    """
    n_features = predictions.shape[2]
    n_sequences = min(n_sequences, predictions.shape[0])

    fig, axes = plt.subplots(n_sequences, n_features,
                            figsize=(8 * n_features, 4 * n_sequences))

    if n_sequences == 1 and n_features == 1:
        axes = np.array([[axes]])
    elif n_sequences == 1:
        axes = axes.reshape(1, -1)
    elif n_features == 1:
        axes = axes.reshape(-1, 1)

    for seq in range(n_sequences):
        for feat in range(n_features):
            ax = axes[seq, feat]

            horizon = len(predictions[seq])
            days = np.arange(1, horizon + 1)

            ax.plot(days, targets[seq, :, feat], 'g-',
                   linewidth=2, marker='o', markersize=6, label='Actual')
            ax.plot(days, predictions[seq, :, feat], 'b--',
                   linewidth=2, marker='s', markersize=6, label='Predicted')

            ax.set_xlabel('Day', fontsize=10)
            ax.set_ylabel(feature_names[feat], fontsize=10)
            ax.set_title(f'Sequence {seq + 1} - {feature_names[feat]}',
                        fontsize=11, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Time series comparison plot saved to {save_path}")

    plt.show()


def create_visualization_report(results: dict, save_dir: str = 'results/figures'):
    """
    Create comprehensive visualization report

    Args:
        results: Dictionary with evaluation results
        save_dir: Directory to save figures
    """
    os.makedirs(save_dir, exist_ok=True)

    print("\nGenerating visualization report...")

    # Get data
    predictions = results['predictions']
    targets = results['targets']

    # Determine feature names
    if 'feature_names' in results:
        feature_names = results['feature_names']
    else:
        n_features = predictions.shape[2]
        feature_names = [f'Feature {i}' for i in range(n_features)]

    # 1. Predictions vs Actual
    print("Creating predictions vs actual plot...")
    plot_predictions_vs_actual(
        predictions, targets, feature_names,
        save_path=os.path.join(save_dir, 'predictions_vs_actual.png')
    )

    # 2. Residual Analysis
    print("Creating residual analysis plots...")
    plot_residuals(
        predictions, targets, feature_names,
        save_path=os.path.join(save_dir, 'residual_analysis.png')
    )

    # 3. Time series comparison
    print("Creating time series comparison plots...")
    plot_time_series_comparison(
        predictions, targets, feature_names,
        n_sequences=5,
        save_path=os.path.join(save_dir, 'time_series_comparison.png')
    )

    # 4. Metrics by horizon
    if 'horizon_metrics' in results:
        print("Creating metrics by horizon plot...")
        plot_metrics_by_horizon(
            results['horizon_metrics'],
            save_path=os.path.join(save_dir, 'metrics_by_horizon.png')
        )

    print(f"\nAll visualizations saved to {save_dir}/")
