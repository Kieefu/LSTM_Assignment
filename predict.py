"""
Script to make weather predictions using a trained model
"""
import argparse
import torch
import pickle
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from model import WeatherLSTM, AttentionLSTM
from forecast import WeatherForecaster
from visualize import plot_forecast_sequence


def load_model_and_preprocessor(checkpoint_path, config_path, preprocessor_path, device='cpu'):
    """Load trained model and preprocessor"""

    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Load preprocessor
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)

    # Create model
    if config['model_type'] == 'lstm':
        model = WeatherLSTM(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            output_size=config['output_size'],
            forecast_horizon=config['forecast_horizon']
        )
    else:
        model = AttentionLSTM(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            output_size=config['output_size'],
            forecast_horizon=config['forecast_horizon']
        )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, preprocessor, config


def prepare_input_from_csv(csv_path, preprocessor, n_rows=None):
    """
    Prepare input sequence from CSV file

    Args:
        csv_path: Path to CSV file
        preprocessor: Fitted preprocessor
        n_rows: Number of most recent rows to use (default: sequence_length)

    Returns:
        Input sequence ready for prediction
    """
    # Load data
    df = pd.read_csv(csv_path)

    # Take most recent rows
    if n_rows is None:
        n_rows = preprocessor.sequence_length

    df_recent = df.tail(n_rows)

    # Select feature columns
    df_features = df_recent[preprocessor.feature_columns]

    # Handle missing values
    df_features = df_features.fillna(method='ffill').fillna(method='bfill')

    # Normalize
    input_normalized = preprocessor.scaler.transform(df_features.values)

    # Ensure correct sequence length
    if len(input_normalized) < preprocessor.sequence_length:
        # Pad with the first row if not enough data
        padding = np.repeat(
            input_normalized[0:1],
            preprocessor.sequence_length - len(input_normalized),
            axis=0
        )
        input_normalized = np.vstack([padding, input_normalized])
    elif len(input_normalized) > preprocessor.sequence_length:
        # Take last sequence_length rows
        input_normalized = input_normalized[-preprocessor.sequence_length:]

    return input_normalized


def main(args):
    """Main prediction function"""

    print("=" * 70)
    print("LSTM Weather Forecasting - Prediction Mode")
    print("=" * 70)

    # Device
    device = 'cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu'
    print(f"\nUsing device: {device}")

    # Load model and preprocessor
    print(f"\nLoading model from: {args.checkpoint_path}")
    model, preprocessor, config = load_model_and_preprocessor(
        args.checkpoint_path,
        args.config_path,
        args.preprocessor_path,
        device=device
    )

    print(f"Model loaded successfully!")
    print(f"Sequence length: {config['sequence_length']}")
    print(f"Forecast horizon: {config['forecast_horizon']}")
    print(f"Target columns: {config['target_columns']}")

    # Create forecaster
    forecaster = WeatherForecaster(model, preprocessor, device=device)

    # Prepare input
    if args.input_csv:
        print(f"\nLoading input data from: {args.input_csv}")
        input_sequence = prepare_input_from_csv(
            args.input_csv,
            preprocessor,
            n_rows=config['sequence_length']
        )
    else:
        raise ValueError("Please provide input CSV file using --input_csv")

    # Make forecast
    print("\nGenerating forecast...")

    if args.with_uncertainty:
        mean_forecast, lower_bound, upper_bound = forecaster.get_forecast_intervals(
            input_sequence,
            n_samples=args.mc_samples
        )
        forecast = mean_forecast
        confidence_intervals = (lower_bound, upper_bound)
    else:
        forecast = forecaster.forecast_from_sequence(
            input_sequence,
            return_original_scale=True
        )
        confidence_intervals = None

    # Create forecast DataFrame
    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    else:
        start_date = datetime.now()

    forecast_df = forecaster.forecast_with_dates(
        input_sequence,
        start_date,
        return_original_scale=True
    )

    # Add confidence intervals if available
    if args.with_uncertainty:
        for i, col in enumerate(config['target_columns']):
            forecast_df[f'{col}_lower_95'] = lower_bound[:, i]
            forecast_df[f'{col}_upper_95'] = upper_bound[:, i]

    # Display forecast
    print("\n" + "=" * 70)
    print("WEATHER FORECAST")
    print("=" * 70)
    print(forecast_df.to_string())

    # Save forecast
    if args.output_csv:
        forecast_df.to_csv(args.output_csv)
        print(f"\nForecast saved to: {args.output_csv}")

    # Plot forecast
    if args.plot:
        print("\nGenerating forecast plot...")
        plot_forecast_sequence(
            forecast,
            feature_names=config['target_columns'],
            confidence_intervals=confidence_intervals,
            save_path=args.plot_path
        )

    print("\n" + "=" * 70)
    print("PREDICTION COMPLETED!")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Make weather predictions using trained LSTM model'
    )

    # Model and data paths
    parser.add_argument('--checkpoint_path', type=str,
                       default='checkpoints/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--config_path', type=str,
                       default='checkpoints/config.json',
                       help='Path to model configuration')
    parser.add_argument('--preprocessor_path', type=str,
                       default='checkpoints/preprocessor.pkl',
                       help='Path to preprocessor')

    # Input/Output
    parser.add_argument('--input_csv', type=str, required=True,
                       help='Path to input CSV file with historical data')
    parser.add_argument('--output_csv', type=str,
                       default='forecast_output.csv',
                       help='Path to save forecast CSV')
    parser.add_argument('--start_date', type=str, default=None,
                       help='Start date for forecast (YYYY-MM-DD)')

    # Prediction options
    parser.add_argument('--with_uncertainty', action='store_true',
                       help='Include uncertainty estimates (confidence intervals)')
    parser.add_argument('--mc_samples', type=int, default=100,
                       help='Number of Monte Carlo samples for uncertainty')

    # Visualization
    parser.add_argument('--plot', action='store_true', default=True,
                       help='Generate forecast plot')
    parser.add_argument('--plot_path', type=str,
                       default='forecast_plot.png',
                       help='Path to save forecast plot')

    # Other
    parser.add_argument('--use_gpu', action='store_true', default=True,
                       help='Use GPU if available')

    args = parser.parse_args()

    main(args)
