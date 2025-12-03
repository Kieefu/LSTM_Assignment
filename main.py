"""
Main script to run complete LSTM weather forecasting pipeline
"""
import argparse
import os
import json
import torch
import numpy as np
from datetime import datetime

from data_preprocessing import WeatherDataPreprocessor
from model import WeatherLSTM, AttentionLSTM, model_summary
from train import WeatherLSTMTrainer
from evaluate import ModelEvaluator
from forecast import WeatherForecaster, create_forecast_report
from visualize import (plot_training_history, plot_predictions_vs_actual,
                       plot_residuals, create_visualization_report)


def main(args):
    """Main execution function"""

    print("=" * 70)
    print("LSTM Weather Forecasting Pipeline")
    print("=" * 70)

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device configuration
    device = 'cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu'
    print(f"\nUsing device: {device}")

    # ========== 1. DATA PREPROCESSING ==========
    print("\n" + "=" * 70)
    print("STEP 1: DATA PREPROCESSING")
    print("=" * 70)

    preprocessor = WeatherDataPreprocessor(
        sequence_length=args.sequence_length,
        forecast_horizon=args.forecast_horizon
    )

    # Load data
    print(f"\nLoading data from: {args.data_path}")
    df = preprocessor.load_data(args.data_path)
    print(f"Loaded {len(df)} records")

    # Prepare features
    df_features = preprocessor.prepare_features(
        df,
        target_cols=args.target_columns,
        exclude_cols=args.exclude_columns
    )

    # Prepare data splits
    X_train, y_train, X_val, y_val, X_test, y_test = preprocessor.prepare_data(
        df_features,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio
    )

    # Create data loaders
    train_loader, val_loader = preprocessor.create_dataloaders(
        X_train, y_train, X_val, y_val,
        batch_size=args.batch_size
    )

    # Test loader for evaluation
    from torch.utils.data import DataLoader
    from data_preprocessing import WeatherDataset
    test_dataset = WeatherDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # ========== 2. MODEL CREATION ==========
    print("\n" + "=" * 70)
    print("STEP 2: MODEL CREATION")
    print("=" * 70)

    input_size = X_train.shape[2]
    output_size = y_train.shape[2]

    if args.model_type == 'lstm':
        model = WeatherLSTM(
            input_size=input_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            output_size=output_size,
            forecast_horizon=args.forecast_horizon,
            dropout=args.dropout
        )
    elif args.model_type == 'attention_lstm':
        model = AttentionLSTM(
            input_size=input_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            output_size=output_size,
            forecast_horizon=args.forecast_horizon,
            dropout=args.dropout
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    model_summary(model)

    # ========== 3. TRAINING ==========
    if not args.skip_training:
        print("\n" + "=" * 70)
        print("STEP 3: MODEL TRAINING")
        print("=" * 70)

        trainer = WeatherLSTMTrainer(model, device=device)

        history = trainer.fit(
            train_loader,
            val_loader,
            epochs=args.epochs,
            lr=args.learning_rate,
            patience=args.patience,
            checkpoint_path=args.checkpoint_path
        )

        # Save training history
        trainer.save_history(args.history_path)

        # Plot training history
        if args.plot:
            plot_training_history(
                history,
                save_path='results/figures/training_history.png'
            )
    else:
        print("\nSkipping training, loading checkpoint...")
        trainer = WeatherLSTMTrainer(model, device=device)
        trainer.load_checkpoint(args.checkpoint_path)

    # ========== 4. EVALUATION ==========
    print("\n" + "=" * 70)
    print("STEP 4: MODEL EVALUATION")
    print("=" * 70)

    evaluator = ModelEvaluator(trainer.model, device=device)
    results = evaluator.evaluate(
        test_loader,
        preprocessor=preprocessor,
        save_results=True
    )

    # Add feature names to results
    results['feature_names'] = preprocessor.target_columns

    # ========== 5. VISUALIZATION ==========
    if args.plot:
        print("\n" + "=" * 70)
        print("STEP 5: CREATING VISUALIZATIONS")
        print("=" * 70)

        create_visualization_report(results, save_dir='results/figures')

    # ========== 6. FORECASTING ==========
    print("\n" + "=" * 70)
    print("STEP 6: GENERATING FORECASTS")
    print("=" * 70)

    forecaster = WeatherForecaster(trainer.model, preprocessor, device=device)

    # Make sample forecasts
    n_samples = min(5, len(X_test))
    print(f"\nGenerating {n_samples} sample forecasts...")

    for i in range(n_samples):
        print(f"\n--- Sample Forecast {i + 1} ---")

        # Generate forecast with uncertainty
        mean_forecast, lower_bound, upper_bound = forecaster.get_forecast_intervals(
            X_test[i]
        )

        print(f"\nForecast for next {args.forecast_horizon} days:")
        for day in range(args.forecast_horizon):
            print(f"Day {day + 1}:")
            for j, col in enumerate(preprocessor.target_columns):
                actual_val = preprocessor.inverse_transform_targets(
                    y_test[i:i+1]
                )[0, day, j]
                print(f"  {col}: {mean_forecast[day, j]:.2f} "
                      f"[{lower_bound[day, j]:.2f}, {upper_bound[day, j]:.2f}] "
                      f"(Actual: {actual_val:.2f})")

    # ========== 7. SAVE FORECASTER ==========
    print("\n" + "=" * 70)
    print("STEP 7: SAVING MODEL AND CONFIGURATION")
    print("=" * 70)

    # Save preprocessor
    import pickle
    with open('checkpoints/preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    print("Preprocessor saved to checkpoints/preprocessor.pkl")

    # Save configuration
    config = {
        'sequence_length': args.sequence_length,
        'forecast_horizon': args.forecast_horizon,
        'input_size': input_size,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'output_size': output_size,
        'model_type': args.model_type,
        'target_columns': preprocessor.target_columns,
        'feature_columns': preprocessor.feature_columns
    }

    with open('checkpoints/config.json', 'w') as f:
        json.dump(config, f, indent=4)
    print("Configuration saved to checkpoints/config.json")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)

    print(f"\nModel checkpoint: {args.checkpoint_path}")
    print("Preprocessor: checkpoints/preprocessor.pkl")
    print("Configuration: checkpoints/config.json")
    print("Results: results/")
    print("Visualizations: results/figures/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='LSTM Weather Forecasting Pipeline'
    )

    # Data arguments
    parser.add_argument('--data_path', type=str, default='cleaned_weather.csv',
                       help='Path to weather data CSV file')
    parser.add_argument('--target_columns', type=str, nargs='+', default=None,
                       help='Target columns to predict (e.g., temperature humidity)')
    parser.add_argument('--exclude_columns', type=str, nargs='+', default=None,
                       help='Columns to exclude from features')
    parser.add_argument('--sequence_length', type=int, default=30,
                       help='Number of past days to use for prediction')
    parser.add_argument('--forecast_horizon', type=int, default=7,
                       help='Number of future days to predict')

    # Data split arguments
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Ratio of training data')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Ratio of validation data')

    # Model arguments
    parser.add_argument('--model_type', type=str, default='lstm',
                       choices=['lstm', 'attention_lstm'],
                       help='Type of model to use')
    parser.add_argument('--hidden_size', type=int, default=128,
                       help='Hidden size of LSTM')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')

    # Paths
    parser.add_argument('--checkpoint_path', type=str,
                       default='checkpoints/best_model.pth',
                       help='Path to save/load model checkpoint')
    parser.add_argument('--history_path', type=str,
                       default='results/training_history.json',
                       help='Path to save training history')

    # Other arguments
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training and load from checkpoint')
    parser.add_argument('--use_gpu', action='store_true', default=True,
                       help='Use GPU if available')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--plot', action='store_true', default=True,
                       help='Generate plots')

    args = parser.parse_args()

    # Create directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)

    # Run main pipeline
    main(args)
