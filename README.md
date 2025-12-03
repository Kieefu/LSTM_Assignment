# LSTM Weather Forecasting

A comprehensive PyTorch-based LSTM model for weather forecasting using historical weather data. This project includes complete data preprocessing, model training, evaluation, and forecasting capabilities.

## Features

- **Advanced LSTM Architecture**: Standard LSTM and Attention-LSTM models for time series forecasting
- **Comprehensive Data Pipeline**: Automated data preprocessing, sequence generation, and normalization
- **Robust Training**: Early stopping, learning rate scheduling, and model checkpointing
- **Detailed Evaluation**: Multiple metrics (RMSE, MAE, MAPE, R²) with per-horizon analysis
- **Uncertainty Quantification**: Monte Carlo dropout for confidence intervals
- **Rich Visualizations**: Training history, predictions vs actual, residual analysis, and more
- **Easy Deployment**: Simple prediction script for making forecasts with trained models

## Project Structure

```
LSTM_Assignment/
├── data_preprocessing.py   # Data loading, preprocessing, and sequence preparation
├── model.py               # LSTM model architectures
├── train.py              # Training pipeline with early stopping
├── evaluate.py           # Evaluation metrics and analysis
├── forecast.py           # Forecasting and prediction functions
├── visualize.py          # Visualization utilities
├── main.py              # Main pipeline script
├── predict.py           # Prediction script for trained models
├── requirements.txt     # Project dependencies
├── cleaned_weather.csv  # Your weather dataset
└── README.md           # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Kieefu/LSTM_Assignment.git
cd LSTM_Assignment
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Format

Your dataset (`cleaned_weather.csv`) should be a CSV file with:
- One row per time step (e.g., daily weather data)
- Numeric columns for weather features (temperature, humidity, pressure, etc.)
- Optional date/time column for tracking

Example:
```csv
date,temperature,humidity,pressure,wind_speed
2020-01-01,15.5,65,1013.2,5.2
2020-01-02,16.2,62,1012.8,4.8
...
```

## Quick Start

### 1. Train a Model

Train the model using your weather data:

```bash
python main.py --data_path cleaned_weather.csv \
               --sequence_length 30 \
               --forecast_horizon 7 \
               --epochs 100
```

This will:
- Load and preprocess your data
- Create train/validation/test splits
- Train an LSTM model with early stopping
- Evaluate the model on test data
- Generate visualizations
- Save the model checkpoint

### 2. Make Predictions

Use a trained model to make forecasts:

```bash
python predict.py --input_csv cleaned_weather.csv \
                  --output_csv forecast_output.csv \
                  --with_uncertainty \
                  --plot
```

## Detailed Usage

### Training Options

```bash
python main.py [OPTIONS]
```

**Data Options:**
- `--data_path`: Path to weather CSV file (default: `cleaned_weather.csv`)
- `--target_columns`: Columns to predict (e.g., `temperature humidity`)
- `--exclude_columns`: Columns to exclude from features
- `--sequence_length`: Past days for prediction (default: 30)
- `--forecast_horizon`: Future days to predict (default: 7)
- `--train_ratio`: Training data ratio (default: 0.7)
- `--val_ratio`: Validation data ratio (default: 0.15)

**Model Options:**
- `--model_type`: Model architecture (`lstm` or `attention_lstm`, default: `lstm`)
- `--hidden_size`: LSTM hidden units (default: 128)
- `--num_layers`: Number of LSTM layers (default: 2)
- `--dropout`: Dropout rate (default: 0.2)

**Training Options:**
- `--batch_size`: Batch size (default: 32)
- `--epochs`: Maximum epochs (default: 100)
- `--learning_rate`: Learning rate (default: 0.001)
- `--patience`: Early stopping patience (default: 15)

**Other Options:**
- `--skip_training`: Load from checkpoint instead of training
- `--checkpoint_path`: Model checkpoint path (default: `checkpoints/best_model.pth`)
- `--use_gpu`: Use GPU if available
- `--seed`: Random seed (default: 42)
- `--plot`: Generate visualizations

### Prediction Options

```bash
python predict.py [OPTIONS]
```

**Required:**
- `--input_csv`: CSV file with historical data for forecasting

**Optional:**
- `--checkpoint_path`: Model checkpoint (default: `checkpoints/best_model.pth`)
- `--config_path`: Model config (default: `checkpoints/config.json`)
- `--preprocessor_path`: Preprocessor (default: `checkpoints/preprocessor.pkl`)
- `--output_csv`: Save forecast to CSV (default: `forecast_output.csv`)
- `--start_date`: Forecast start date (YYYY-MM-DD)
- `--with_uncertainty`: Include confidence intervals
- `--mc_samples`: MC samples for uncertainty (default: 100)
- `--plot`: Generate forecast plot

## Examples

### Example 1: Train with Specific Targets

Train a model to predict temperature and humidity:

```bash
python main.py --data_path cleaned_weather.csv \
               --target_columns temperature humidity \
               --sequence_length 60 \
               --forecast_horizon 14 \
               --hidden_size 256 \
               --num_layers 3
```

### Example 2: Use Attention LSTM

Train with attention mechanism:

```bash
python main.py --data_path cleaned_weather.csv \
               --model_type attention_lstm \
               --hidden_size 128 \
               --epochs 150
```

### Example 3: Make Predictions with Uncertainty

Generate forecasts with 95% confidence intervals:

```bash
python predict.py --input_csv cleaned_weather.csv \
                  --with_uncertainty \
                  --mc_samples 200 \
                  --start_date 2024-01-01 \
                  --output_csv predictions.csv
```

### Example 4: Resume Training

Load a checkpoint and continue training:

```bash
python main.py --checkpoint_path checkpoints/best_model.pth
```

## Model Architecture

### Standard LSTM
- Multi-layer LSTM with configurable hidden size
- Dropout for regularization
- Fully connected output layer
- Supports multi-step forecasting

### Attention LSTM
- LSTM with attention mechanism
- Learns to focus on relevant time steps
- Better for long sequences
- Slightly slower but often more accurate

## Outputs

After training, you'll find:

**Checkpoints:**
- `checkpoints/best_model.pth`: Best model weights
- `checkpoints/preprocessor.pkl`: Fitted data preprocessor
- `checkpoints/config.json`: Model configuration

**Results:**
- `results/training_history.json`: Training metrics
- `results/evaluation_results.npz`: Evaluation results
- `results/evaluation_results_horizon_metrics.csv`: Per-horizon metrics
- `results/evaluation_results_accuracy_by_horizon.csv`: Accuracy degradation

**Visualizations:**
- `results/figures/training_history.png`: Loss curves
- `results/figures/predictions_vs_actual.png`: Scatter plots
- `results/figures/residual_analysis.png`: Error analysis
- `results/figures/time_series_comparison.png`: Forecast sequences
- `results/figures/metrics_by_horizon.png`: Performance by day

## Evaluation Metrics

The model is evaluated using:
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **R²**: Coefficient of determination
- **Explained Variance**: Proportion of variance explained

Metrics are calculated:
- Overall (all forecasts)
- Per forecast horizon (day 1, day 2, ..., day N)
- Per feature (if multiple targets)

## Advanced Features

### Monte Carlo Dropout

Get prediction uncertainty estimates:

```python
from forecast import WeatherForecaster

forecaster = WeatherForecaster(model, preprocessor)
mean, lower, upper = forecaster.get_forecast_intervals(
    input_sequence,
    n_samples=100
)
```

### Multi-Step Forecasting

Make predictions beyond the training horizon:

```python
long_forecast = forecaster.multi_step_forecast(
    initial_sequence,
    n_steps=30  # Predict 30 days ahead
)
```

### Batch Predictions

Process multiple sequences efficiently:

```python
forecasts = forecaster.batch_forecast(sequences)
```

## Tips for Best Results

1. **Data Quality**: Ensure your data is clean with no large gaps
2. **Sequence Length**: Use 30-90 days for daily weather data
3. **Normalization**: The preprocessor handles this automatically
4. **Early Stopping**: Prevents overfitting - use patience=15-20
5. **Learning Rate**: Default 0.001 works well, try 0.0001 for fine-tuning
6. **Hidden Size**: Start with 128, increase to 256 for complex patterns
7. **Validation Split**: Keep 15-20% for validation to monitor overfitting

## Troubleshooting

**Out of Memory Error:**
- Reduce `--batch_size` (try 16 or 8)
- Reduce `--hidden_size`
- Use `--use_gpu False` to train on CPU

**Poor Performance:**
- Increase `--sequence_length` (try 60 or 90)
- Increase `--hidden_size` (try 256)
- Add more layers with `--num_layers 3`
- Try `--model_type attention_lstm`
- Ensure target columns are correctly specified

**Training Too Slow:**
- Decrease `--sequence_length`
- Use smaller `--hidden_size`
- Reduce `--batch_size` if GPU memory is bottleneck

## Requirements

- Python 3.7+
- PyTorch 2.0+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- tqdm

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{lstm_weather_forecasting,
  title={LSTM Weather Forecasting},
  author={Your Name},
  year={2024},
  url={https://github.com/Kieefu/LSTM_Assignment}
}
```

## Contact

For questions or issues, please open an issue on GitHub.
