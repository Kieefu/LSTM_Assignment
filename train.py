"""
Training pipeline for LSTM weather forecasting model
"""
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
import json
from typing import Dict, Tuple


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""

    def __init__(self, patience: int = 10, min_delta: float = 0.0,
                 verbose: bool = True):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change in monitored value to qualify as improvement
            verbose: Print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop

        Args:
            val_loss: Current validation loss

        Returns:
            True if training should stop
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

        return self.early_stop


class ModelCheckpoint:
    """Save model checkpoints during training"""

    def __init__(self, filepath: str, monitor: str = 'val_loss',
                 save_best_only: bool = True, verbose: bool = True):
        """
        Args:
            filepath: Path to save model
            monitor: Metric to monitor
            save_best_only: Only save best model
            verbose: Print messages
        """
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.verbose = verbose
        self.best_loss = float('inf')

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

    def __call__(self, model, val_loss: float, epoch: int, optimizer, history: Dict):
        """
        Save checkpoint if condition is met

        Args:
            model: Model to save
            val_loss: Current validation loss
            epoch: Current epoch
            optimizer: Optimizer state
            history: Training history
        """
        if not self.save_best_only or val_loss < self.best_loss:
            if self.verbose and val_loss < self.best_loss:
                print(f'\nValidation loss improved from {self.best_loss:.6f} to {val_loss:.6f}')
                print(f'Saving model to {self.filepath}')

            self.best_loss = min(val_loss, self.best_loss)

            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'best_loss': self.best_loss,
                'history': history
            }
            torch.save(checkpoint, self.filepath)


class WeatherLSTMTrainer:
    """Trainer for LSTM weather forecasting model"""

    def __init__(self, model, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize trainer

        Args:
            model: LSTM model
            device: Device to train on
        """
        self.model = model.to(device)
        self.device = device
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }

    def train_epoch(self, train_loader, optimizer, criterion) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc='Training')
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            output = self.model(data)

            # Calculate loss
            loss = criterion(output, target)

            # Backward pass
            loss.backward()

            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Update weights
            optimizer.step()

            # Track loss
            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})

        return total_loss / num_batches

    def validate(self, val_loader, criterion) -> float:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)

                # Forward pass
                output = self.model(data)

                # Calculate loss
                loss = criterion(output, target)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def fit(self, train_loader, val_loader,
            epochs: int = 100,
            lr: float = 0.001,
            patience: int = 15,
            checkpoint_path: str = 'checkpoints/best_model.pth') -> Dict:
        """
        Train the model

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            lr: Learning rate
            patience: Early stopping patience
            checkpoint_path: Path to save checkpoints

        Returns:
            Training history
        """
        # Loss function
        criterion = nn.MSELoss()

        # Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        # Early stopping
        early_stopping = EarlyStopping(patience=patience, verbose=True)

        # Model checkpoint
        checkpoint = ModelCheckpoint(checkpoint_path, verbose=True)

        print(f"\nTraining on device: {self.device}")
        print(f"Total epochs: {epochs}")
        print(f"Learning rate: {lr}")
        print(f"Early stopping patience: {patience}\n")

        # Training loop
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 50)

            # Train
            train_loss = self.train_epoch(train_loader, optimizer, criterion)

            # Validate
            val_loss = self.validate(val_loader, criterion)

            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rates'].append(current_lr)

            # Print metrics
            print(f"\nTrain Loss: {train_loss:.6f}")
            print(f"Val Loss: {val_loss:.6f}")
            print(f"Learning Rate: {current_lr:.6f}")

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Save checkpoint
            checkpoint(self.model, val_loss, epoch, optimizer, self.history)

            # Early stopping
            if early_stopping(val_loss):
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break

        print("\n" + "=" * 50)
        print("Training completed!")
        print("=" * 50)

        # Load best model
        self.load_checkpoint(checkpoint_path)

        return self.history

    def save_model(self, filepath: str):
        """Save model state"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history
        }, filepath)
        print(f"Model saved to {filepath}")

    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if 'history' in checkpoint:
                self.history = checkpoint['history']
            print(f"Checkpoint loaded from {filepath}")
        else:
            print(f"No checkpoint found at {filepath}")

    def save_history(self, filepath: str):
        """Save training history to JSON"""
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f"Training history saved to {filepath}")
