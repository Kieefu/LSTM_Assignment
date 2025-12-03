"""
LSTM model architecture for weather forecasting
"""
import torch
import torch.nn as nn


class WeatherLSTM(nn.Module):
    """LSTM model for multi-step weather forecasting"""

    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, output_size: int = 1,
                 forecast_horizon: int = 7, dropout: float = 0.2):
        """
        Initialize LSTM model

        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units in LSTM
            num_layers: Number of LSTM layers
            output_size: Number of output features (target variables)
            forecast_horizon: Number of time steps to forecast
            dropout: Dropout probability
        """
        super(WeatherLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.forecast_horizon = forecast_horizon

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Fully connected layer to produce forecast
        self.fc = nn.Linear(hidden_size, output_size * forecast_horizon)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

        nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Output tensor of shape (batch_size, forecast_horizon, output_size)
        """
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)

        # Use the last hidden state
        last_hidden = lstm_out[:, -1, :]

        # Apply dropout
        last_hidden = self.dropout(last_hidden)

        # Generate forecast
        output = self.fc(last_hidden)

        # Reshape to (batch_size, forecast_horizon, output_size)
        output = output.view(-1, self.forecast_horizon, self.output_size)

        return output

    def predict(self, x, device='cpu'):
        """
        Make predictions (inference mode)

        Args:
            x: Input tensor
            device: Device to run prediction on

        Returns:
            Predictions
        """
        self.eval()
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.FloatTensor(x)
            x = x.to(device)
            predictions = self.forward(x)
        return predictions.cpu().numpy()


class AttentionLSTM(nn.Module):
    """LSTM with attention mechanism for weather forecasting"""

    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, output_size: int = 1,
                 forecast_horizon: int = 7, dropout: float = 0.2):
        """
        Initialize Attention LSTM model

        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units in LSTM
            num_layers: Number of LSTM layers
            output_size: Number of output features
            forecast_horizon: Number of time steps to forecast
            dropout: Dropout probability
        """
        super(AttentionLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.forecast_horizon = forecast_horizon

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Attention mechanism
        self.attention = nn.Linear(hidden_size, 1)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Output layer
        self.fc = nn.Linear(hidden_size, output_size * forecast_horizon)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

        nn.init.xavier_uniform_(self.attention.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        self.attention.bias.data.fill_(0)
        self.fc.bias.data.fill_(0)

    def forward(self, x):
        """
        Forward pass with attention

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Output tensor of shape (batch_size, forecast_horizon, output_size)
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_size)

        # Attention weights
        attention_weights = torch.softmax(
            self.attention(lstm_out).squeeze(-1), dim=1
        )  # (batch_size, seq_len)

        # Apply attention
        context = torch.bmm(
            attention_weights.unsqueeze(1),
            lstm_out
        ).squeeze(1)  # (batch_size, hidden_size)

        # Dropout
        context = self.dropout(context)

        # Generate forecast
        output = self.fc(context)

        # Reshape
        output = output.view(-1, self.forecast_horizon, self.output_size)

        return output

    def predict(self, x, device='cpu'):
        """Make predictions"""
        self.eval()
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.FloatTensor(x)
            x = x.to(device)
            predictions = self.forward(x)
        return predictions.cpu().numpy()


def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model):
    """Print model summary"""
    print("=" * 70)
    print(f"Model: {model.__class__.__name__}")
    print("=" * 70)
    print(f"Total parameters: {count_parameters(model):,}")
    print("=" * 70)
    print("\nModel architecture:")
    print(model)
    print("=" * 70)
