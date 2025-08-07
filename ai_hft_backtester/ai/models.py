"""AI models for HFT predictions"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional


class LSTMPredictor(nn.Module):
    """LSTM model for price movement prediction"""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 3  # [down, neutral, up]
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # bidirectional
            num_heads=8,
            dropout=dropout
        )
        
        # Output layers
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
        
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Apply attention
        # Reshape for attention: (seq_len, batch, features)
        lstm_out = lstm_out.transpose(0, 1)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = attn_out.transpose(0, 1)
        
        # Take the last timestep
        last_out = attn_out[:, -1, :]
        
        # Dense layers
        out = F.relu(self.fc1(last_out))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class TransformerPredictor(nn.Module):
    """Transformer model for multi-horizon prediction"""
    
    def __init__(
        self,
        input_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        max_seq_length: int = 1000
    ):
        super().__init__()
        
        # Input embedding
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.positional_encoding = self._create_positional_encoding(max_seq_length, d_model)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output heads for different prediction horizons
        self.prediction_heads = nn.ModuleDict({
            '1s': nn.Linear(d_model, 3),   # 1 second
            '5s': nn.Linear(d_model, 3),   # 5 seconds
            '30s': nn.Linear(d_model, 3),  # 30 seconds
            '1m': nn.Linear(d_model, 3),   # 1 minute
        })
        
        self.dropout = nn.Dropout(dropout)
        
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encoding"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            mask: Optional attention mask
        
        Returns:
            Dictionary of predictions for different horizons
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input to model dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.positional_encoding[:, :seq_len, :].to(x.device)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Transformer expects (seq_len, batch, features)
        x = x.transpose(0, 1)
        
        # Pass through transformer
        transformer_out = self.transformer(x, src_key_padding_mask=mask)
        
        # Take the last timestep
        last_out = transformer_out[-1, :, :]
        
        # Generate predictions for different horizons
        predictions = {}
        for horizon, head in self.prediction_heads.items():
            predictions[horizon] = head(last_out)
        
        return predictions


class MarketMakingPolicy(nn.Module):
    """Deep RL policy for market making decisions"""
    
    def __init__(
        self,
        state_size: int,
        hidden_size: int = 256,
        n_actions: int = 25  # Different spread/size combinations
    ):
        super().__init__()
        
        # Feature extraction
        self.feature_net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Dueling DQN architecture
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, n_actions)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Q-value estimation
        
        Args:
            state: Current market state
        
        Returns:
            Q-values for each action
        """
        features = self.feature_net(state)
        
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine value and advantages (Dueling DQN)
        q_values = value + advantages - advantages.mean(dim=1, keepdim=True)
        
        return q_values
    
    def get_action(
        self,
        state: torch.Tensor,
        epsilon: float = 0.0
    ) -> Tuple[int, Dict[str, float]]:
        """
        Get action with epsilon-greedy exploration
        
        Returns:
            Tuple of (action_idx, action_params)
        """
        if np.random.random() < epsilon:
            # Random action
            action_idx = np.random.randint(0, 25)
        else:
            # Greedy action
            with torch.no_grad():
                q_values = self.forward(state)
                action_idx = q_values.argmax().item()
        
        # Decode action to spread and size
        spread_idx = action_idx // 5
        size_idx = action_idx % 5
        
        spreads = [0.0001, 0.0002, 0.0005, 0.001, 0.002]  # Different spread levels
        sizes = [0.1, 0.2, 0.5, 1.0, 2.0]  # Different order sizes
        
        action_params = {
            'spread': spreads[spread_idx],
            'size': sizes[size_idx]
        }
        
        return action_idx, action_params


def save_model(model: nn.Module, filepath: str, metadata: Dict = None):
    """
    Save model with metadata
    
    Args:
        model: PyTorch model to save
        filepath: Path to save file
        metadata: Optional metadata to save with model
    """
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
        'metadata': metadata or {}
    }
    torch.save(save_dict, filepath)


def load_model(filepath: str, model_class, **kwargs) -> Tuple[nn.Module, Dict]:
    """
    Load model from file
    
    Args:
        filepath: Path to saved model
        model_class: Model class to instantiate
        **kwargs: Arguments for model initialization
    
    Returns:
        Tuple of (model, metadata)
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    model = model_class(**kwargs)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint.get('metadata', {})


class EnsemblePredictor:
    """Ensemble of multiple models for robust predictions"""
    
    def __init__(self, models: Dict[str, nn.Module], weights: Optional[Dict[str, float]] = None):
        """
        Initialize ensemble
        
        Args:
            models: Dictionary of model_name -> model
            weights: Optional weights for each model
        """
        self.models = models
        self.weights = weights or {name: 1.0 / len(models) for name in models}
        
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get ensemble prediction
        
        Args:
            x: Input features
        
        Returns:
            Weighted average of predictions
        """
        predictions = []
        
        for name, model in self.models.items():
            model.eval()
            with torch.no_grad():
                pred = model(x)
                if isinstance(pred, dict):
                    # For multi-horizon models, use shortest horizon
                    pred = pred['1s']
                predictions.append(pred * self.weights[name])
        
        # Weighted average
        ensemble_pred = torch.stack(predictions).sum(dim=0)
        
        return F.softmax(ensemble_pred, dim=-1)
    
    def update_weights(self, performance_metrics: Dict[str, float]):
        """
        Update model weights based on recent performance
        
        Args:
            performance_metrics: Dictionary of model_name -> performance_score
        """
        # Normalize scores to sum to 1
        total_score = sum(performance_metrics.values())
        if total_score > 0:
            self.weights = {
                name: score / total_score 
                for name, score in performance_metrics.items()
            }