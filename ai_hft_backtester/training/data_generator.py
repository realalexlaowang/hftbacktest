"""Generate training data from historical market data"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from numba import jit
import h5py
from datetime import datetime, timedelta

from ..core.orderbook import OrderBook
from ..ai.features import FeatureEngineer


@jit(nopython=True)
def calculate_labels(
    prices: np.ndarray,
    horizons: np.ndarray,
    threshold: float = 0.0002
) -> np.ndarray:
    """
    Calculate price movement labels for different prediction horizons
    
    Args:
        prices: Array of prices
        horizons: Prediction horizons in ticks
        threshold: Minimum price change threshold for directional labels
    
    Returns:
        Labels array (n_samples, n_horizons, 3) for [down, neutral, up]
    """
    n_samples = len(prices)
    n_horizons = len(horizons)
    labels = np.zeros((n_samples, n_horizons, 3))
    
    for i in range(n_samples):
        for j, horizon in enumerate(horizons):
            if i + horizon < n_samples:
                future_price = prices[i + horizon]
                current_price = prices[i]
                price_change = (future_price - current_price) / current_price
                
                if price_change < -threshold:
                    labels[i, j, 0] = 1  # Down
                elif price_change > threshold:
                    labels[i, j, 2] = 1  # Up
                else:
                    labels[i, j, 1] = 1  # Neutral
            else:
                labels[i, j, 1] = 1  # Default to neutral at end
    
    return labels


class TrainingDataGenerator:
    """Generate labeled training data for AI models"""
    
    def __init__(
        self,
        feature_engineer: FeatureEngineer,
        label_config: Dict = None
    ):
        """
        Initialize data generator
        
        Args:
            feature_engineer: Feature extraction engine
            label_config: Label generation configuration
        """
        self.feature_engineer = feature_engineer
        self.label_config = label_config or {}
        
        # Label generation parameters
        self.prediction_horizons = self.label_config.get(
            'horizons', [10, 50, 300, 600]  # 1s, 5s, 30s, 1m at 10Hz
        )
        self.price_threshold = self.label_config.get('threshold', 0.0002)  # 2 bps
        
        # Data storage
        self.feature_buffer = []
        self.label_buffer = []
        self.metadata_buffer = []
    
    def process_orderbook_snapshot(
        self,
        orderbook: OrderBook,
        timestamp: int,
        mid_price: float
    ):
        """
        Process single orderbook snapshot
        
        Args:
            orderbook: Current orderbook state
            timestamp: Snapshot timestamp
            mid_price: Current mid price
        """
        # Extract features
        features = orderbook.get_features()
        
        # Store with metadata
        self.feature_buffer.append(features)
        self.metadata_buffer.append({
            'timestamp': timestamp,
            'mid_price': mid_price,
            'spread': orderbook.asks[0, 0] - orderbook.bids[0, 0] if len(orderbook.bids) > 0 and len(orderbook.asks) > 0 else np.nan
        })
    
    def generate_labels(self):
        """Generate labels for collected features"""
        if not self.metadata_buffer:
            return
        
        # Extract prices
        prices = np.array([m['mid_price'] for m in self.metadata_buffer])
        
        # Calculate labels
        labels = calculate_labels(
            prices,
            np.array(self.prediction_horizons),
            self.price_threshold
        )
        
        self.label_buffer = labels
    
    def create_training_dataset(
        self,
        output_path: str,
        sequence_length: int = 100,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1
    ) -> Dict[str, str]:
        """
        Create HDF5 training dataset
        
        Args:
            output_path: Path to save dataset
            sequence_length: Length of input sequences
            train_ratio: Training data ratio
            val_ratio: Validation data ratio
        
        Returns:
            Dictionary with paths to train/val/test files
        """
        # Generate labels if not done
        if not self.label_buffer:
            self.generate_labels()
        
        # Convert to arrays
        features = np.array(self.feature_buffer)
        labels = np.array(self.label_buffer)
        
        # Create sequences
        n_samples = len(features) - sequence_length
        n_features = features.shape[1]
        
        X = np.zeros((n_samples, sequence_length, n_features))
        y = np.zeros((n_samples, len(self.prediction_horizons), 3))
        
        for i in range(n_samples):
            X[i] = features[i:i+sequence_length]
            y[i] = labels[i+sequence_length-1]  # Label at last timestep
        
        # Split data
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        
        train_idx = slice(0, n_train)
        val_idx = slice(n_train, n_train + n_val)
        test_idx = slice(n_train + n_val, None)
        
        # Save datasets
        datasets = {
            'train': (X[train_idx], y[train_idx]),
            'val': (X[val_idx], y[val_idx]),
            'test': (X[test_idx], y[test_idx])
        }
        
        paths = {}
        for split, (X_split, y_split) in datasets.items():
            filepath = f"{output_path}_{split}.h5"
            with h5py.File(filepath, 'w') as f:
                f.create_dataset('features', data=X_split, compression='gzip')
                f.create_dataset('labels', data=y_split, compression='gzip')
                f.attrs['sequence_length'] = sequence_length
                f.attrs['n_features'] = n_features
                f.attrs['prediction_horizons'] = self.prediction_horizons
            paths[split] = filepath
        
        return paths
    
    def add_market_regime_labels(self):
        """Add market regime classification labels"""
        if not self.metadata_buffer:
            return
        
        prices = np.array([m['mid_price'] for m in self.metadata_buffer])
        spreads = np.array([m['spread'] for m in self.metadata_buffer])
        
        # Calculate volatility regimes
        returns = np.diff(np.log(prices))
        volatility = pd.Series(returns).rolling(100).std()
        
        # Define regimes
        vol_percentiles = np.percentile(volatility.dropna(), [33, 67])
        
        regime_labels = []
        for vol in volatility:
            if pd.isna(vol):
                regime_labels.append(1)  # Normal
            elif vol < vol_percentiles[0]:
                regime_labels.append(0)  # Low volatility
            elif vol > vol_percentiles[1]:
                regime_labels.append(2)  # High volatility
            else:
                regime_labels.append(1)  # Normal
        
        # Add to metadata
        for i, regime in enumerate(regime_labels):
            if i < len(self.metadata_buffer):
                self.metadata_buffer[i]['market_regime'] = regime


class ReinforcementLearningDataGenerator:
    """Generate experience replay data for RL training"""
    
    def __init__(self, buffer_size: int = 1000000):
        """
        Initialize RL data generator
        
        Args:
            buffer_size: Maximum replay buffer size
        """
        self.buffer_size = buffer_size
        self.buffer = []
        self.position = 0
    
    def add_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """
        Add experience to replay buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode termination flag
        """
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        }
        
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
            self.position = (self.position + 1) % self.buffer_size
    
    def sample_batch(self, batch_size: int) -> Dict[str, np.ndarray]:
        """
        Sample random batch from replay buffer
        
        Args:
            batch_size: Number of experiences to sample
        
        Returns:
            Dictionary with batched experiences
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        batch = {
            'states': np.array([self.buffer[i]['state'] for i in indices]),
            'actions': np.array([self.buffer[i]['action'] for i in indices]),
            'rewards': np.array([self.buffer[i]['reward'] for i in indices]),
            'next_states': np.array([self.buffer[i]['next_state'] for i in indices]),
            'dones': np.array([self.buffer[i]['done'] for i in indices])
        }
        
        return batch
    
    def calculate_returns(self, gamma: float = 0.99) -> np.ndarray:
        """
        Calculate discounted returns for all experiences
        
        Args:
            gamma: Discount factor
        
        Returns:
            Array of discounted returns
        """
        returns = np.zeros(len(self.buffer))
        running_return = 0
        
        # Calculate returns backwards
        for i in reversed(range(len(self.buffer))):
            if self.buffer[i]['done']:
                running_return = 0
            running_return = self.buffer[i]['reward'] + gamma * running_return
            returns[i] = running_return
        
        return returns


class StreamingDataGenerator:
    """Generate training data in streaming fashion for online learning"""
    
    def __init__(
        self,
        feature_engineer: FeatureEngineer,
        window_size: int = 1000,
        update_frequency: int = 100
    ):
        """
        Initialize streaming data generator
        
        Args:
            feature_engineer: Feature extraction engine
            window_size: Size of sliding window
            update_frequency: How often to generate new training batch
        """
        self.feature_engineer = feature_engineer
        self.window_size = window_size
        self.update_frequency = update_frequency
        
        self.feature_window = []
        self.price_window = []
        self.tick_count = 0
    
    def process_tick(
        self,
        orderbook: OrderBook,
        mid_price: float,
        timestamp: int
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Process new tick and potentially generate training data
        
        Args:
            orderbook: Current orderbook
            mid_price: Current mid price
            timestamp: Tick timestamp
        
        Returns:
            Optional tuple of (features, labels) if update triggered
        """
        # Extract features
        features = orderbook.get_features()
        
        # Update windows
        self.feature_window.append(features)
        self.price_window.append(mid_price)
        
        # Maintain window size
        if len(self.feature_window) > self.window_size:
            self.feature_window.pop(0)
            self.price_window.pop(0)
        
        self.tick_count += 1
        
        # Check if should generate training data
        if self.tick_count % self.update_frequency == 0 and len(self.feature_window) == self.window_size:
            # Generate features and labels
            X = np.array(self.feature_window[:-10])  # Leave last 10 for labels
            prices = np.array(self.price_window)
            
            # Simple label: price change in next 10 ticks
            current_price = prices[-11]
            future_price = prices[-1]
            price_change = (future_price - current_price) / current_price
            
            if price_change > 0.0002:
                label = 2  # Up
            elif price_change < -0.0002:
                label = 0  # Down
            else:
                label = 1  # Neutral
            
            return X, label
        
        return None