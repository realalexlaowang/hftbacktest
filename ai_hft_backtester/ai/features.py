"""Feature engineering for AI models"""

import numpy as np
from numba import jit
from typing import Dict, List, Tuple
import pandas as pd


@jit(nopython=True)
def calculate_price_features(
    prices: np.ndarray,
    volumes: np.ndarray,
    window_sizes: np.ndarray
) -> np.ndarray:
    """
    Calculate price-based technical features
    
    Args:
        prices: Array of prices
        volumes: Array of volumes
        window_sizes: Different window sizes for features
    
    Returns:
        Array of features
    """
    n_prices = len(prices)
    n_windows = len(window_sizes)
    features = np.zeros((n_prices, n_windows * 10))
    
    for i, window in enumerate(window_sizes):
        if window > n_prices:
            continue
            
        # Moving averages
        ma = np.convolve(prices, np.ones(window) / window, mode='valid')
        features[window-1:, i*10] = ma
        
        # Price momentum
        if window < n_prices:
            features[window:, i*10+1] = prices[window:] / prices[:-window] - 1
        
        # Volatility (rolling std)
        for j in range(window-1, n_prices):
            features[j, i*10+2] = np.std(prices[j-window+1:j+1])
        
        # Volume-weighted average price (VWAP)
        for j in range(window-1, n_prices):
            vwap = np.sum(prices[j-window+1:j+1] * volumes[j-window+1:j+1]) / np.sum(volumes[j-window+1:j+1])
            features[j, i*10+3] = vwap
        
        # Price relative to VWAP
        features[window-1:, i*10+4] = prices[window-1:] / features[window-1:, i*10+3] - 1
        
        # Volume momentum
        if window < n_prices:
            features[window:, i*10+5] = volumes[window:] / volumes[:-window] - 1
    
    return features


@jit(nopython=True)
def calculate_microstructure_features(
    bid_prices: np.ndarray,
    ask_prices: np.ndarray,
    bid_volumes: np.ndarray,
    ask_volumes: np.ndarray,
    trades: np.ndarray
) -> np.ndarray:
    """
    Calculate market microstructure features
    
    Args:
        bid_prices: Best bid prices
        ask_prices: Best ask prices
        bid_volumes: Bid volumes at best price
        ask_volumes: Ask volumes at best price
        trades: Trade data (price, volume, side)
    
    Returns:
        Array of microstructure features
    """
    n_points = len(bid_prices)
    features = np.zeros((n_points, 15))
    
    # Spread features
    spreads = ask_prices - bid_prices
    features[:, 0] = spreads
    features[:, 1] = spreads / ((bid_prices + ask_prices) / 2)  # Relative spread
    
    # Volume imbalance
    total_volume = bid_volumes + ask_volumes
    features[:, 2] = (bid_volumes - ask_volumes) / total_volume
    
    # Micro-price (volume-weighted mid price)
    features[:, 3] = (bid_prices * ask_volumes + ask_prices * bid_volumes) / total_volume
    
    # Trade flow features
    if len(trades) > 0:
        # Calculate order flow imbalance over different windows
        windows = [10, 50, 100]
        for i, window in enumerate(windows):
            if window <= n_points:
                for j in range(window-1, n_points):
                    buy_volume = np.sum(trades[j-window+1:j+1, 1] * (trades[j-window+1:j+1, 2] == 1))
                    sell_volume = np.sum(trades[j-window+1:j+1, 1] * (trades[j-window+1:j+1, 2] == -1))
                    total = buy_volume + sell_volume
                    if total > 0:
                        features[j, 4+i] = (buy_volume - sell_volume) / total
    
    return features


class FeatureEngineer:
    """Advanced feature engineering for HFT AI models"""
    
    def __init__(self, config: Dict = None):
        """Initialize feature engineer with configuration"""
        self.config = config or {}
        self.window_sizes = np.array(self.config.get('window_sizes', [10, 30, 60, 300, 600]))
        self.feature_names = []
        self._build_feature_names()
    
    def _build_feature_names(self):
        """Build list of feature names for interpretability"""
        # Price features
        for window in self.window_sizes:
            self.feature_names.extend([
                f'ma_{window}',
                f'momentum_{window}',
                f'volatility_{window}',
                f'vwap_{window}',
                f'price_to_vwap_{window}',
                f'volume_momentum_{window}'
            ])
        
        # Microstructure features
        self.feature_names.extend([
            'spread',
            'relative_spread',
            'volume_imbalance',
            'micro_price',
            'order_flow_10',
            'order_flow_50',
            'order_flow_100'
        ])
        
        # Order book features
        for i in range(10):
            self.feature_names.append(f'book_level_{i}_imbalance')
        
        self.feature_names.extend([
            'total_volume_imbalance',
            'weighted_bid_price',
            'weighted_ask_price'
        ])
    
    def extract_features(
        self,
        price_data: pd.DataFrame,
        orderbook_data: Dict,
        trade_data: pd.DataFrame
    ) -> np.ndarray:
        """
        Extract all features from market data
        
        Args:
            price_data: DataFrame with OHLCV data
            orderbook_data: Dictionary with order book snapshots
            trade_data: DataFrame with individual trades
        
        Returns:
            Feature matrix
        """
        # Extract price features
        prices = price_data['close'].values
        volumes = price_data['volume'].values
        price_features = calculate_price_features(prices, volumes, self.window_sizes)
        
        # Extract microstructure features
        if 'bid' in price_data.columns and 'ask' in price_data.columns:
            micro_features = calculate_microstructure_features(
                price_data['bid'].values,
                price_data['ask'].values,
                price_data['bid_volume'].values,
                price_data['ask_volume'].values,
                trade_data.values if len(trade_data) > 0 else np.array([])
            )
        else:
            micro_features = np.zeros((len(prices), 15))
        
        # Combine all features
        features = np.hstack([price_features, micro_features])
        
        # Add order book features if available
        if orderbook_data:
            # This would be expanded with actual order book feature extraction
            pass
        
        return features
    
    def create_training_data(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        sequence_length: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for training LSTM/Transformer models
        
        Args:
            features: Feature matrix
            labels: Target labels
            sequence_length: Length of input sequences
        
        Returns:
            Tuple of (X, y) for training
        """
        n_samples = len(features) - sequence_length
        n_features = features.shape[1]
        
        X = np.zeros((n_samples, sequence_length, n_features))
        y = np.zeros((n_samples,))
        
        for i in range(n_samples):
            X[i] = features[i:i+sequence_length]
            y[i] = labels[i+sequence_length]
        
        return X, y
    
    def normalize_features(
        self,
        features: np.ndarray,
        method: str = 'standard'
    ) -> np.ndarray:
        """
        Normalize features for neural network training
        
        Args:
            features: Raw features
            method: Normalization method ('standard', 'minmax', 'robust')
        
        Returns:
            Normalized features
        """
        if method == 'standard':
            mean = np.nanmean(features, axis=0)
            std = np.nanstd(features, axis=0)
            std[std == 0] = 1  # Avoid division by zero
            return (features - mean) / std
        elif method == 'minmax':
            min_val = np.nanmin(features, axis=0)
            max_val = np.nanmax(features, axis=0)
            range_val = max_val - min_val
            range_val[range_val == 0] = 1
            return (features - min_val) / range_val
        else:
            raise ValueError(f"Unknown normalization method: {method}")


class RealtimeFeatureEngine:
    """Optimized feature extraction for live trading"""
    
    def __init__(self, feature_config: Dict):
        """Initialize with pre-computed statistics for fast normalization"""
        self.feature_means = feature_config.get('means')
        self.feature_stds = feature_config.get('stds')
        self.window_sizes = feature_config.get('window_sizes', [10, 30, 60])
        self.price_buffer = []
        self.volume_buffer = []
        self.max_buffer_size = max(self.window_sizes) * 2
    
    def update_buffers(self, price: float, volume: float):
        """Update price and volume buffers"""
        self.price_buffer.append(price)
        self.volume_buffer.append(volume)
        
        # Keep buffer size manageable
        if len(self.price_buffer) > self.max_buffer_size:
            self.price_buffer.pop(0)
            self.volume_buffer.pop(0)
    
    def get_features(self, orderbook_features: np.ndarray) -> np.ndarray:
        """Extract features for real-time prediction"""
        if len(self.price_buffer) < max(self.window_sizes):
            # Not enough data yet
            return np.zeros(len(self.feature_means))
        
        # Calculate price features
        prices = np.array(self.price_buffer)
        volumes = np.array(self.volume_buffer)
        price_features = calculate_price_features(
            prices[-max(self.window_sizes):],
            volumes[-max(self.window_sizes):],
            np.array(self.window_sizes)
        )
        
        # Take the last row of features
        latest_features = price_features[-1]
        
        # Combine with orderbook features
        all_features = np.hstack([latest_features, orderbook_features])
        
        # Normalize
        normalized = (all_features - self.feature_means) / self.feature_stds
        
        return normalized