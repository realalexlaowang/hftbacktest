"""Unit tests for feature engineering module"""

import numpy as np
import pandas as pd
import pytest
from ..ai.features import (
    calculate_price_features,
    calculate_microstructure_features,
    FeatureEngineer,
    RealtimeFeatureEngine
)


class TestPriceFeatures:
    """Test price feature calculations"""
    
    def test_calculate_price_features_basic(self):
        """Test basic price feature calculation"""
        # Create simple price and volume data
        prices = np.array([100.0, 101.0, 102.0, 101.5, 101.0])
        volumes = np.array([10.0, 15.0, 20.0, 12.0, 18.0])
        window_sizes = np.array([2, 3])
        
        features = calculate_price_features(prices, volumes, window_sizes)
        
        # Check shape
        assert features.shape == (5, 20)  # 5 prices, 2 windows * 10 features
        
        # Check moving average for window size 2
        # MA at index 1 should be (100 + 101) / 2 = 100.5
        assert features[1, 0] == 100.5
        
        # Check VWAP for window size 2 at index 1
        # VWAP = (100*10 + 101*15) / (10 + 15) = 2515 / 25 = 100.6
        assert features[1, 3] == pytest.approx(100.6)
    
    def test_calculate_price_features_momentum(self):
        """Test momentum calculation"""
        prices = np.array([100.0, 102.0, 104.0, 103.0, 105.0])
        volumes = np.ones(5) * 10.0
        window_sizes = np.array([2])
        
        features = calculate_price_features(prices, volumes, window_sizes)
        
        # Momentum at index 2: 104/100 - 1 = 0.04
        assert features[2, 1] == pytest.approx(0.04)
    
    def test_calculate_price_features_volatility(self):
        """Test volatility calculation"""
        prices = np.array([100.0, 100.0, 100.0, 100.0, 100.0])  # No volatility
        volumes = np.ones(5) * 10.0
        window_sizes = np.array([3])
        
        features = calculate_price_features(prices, volumes, window_sizes)
        
        # Volatility should be 0 for constant prices
        assert features[2, 2] == 0.0
        
        # Test with volatile prices
        prices = np.array([100.0, 102.0, 98.0, 101.0, 99.0])
        features = calculate_price_features(prices, volumes, window_sizes)
        
        # Volatility at index 2 should be std([100, 102, 98])
        expected_vol = np.std([100.0, 102.0, 98.0])
        assert features[2, 2] == pytest.approx(expected_vol)


class TestMicrostructureFeatures:
    """Test microstructure feature calculations"""
    
    def test_calculate_microstructure_features_basic(self):
        """Test basic microstructure features"""
        bid_prices = np.array([99.5, 99.6, 99.7, 99.8, 99.9])
        ask_prices = np.array([100.5, 100.4, 100.3, 100.2, 100.1])
        bid_volumes = np.array([10.0, 15.0, 20.0, 25.0, 30.0])
        ask_volumes = np.array([10.0, 15.0, 20.0, 25.0, 30.0])
        trades = np.array([])  # No trades for basic test
        
        features = calculate_microstructure_features(
            bid_prices, ask_prices, bid_volumes, ask_volumes, trades
        )
        
        # Check shape
        assert features.shape == (5, 15)
        
        # Check spread
        assert features[0, 0] == 1.0  # 100.5 - 99.5
        assert features[1, 0] == 0.8  # 100.4 - 99.6
        
        # Check relative spread
        assert features[0, 1] == pytest.approx(1.0 / 100.0)  # 1.0 / ((99.5 + 100.5) / 2)
        
        # Check volume imbalance (should be 0 for equal volumes)
        assert np.all(features[:, 2] == 0.0)
        
        # Check micro-price
        # Micro-price = (bid * ask_vol + ask * bid_vol) / (bid_vol + ask_vol)
        # For equal volumes: (99.5 * 10 + 100.5 * 10) / 20 = 100.0
        assert features[0, 3] == 100.0
    
    def test_calculate_microstructure_features_with_trades(self):
        """Test microstructure features with trade data"""
        bid_prices = np.ones(100) * 99.5
        ask_prices = np.ones(100) * 100.5
        bid_volumes = np.ones(100) * 10.0
        ask_volumes = np.ones(100) * 10.0
        
        # Create trade data: price, volume, side (1 for buy, -1 for sell)
        trades = np.array([
            [100.0, 5.0, 1],   # Buy
            [100.0, 3.0, -1],  # Sell
            [100.1, 10.0, 1],  # Buy
            [99.9, 7.0, -1],   # Sell
        ])
        
        features = calculate_microstructure_features(
            bid_prices, ask_prices, bid_volumes, ask_volumes, trades
        )
        
        # Order flow imbalance should be calculated
        # For window 10, all 4 trades are included
        # Buy volume: 5 + 10 = 15
        # Sell volume: 3 + 7 = 10
        # Imbalance: (15 - 10) / (15 + 10) = 5/25 = 0.2
        assert features[9, 4] == pytest.approx(0.2)


class TestFeatureEngineer:
    """Test FeatureEngineer class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        config = {
            'window_sizes': [10, 30, 60]
        }
        self.feature_engineer = FeatureEngineer(config)
    
    def test_initialization(self):
        """Test feature engineer initialization"""
        assert len(self.feature_engineer.window_sizes) == 3
        assert self.feature_engineer.window_sizes[0] == 10
        assert len(self.feature_engineer.feature_names) > 0
    
    def test_extract_features(self):
        """Test feature extraction"""
        # Create dummy data
        price_data = pd.DataFrame({
            'close': np.random.uniform(99, 101, 100),
            'volume': np.random.uniform(10, 20, 100),
            'bid': np.random.uniform(98.5, 100.5, 100),
            'ask': np.random.uniform(99.5, 101.5, 100),
            'bid_volume': np.random.uniform(5, 15, 100),
            'ask_volume': np.random.uniform(5, 15, 100)
        })
        
        orderbook_data = {}  # Empty for now
        trade_data = pd.DataFrame()  # Empty trades
        
        features = self.feature_engineer.extract_features(
            price_data, orderbook_data, trade_data
        )
        
        # Check shape
        assert features.shape[0] == 100  # Same as input data
        assert features.shape[1] > 0  # Should have features
    
    def test_create_training_data(self):
        """Test training data creation"""
        # Create dummy features and labels
        features = np.random.randn(200, 50)
        labels = np.random.randint(0, 3, 200)
        sequence_length = 10
        
        X, y = self.feature_engineer.create_training_data(
            features, labels, sequence_length
        )
        
        # Check shapes
        assert X.shape == (190, 10, 50)  # 200 - 10 = 190 samples
        assert y.shape == (190,)
        
        # Check that sequences are created correctly
        assert np.array_equal(X[0], features[0:10])
        assert y[0] == labels[10]
    
    def test_normalize_features(self):
        """Test feature normalization"""
        features = np.array([[1.0, 2.0, 3.0],
                           [4.0, 5.0, 6.0],
                           [7.0, 8.0, 9.0]])
        
        # Test standard normalization
        normalized = self.feature_engineer.normalize_features(features, 'standard')
        
        # Check mean is 0 and std is 1
        assert np.allclose(np.mean(normalized, axis=0), 0)
        assert np.allclose(np.std(normalized, axis=0), 1)
        
        # Test minmax normalization
        normalized = self.feature_engineer.normalize_features(features, 'minmax')
        
        # Check range is [0, 1]
        assert np.allclose(np.min(normalized, axis=0), 0)
        assert np.allclose(np.max(normalized, axis=0), 1)


class TestRealtimeFeatureEngine:
    """Test RealtimeFeatureEngine class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        feature_config = {
            'means': np.zeros(50),
            'stds': np.ones(50),
            'window_sizes': [10, 30]
        }
        self.engine = RealtimeFeatureEngine(feature_config)
    
    def test_initialization(self):
        """Test realtime engine initialization"""
        assert len(self.engine.window_sizes) == 2
        assert len(self.engine.price_buffer) == 0
        assert self.engine.max_buffer_size == 60  # max(window_sizes) * 2
    
    def test_update_buffers(self):
        """Test buffer updates"""
        # Add some data
        for i in range(100):
            self.engine.update_buffers(100.0 + i * 0.1, 10.0 + i)
        
        # Check buffer size is limited
        assert len(self.engine.price_buffer) == self.engine.max_buffer_size
        assert len(self.engine.volume_buffer) == self.engine.max_buffer_size
        
        # Check most recent values
        assert self.engine.price_buffer[-1] == 100.0 + 99 * 0.1
        assert self.engine.volume_buffer[-1] == 10.0 + 99
    
    def test_get_features(self):
        """Test feature extraction"""
        # Not enough data initially
        orderbook_features = np.zeros(10)
        features = self.engine.get_features(orderbook_features)
        assert len(features) == 50  # Should return zeros
        assert np.all(features == 0)
        
        # Add enough data
        for i in range(35):
            self.engine.update_buffers(100.0 + i * 0.1, 10.0)
        
        features = self.engine.get_features(orderbook_features)
        assert len(features) == 50
        # Features should be normalized (mean 0, std 1)
        # Since we initialized with zeros mean and ones std,
        # the features should be unchanged from raw values


@pytest.mark.parametrize("window_size,n_features", [
    ([10], 10),
    ([10, 20], 20),
    ([10, 20, 30], 30),
])
def test_feature_dimensions(window_size, n_features):
    """Test that feature dimensions match window configurations"""
    prices = np.random.uniform(99, 101, 100)
    volumes = np.random.uniform(10, 20, 100)
    window_sizes = np.array(window_size)
    
    features = calculate_price_features(prices, volumes, window_sizes)
    
    # Each window creates 10 features
    assert features.shape[1] == n_features