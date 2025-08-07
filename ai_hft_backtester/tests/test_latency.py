"""Unit tests for latency module"""

import numpy as np
import pytest
from ..core.latency import (
    calculate_network_latency,
    calculate_processing_latency,
    LatencyModel,
    BinanceLatencyModel
)


class TestLatencyFunctions:
    """Test latency calculation functions"""
    
    def test_calculate_network_latency_basic(self):
        """Test basic network latency calculation"""
        base_latency = 1000.0  # 1ms
        jitter = 100.0  # 0.1ms
        packet_loss_rate = 0.0  # No packet loss
        random_state = np.random.RandomState(42)
        
        latency = calculate_network_latency(
            base_latency, jitter, packet_loss_rate, random_state
        )
        
        # Latency should be within base +/- jitter
        assert base_latency - jitter <= latency <= base_latency + jitter
    
    def test_calculate_network_latency_with_packet_loss(self):
        """Test network latency with packet loss"""
        base_latency = 1000.0
        jitter = 100.0
        packet_loss_rate = 1.0  # 100% packet loss
        random_state = np.random.RandomState(42)
        
        latency = calculate_network_latency(
            base_latency, jitter, packet_loss_rate, random_state
        )
        
        # With packet loss, latency should include retransmission
        assert latency >= base_latency * 3 - jitter  # RTT for retransmission
    
    def test_calculate_processing_latency_market_order(self):
        """Test processing latency for market order"""
        order_type = 0  # Market order
        order_size = 1.0
        market_volatility = 0.5
        system_load = 0.5
        
        latency = calculate_processing_latency(
            order_type, order_size, market_volatility, system_load
        )
        
        # Market orders have base latency of 50 microseconds
        assert latency >= 50.0
        
    def test_calculate_processing_latency_scaling(self):
        """Test processing latency scaling with parameters"""
        # Large order should take longer
        latency_small = calculate_processing_latency(1, 0.1, 0.5, 0.5)
        latency_large = calculate_processing_latency(1, 10.0, 0.5, 0.5)
        assert latency_large > latency_small
        
        # High volatility should increase latency
        latency_low_vol = calculate_processing_latency(1, 1.0, 0.1, 0.5)
        latency_high_vol = calculate_processing_latency(1, 1.0, 0.9, 0.5)
        assert latency_high_vol > latency_low_vol
        
        # High system load should increase latency
        latency_low_load = calculate_processing_latency(1, 1.0, 0.5, 0.1)
        latency_high_load = calculate_processing_latency(1, 1.0, 0.5, 0.9)
        assert latency_high_load > latency_low_load


class TestLatencyModel:
    """Test LatencyModel class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.config = {
            'feed_base_latency': 500,
            'feed_jitter': 100,
            'order_base_latency': 1000,
            'order_jitter': 200,
            'feed_packet_loss': 0.001,
            'order_packet_loss': 0.0005,
            'random_seed': 42
        }
        self.model = LatencyModel(self.config)
    
    def test_initialization(self):
        """Test model initialization"""
        assert self.model.feed_base_latency == 500
        assert self.model.feed_jitter == 100
        assert self.model.order_base_latency == 1000
        assert self.model.order_jitter == 200
    
    def test_get_feed_latency(self):
        """Test feed latency calculation"""
        latencies = [self.model.get_feed_latency() for _ in range(100)]
        
        # Check range
        min_expected = self.config['feed_base_latency'] - self.config['feed_jitter']
        max_expected = self.config['feed_base_latency'] + self.config['feed_jitter']
        
        # Most latencies should be within normal range
        normal_latencies = [l for l in latencies if min_expected <= l <= max_expected]
        assert len(normal_latencies) > 90  # At least 90% should be normal
    
    def test_get_order_latency(self):
        """Test order latency calculation"""
        send_latency, proc_latency = self.model.get_order_latency(
            order_type=1,  # Limit order
            order_size=1.0,
            market_volatility=0.5,
            system_load=0.5
        )
        
        assert send_latency > 0
        assert proc_latency > 0
        
        # Send latency should be around base +/- jitter
        min_send = self.config['order_base_latency'] - self.config['order_jitter']
        max_send = self.config['order_base_latency'] + self.config['order_jitter'] * 3
        assert min_send <= send_latency <= max_send
    
    def test_get_total_roundtrip_latency(self):
        """Test total roundtrip latency calculation"""
        total_latency = self.model.get_total_roundtrip_latency(
            order_type=0,  # Market order
            order_size=1.0,
            market_volatility=0.3,
            system_load=0.3
        )
        
        # Total should be at least send + processing + response
        min_expected = (
            self.config['order_base_latency'] * 2 +  # Send + response
            50.0  # Minimum processing for market order
        )
        assert total_latency >= min_expected


class TestBinanceLatencyModel:
    """Test Binance-specific latency model"""
    
    def test_tokyo_location(self):
        """Test Tokyo location latency"""
        model = BinanceLatencyModel(location="tokyo")
        
        assert model.feed_base_latency == 500
        assert model.order_base_latency == 1000
        
        # Test actual latency generation
        feed_latency = model.get_feed_latency()
        assert 0 < feed_latency < 2000  # Reasonable range
    
    def test_different_locations(self):
        """Test different geographical locations"""
        locations = {
            "tokyo": {"feed": 500, "order": 1000},
            "singapore": {"feed": 800, "order": 1500},
            "europe": {"feed": 15000, "order": 20000},
            "us": {"feed": 50000, "order": 60000}
        }
        
        for location, expected in locations.items():
            model = BinanceLatencyModel(location=location)
            assert model.feed_base_latency == expected["feed"]
            assert model.order_base_latency == expected["order"]
    
    def test_default_location(self):
        """Test default location fallback"""
        model = BinanceLatencyModel(location="unknown")
        
        # Should default to Tokyo
        assert model.feed_base_latency == 500
        assert model.order_base_latency == 1000
    
    @pytest.mark.parametrize("location,order_type,expected_range", [
        ("tokyo", 0, (1000, 5000)),      # Market order from Tokyo
        ("tokyo", 1, (1000, 5000)),      # Limit order from Tokyo
        ("us", 0, (50000, 150000)),     # Market order from US
        ("us", 1, (50000, 150000)),     # Limit order from US
    ])
    def test_realistic_latencies(self, location, order_type, expected_range):
        """Test realistic latency ranges for different scenarios"""
        model = BinanceLatencyModel(location=location)
        
        total_latency = model.get_total_roundtrip_latency(
            order_type=order_type,
            order_size=1.0,
            market_volatility=0.5,
            system_load=0.5
        )
        
        min_latency, max_latency = expected_range
        assert min_latency <= total_latency <= max_latency


def test_latency_consistency():
    """Test that latency calculations are consistent with same seed"""
    config = {'random_seed': 42}
    model1 = LatencyModel(config)
    model2 = LatencyModel(config)
    
    # Should produce same sequence of latencies
    latencies1 = [model1.get_feed_latency() for _ in range(10)]
    latencies2 = [model2.get_feed_latency() for _ in range(10)]
    
    assert latencies1 == latencies2