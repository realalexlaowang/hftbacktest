"""Latency modeling for realistic simulation"""

import numpy as np
from numba import jit
from typing import Dict, Tuple


@jit(nopython=True)
def calculate_network_latency(
    base_latency: float,
    jitter: float,
    packet_loss_rate: float,
    random_state: np.random.RandomState
) -> float:
    """
    Calculate network latency with jitter and packet loss
    
    Args:
        base_latency: Base network latency in microseconds
        jitter: Maximum jitter in microseconds
        packet_loss_rate: Probability of packet loss requiring retransmission
        random_state: Random state for reproducibility
    
    Returns:
        Total latency in microseconds
    """
    # Base latency with jitter
    latency = base_latency + random_state.uniform(-jitter, jitter)
    
    # Simulate packet loss and retransmission
    if random_state.random() < packet_loss_rate:
        # Add retransmission delay
        latency += base_latency * 2  # RTT for retransmission
    
    return max(0, latency)


@jit(nopython=True)
def calculate_processing_latency(
    order_type: int,  # 0: market, 1: limit, 2: stop
    order_size: float,
    market_volatility: float,
    system_load: float
) -> float:
    """
    Calculate order processing latency based on order characteristics
    
    Args:
        order_type: Type of order (0: market, 1: limit, 2: stop)
        order_size: Size of the order relative to average
        market_volatility: Current market volatility (0-1)
        system_load: Exchange system load (0-1)
    
    Returns:
        Processing latency in microseconds
    """
    # Base processing time by order type
    base_times = np.array([50.0, 100.0, 150.0])  # microseconds
    base_latency = base_times[order_type]
    
    # Size factor (larger orders take longer)
    size_factor = 1.0 + 0.1 * np.log1p(order_size)
    
    # Volatility factor (higher volatility = more processing)
    volatility_factor = 1.0 + 0.5 * market_volatility
    
    # System load factor
    load_factor = 1.0 + 2.0 * system_load ** 2
    
    return base_latency * size_factor * volatility_factor * load_factor


class LatencyModel:
    """Comprehensive latency model for HFT simulation"""
    
    def __init__(self, config: Dict = None):
        """
        Initialize latency model
        
        Args:
            config: Configuration dictionary with latency parameters
        """
        config = config or {}
        
        # Network latency parameters (in microseconds)
        self.feed_base_latency = config.get('feed_base_latency', 500)  # 0.5ms
        self.feed_jitter = config.get('feed_jitter', 100)  # 0.1ms
        self.order_base_latency = config.get('order_base_latency', 1000)  # 1ms
        self.order_jitter = config.get('order_jitter', 200)  # 0.2ms
        
        # Packet loss rates
        self.feed_packet_loss = config.get('feed_packet_loss', 0.001)  # 0.1%
        self.order_packet_loss = config.get('order_packet_loss', 0.0005)  # 0.05%
        
        # Processing parameters
        self.base_processing_time = config.get('base_processing_time', 100)  # 0.1ms
        
        # Random state for reproducibility
        self.random_state = np.random.RandomState(config.get('random_seed', 42))
    
    def get_feed_latency(self) -> float:
        """Calculate latency for market data feed"""
        return calculate_network_latency(
            self.feed_base_latency,
            self.feed_jitter,
            self.feed_packet_loss,
            self.random_state
        )
    
    def get_order_latency(
        self,
        order_type: int,
        order_size: float,
        market_volatility: float = 0.5,
        system_load: float = 0.5
    ) -> Tuple[float, float]:
        """
        Calculate total order latency (send + processing)
        
        Returns:
            Tuple of (send_latency, processing_latency) in microseconds
        """
        # Network latency to send order
        send_latency = calculate_network_latency(
            self.order_base_latency,
            self.order_jitter,
            self.order_packet_loss,
            self.random_state
        )
        
        # Exchange processing latency
        processing_latency = calculate_processing_latency(
            order_type,
            order_size,
            market_volatility,
            system_load
        )
        
        return send_latency, processing_latency
    
    def get_total_roundtrip_latency(
        self,
        order_type: int,
        order_size: float,
        market_volatility: float = 0.5,
        system_load: float = 0.5
    ) -> float:
        """Calculate total roundtrip latency for an order"""
        send_latency, processing_latency = self.get_order_latency(
            order_type, order_size, market_volatility, system_load
        )
        
        # Add response latency (typically similar to send latency)
        response_latency = calculate_network_latency(
            self.order_base_latency,
            self.order_jitter,
            self.order_packet_loss,
            self.random_state
        )
        
        return send_latency + processing_latency + response_latency


class BinanceLatencyModel(LatencyModel):
    """Binance-specific latency model with realistic parameters"""
    
    def __init__(self, location: str = "tokyo"):
        """
        Initialize Binance-specific latency model
        
        Args:
            location: Server location (affects base latency)
        """
        # Base latencies by location (in microseconds)
        location_latencies = {
            "tokyo": {"feed": 500, "order": 1000},
            "singapore": {"feed": 800, "order": 1500},
            "europe": {"feed": 15000, "order": 20000},
            "us": {"feed": 50000, "order": 60000},
        }
        
        latencies = location_latencies.get(location, location_latencies["tokyo"])
        
        config = {
            'feed_base_latency': latencies["feed"],
            'feed_jitter': latencies["feed"] * 0.2,
            'order_base_latency': latencies["order"],
            'order_jitter': latencies["order"] * 0.2,
            'feed_packet_loss': 0.001,
            'order_packet_loss': 0.0005,
        }
        
        super().__init__(config)