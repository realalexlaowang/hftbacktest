"""Unit tests for orderbook module"""

import numpy as np
import pytest
from ..core.orderbook import OrderBook, update_orderbook, calculate_orderbook_features


class TestOrderBook:
    """Test cases for OrderBook class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.orderbook = OrderBook("BTCUSDT")
    
    def test_initialization(self):
        """Test orderbook initialization"""
        assert self.orderbook.symbol == "BTCUSDT"
        assert len(self.orderbook.bids) == 0
        assert len(self.orderbook.asks) == 0
        assert self.orderbook.last_update_time == 0
        assert self.orderbook.sequence_number == 0
    
    def test_update_bids(self):
        """Test bid updates"""
        updates = [
            {'side': 'bid', 'price': 50000.0, 'quantity': 1.0},
            {'side': 'bid', 'price': 49999.0, 'quantity': 2.0},
            {'side': 'bid', 'price': 50001.0, 'quantity': 0.5}
        ]
        
        self.orderbook.update(updates, timestamp=1000)
        
        # Check bids are sorted correctly (descending)
        assert len(self.orderbook.bids) == 3
        assert self.orderbook.bids[0, 0] == 50001.0  # Highest bid first
        assert self.orderbook.bids[1, 0] == 50000.0
        assert self.orderbook.bids[2, 0] == 49999.0
        
        # Check quantities
        assert self.orderbook.bids[0, 1] == 0.5
        assert self.orderbook.bids[1, 1] == 1.0
        assert self.orderbook.bids[2, 1] == 2.0
    
    def test_update_asks(self):
        """Test ask updates"""
        updates = [
            {'side': 'ask', 'price': 50100.0, 'quantity': 1.0},
            {'side': 'ask', 'price': 50101.0, 'quantity': 2.0},
            {'side': 'ask', 'price': 50099.0, 'quantity': 0.5}
        ]
        
        self.orderbook.update(updates, timestamp=2000)
        
        # Check asks are sorted correctly (ascending)
        assert len(self.orderbook.asks) == 3
        assert self.orderbook.asks[0, 0] == 50099.0  # Lowest ask first
        assert self.orderbook.asks[1, 0] == 50100.0
        assert self.orderbook.asks[2, 0] == 50101.0
    
    def test_remove_price_level(self):
        """Test removing price levels"""
        # Add some levels
        updates = [
            {'side': 'bid', 'price': 50000.0, 'quantity': 1.0},
            {'side': 'bid', 'price': 49999.0, 'quantity': 2.0}
        ]
        self.orderbook.update(updates, timestamp=1000)
        
        # Remove a level by setting quantity to 0
        updates = [{'side': 'bid', 'price': 50000.0, 'quantity': 0.0}]
        self.orderbook.update(updates, timestamp=2000)
        
        assert len(self.orderbook.bids) == 1
        assert self.orderbook.bids[0, 0] == 49999.0
    
    def test_get_best_bid_ask(self):
        """Test getting best bid and ask"""
        # Empty orderbook
        best_bid, best_ask = self.orderbook.get_best_bid_ask()
        assert best_bid is None
        assert best_ask is None
        
        # Add some levels
        updates = [
            {'side': 'bid', 'price': 50000.0, 'quantity': 1.0},
            {'side': 'bid', 'price': 49999.0, 'quantity': 2.0},
            {'side': 'ask', 'price': 50001.0, 'quantity': 1.0},
            {'side': 'ask', 'price': 50002.0, 'quantity': 2.0}
        ]
        self.orderbook.update(updates, timestamp=1000)
        
        best_bid, best_ask = self.orderbook.get_best_bid_ask()
        assert best_bid == 50000.0
        assert best_ask == 50001.0
    
    def test_get_mid_price(self):
        """Test mid price calculation"""
        # Empty orderbook
        assert self.orderbook.get_mid_price() is None
        
        # Add levels
        updates = [
            {'side': 'bid', 'price': 50000.0, 'quantity': 1.0},
            {'side': 'ask', 'price': 50002.0, 'quantity': 1.0}
        ]
        self.orderbook.update(updates, timestamp=1000)
        
        mid_price = self.orderbook.get_mid_price()
        assert mid_price == 50001.0
    
    def test_get_features(self):
        """Test feature extraction"""
        # Add some orderbook data
        updates = [
            {'side': 'bid', 'price': 50000.0, 'quantity': 1.0},
            {'side': 'bid', 'price': 49999.0, 'quantity': 2.0},
            {'side': 'bid', 'price': 49998.0, 'quantity': 3.0},
            {'side': 'ask', 'price': 50001.0, 'quantity': 1.0},
            {'side': 'ask', 'price': 50002.0, 'quantity': 2.0},
            {'side': 'ask', 'price': 50003.0, 'quantity': 3.0}
        ]
        self.orderbook.update(updates, timestamp=1000)
        
        features = self.orderbook.get_features(depth=3)
        
        # Check feature dimensions
        assert len(features) == 20
        
        # Check spread
        assert features[0] == 1.0  # 50001 - 50000
        
        # Check mid price
        assert features[1] == 50000.5  # (50000 + 50001) / 2


class TestOrderBookJITFunctions:
    """Test JIT-compiled orderbook functions"""
    
    def test_update_orderbook_add_bid(self):
        """Test adding bid to orderbook"""
        bids = np.array([[50000.0, 1.0], [49999.0, 2.0]])
        asks = np.array([[50001.0, 1.0], [50002.0, 2.0]])
        
        # Add new bid
        new_bids, new_asks = update_orderbook(
            bids, asks, 
            update_side=0,  # bid
            price=49998.0,
            quantity=3.0,
            update_type=0  # add
        )
        
        # Check bid was added in correct position
        assert len(new_bids) == 3
        assert new_bids[2, 0] == 49998.0
        assert new_bids[2, 1] == 3.0
        
        # Asks should be unchanged
        assert np.array_equal(new_asks, asks)
    
    def test_update_orderbook_update_existing(self):
        """Test updating existing price level"""
        bids = np.array([[50000.0, 1.0], [49999.0, 2.0]])
        asks = np.array([[50001.0, 1.0], [50002.0, 2.0]])
        
        # Update existing bid
        new_bids, new_asks = update_orderbook(
            bids, asks,
            update_side=0,  # bid
            price=50000.0,
            quantity=5.0,
            update_type=0  # add/update
        )
        
        assert new_bids[0, 1] == 5.0  # Quantity updated
        assert len(new_bids) == 2  # No new level added
    
    def test_calculate_orderbook_features(self):
        """Test orderbook feature calculation"""
        bids = np.array([
            [50000.0, 1.0],
            [49999.0, 2.0],
            [49998.0, 3.0],
            [49997.0, 4.0],
            [49996.0, 5.0]
        ])
        asks = np.array([
            [50001.0, 1.0],
            [50002.0, 2.0],
            [50003.0, 3.0],
            [50004.0, 4.0],
            [50005.0, 5.0]
        ])
        
        features = calculate_orderbook_features(bids, asks, depth=5)
        
        # Check basic features
        assert features[0] == 1.0  # Spread
        assert features[1] == 50000.5  # Mid price
        
        # Check volume imbalance at first level
        # (1 - 1) / (1 + 1) = 0
        assert features[2] == 0.0
        
        # Check aggregate volume imbalance
        bid_total = 15.0  # 1+2+3+4+5
        ask_total = 15.0  # 1+2+3+4+5
        expected_imbalance = (bid_total - ask_total) / (bid_total + ask_total)
        assert features[12] == expected_imbalance


@pytest.mark.parametrize("update_side,price,quantity,expected_len", [
    (0, 50001.0, 1.0, 3),  # Add higher bid
    (0, 49998.0, 1.0, 3),  # Add lower bid
    (1, 50000.0, 1.0, 3),  # Add lower ask
    (1, 50003.0, 1.0, 3),  # Add higher ask
])
def test_orderbook_updates_parametrized(update_side, price, quantity, expected_len):
    """Parametrized test for various orderbook updates"""
    bids = np.array([[50000.0, 1.0], [49999.0, 2.0]])
    asks = np.array([[50001.0, 1.0], [50002.0, 2.0]])
    
    new_bids, new_asks = update_orderbook(
        bids, asks, update_side, price, quantity, 0
    )
    
    if update_side == 0:
        assert len(new_bids) == expected_len
    else:
        assert len(new_asks) == expected_len