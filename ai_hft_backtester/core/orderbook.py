"""Order book reconstruction and management with Numba optimization"""

import numpy as np
from numba import jit, types
from numba.typed import Dict
from typing import Tuple, Optional


@jit(nopython=True)
def update_orderbook(
    bids: np.ndarray,
    asks: np.ndarray,
    update_side: int,  # 0 for bid, 1 for ask
    price: float,
    quantity: float,
    update_type: int,  # 0 for add/update, 1 for remove
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Update order book with new market data
    
    Args:
        bids: Current bid prices and quantities (N x 2)
        asks: Current ask prices and quantities (N x 2)
        update_side: 0 for bid, 1 for ask
        price: Price level to update
        quantity: New quantity (0 to remove)
        update_type: 0 for add/update, 1 for remove
    
    Returns:
        Updated bids and asks arrays
    """
    if update_side == 0:  # Update bids
        if update_type == 1 or quantity == 0:  # Remove
            mask = bids[:, 0] != price
            bids = bids[mask]
        else:  # Add or update
            idx = np.searchsorted(-bids[:, 0], -price)
            if idx < len(bids) and bids[idx, 0] == price:
                bids[idx, 1] = quantity
            else:
                new_bid = np.array([[price, quantity]])
                bids = np.concatenate((bids[:idx], new_bid, bids[idx:]))
    else:  # Update asks
        if update_type == 1 or quantity == 0:  # Remove
            mask = asks[:, 0] != price
            asks = asks[mask]
        else:  # Add or update
            idx = np.searchsorted(asks[:, 0], price)
            if idx < len(asks) and asks[idx, 0] == price:
                asks[idx, 1] = quantity
            else:
                new_ask = np.array([[price, quantity]])
                asks = np.concatenate((asks[:idx], new_ask, asks[idx:]))
    
    return bids, asks


@jit(nopython=True)
def calculate_orderbook_features(
    bids: np.ndarray,
    asks: np.ndarray,
    depth: int = 10
) -> np.ndarray:
    """
    Calculate order book features for AI models
    
    Features include:
    - Bid-ask spread
    - Mid price
    - Volume imbalance
    - Depth imbalance
    - Weighted average prices
    """
    features = np.zeros(20)
    
    if len(bids) > 0 and len(asks) > 0:
        # Basic features
        best_bid = bids[0, 0]
        best_ask = asks[0, 0]
        features[0] = best_ask - best_bid  # Spread
        features[1] = (best_bid + best_ask) / 2  # Mid price
        
        # Volume features at different depths
        for i in range(min(depth, len(bids), len(asks))):
            bid_vol = bids[i, 1] if i < len(bids) else 0
            ask_vol = asks[i, 1] if i < len(asks) else 0
            
            # Volume imbalance at level i
            total_vol = bid_vol + ask_vol
            if total_vol > 0:
                features[2 + i] = (bid_vol - ask_vol) / total_vol
        
        # Aggregate volume features
        bid_total = np.sum(bids[:depth, 1]) if len(bids) > 0 else 0
        ask_total = np.sum(asks[:depth, 1]) if len(asks) > 0 else 0
        
        if bid_total + ask_total > 0:
            features[12] = (bid_total - ask_total) / (bid_total + ask_total)
        
        # Weighted average prices
        if bid_total > 0:
            features[13] = np.sum(bids[:depth, 0] * bids[:depth, 1]) / bid_total
        if ask_total > 0:
            features[14] = np.sum(asks[:depth, 0] * asks[:depth, 1]) / ask_total
    
    return features


class OrderBook:
    """Order book manager with Level-2 and Level-3 support"""
    
    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol
        self.bids = np.empty((0, 2), dtype=np.float64)
        self.asks = np.empty((0, 2), dtype=np.float64)
        self.last_update_time = 0
        self.sequence_number = 0
        
    def update(self, updates: list, timestamp: int):
        """Process order book updates"""
        for update in updates:
            side = 0 if update['side'] == 'bid' else 1
            price = float(update['price'])
            quantity = float(update['quantity'])
            update_type = 0  # Default to add/update
            
            self.bids, self.asks = update_orderbook(
                self.bids, self.asks, side, price, quantity, update_type
            )
        
        self.last_update_time = timestamp
        self.sequence_number += 1
    
    def get_features(self, depth: int = 10) -> np.ndarray:
        """Extract features for AI models"""
        return calculate_orderbook_features(self.bids, self.asks, depth)
    
    def get_best_bid_ask(self) -> Tuple[Optional[float], Optional[float]]:
        """Get best bid and ask prices"""
        best_bid = self.bids[0, 0] if len(self.bids) > 0 else None
        best_ask = self.asks[0, 0] if len(self.asks) > 0 else None
        return best_bid, best_ask
    
    def get_mid_price(self) -> Optional[float]:
        """Calculate mid price"""
        best_bid, best_ask = self.get_best_bid_ask()
        if best_bid and best_ask:
            return (best_bid + best_ask) / 2
        return None