"""Simple market maker strategy for demonstration"""

from typing import List, Dict
import numpy as np
from .base import BaseStrategy
from ..core.orderbook import OrderBook


class SimpleMarketMaker(BaseStrategy):
    """Simple spread-based market making strategy"""
    
    def __init__(self, spread: float = 0.0002, order_size: float = 0.1):
        """
        Initialize simple market maker
        
        Args:
            spread: Target spread (as fraction of price)
            order_size: Size of each order
        """
        super().__init__("SimpleMarketMaker")
        self.spread = spread
        self.order_size = order_size
        
    def on_initialize(self, **kwargs):
        """Initialize strategy parameters"""
        self.max_position = kwargs.get('max_position', 1.0)
        self.inventory_target = kwargs.get('inventory_target', 0.0)
        self.active_orders = {}
        
    def on_orderbook_update(self, orderbook: OrderBook, timestamp: int) -> List[Dict]:
        """Generate orders based on orderbook state"""
        actions = []
        
        # Get best bid/ask
        best_bid, best_ask = orderbook.get_best_bid_ask()
        if not best_bid or not best_ask:
            return actions
        
        # Calculate mid price and spread
        mid_price = (best_bid + best_ask) / 2
        current_spread = (best_ask - best_bid) / mid_price
        
        # Check position limits
        current_position = self.get_current_position()
        if abs(current_position) >= self.max_position:
            # Cancel all orders if at position limit
            for order_id in list(self.active_orders.keys()):
                actions.append({
                    'action': 'cancel_order',
                    'order_id': order_id
                })
            return actions
        
        # Only place orders if spread is wide enough
        if current_spread > self.spread:
            # Adjust order size based on inventory
            buy_size = self._adjust_size_for_inventory(self.order_size, current_position, 'buy')
            sell_size = self._adjust_size_for_inventory(self.order_size, current_position, 'sell')
            
            # Place buy order
            if buy_size > 0:
                buy_price = best_bid + 0.01  # Improve best bid by 1 cent
                actions.append({
                    'action': 'place_order',
                    'side': 'buy',
                    'price': buy_price,
                    'quantity': buy_size,
                    'order_type': 'limit'
                })
            
            # Place sell order
            if sell_size > 0:
                sell_price = best_ask - 0.01  # Improve best ask by 1 cent
                actions.append({
                    'action': 'place_order',
                    'side': 'sell',
                    'price': sell_price,
                    'quantity': sell_size,
                    'order_type': 'limit'
                })
        
        return actions
    
    def on_trade(self, trade: Dict, timestamp: int):
        """Handle trade execution"""
        # Record trade
        self.trades.append({
            'timestamp': timestamp,
            **trade
        })
        
        # Update active orders
        order_id = trade.get('order_id')
        if order_id and order_id in self.active_orders:
            del self.active_orders[order_id]
    
    def _adjust_size_for_inventory(self, base_size: float, current_position: float, side: str) -> float:
        """
        Adjust order size based on inventory
        
        Args:
            base_size: Base order size
            current_position: Current position
            side: 'buy' or 'sell'
        
        Returns:
            Adjusted order size
        """
        # Calculate inventory imbalance
        inventory_ratio = current_position / self.max_position if self.max_position > 0 else 0
        
        if side == 'buy':
            # Reduce buy size if long
            if inventory_ratio > 0.5:
                adjustment = 1 - (inventory_ratio - 0.5) * 2
                return base_size * max(0.1, adjustment)
            # Increase buy size if short
            elif inventory_ratio < -0.5:
                adjustment = 1 + abs(inventory_ratio + 0.5) * 2
                return base_size * min(2.0, adjustment)
        else:  # sell
            # Reduce sell size if short
            if inventory_ratio < -0.5:
                adjustment = 1 - (abs(inventory_ratio) - 0.5) * 2
                return base_size * max(0.1, adjustment)
            # Increase sell size if long
            elif inventory_ratio > 0.5:
                adjustment = 1 + (inventory_ratio - 0.5) * 2
                return base_size * min(2.0, adjustment)
        
        return base_size
    
    def on_reset(self):
        """Reset strategy state"""
        self.active_orders = {}