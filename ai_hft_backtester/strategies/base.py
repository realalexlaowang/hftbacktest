"""Base strategy class for all trading strategies"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import numpy as np
from ..core.orderbook import OrderBook


class BaseStrategy(ABC):
    """Abstract base class for trading strategies"""
    
    def __init__(self, name: str = None):
        """
        Initialize base strategy
        
        Args:
            name: Strategy name
        """
        self.name = name or self.__class__.__name__
        self.is_initialized = False
        
        # Performance tracking
        self.trades = []
        self.orders = []
        self.positions = []
        self.pnl_history = []
        
    def initialize(self, initial_capital: float, **kwargs):
        """
        Initialize strategy with parameters
        
        Args:
            initial_capital: Starting capital
            **kwargs: Additional parameters
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.is_initialized = True
        
        # Call child class initialization
        self.on_initialize(**kwargs)
    
    @abstractmethod
    def on_initialize(self, **kwargs):
        """
        Strategy-specific initialization
        
        Args:
            **kwargs: Strategy parameters
        """
        pass
    
    @abstractmethod
    def on_orderbook_update(self, orderbook: OrderBook, timestamp: int) -> List[Dict]:
        """
        Handle orderbook update
        
        Args:
            orderbook: Current orderbook state
            timestamp: Update timestamp
        
        Returns:
            List of order actions
        """
        pass
    
    @abstractmethod
    def on_trade(self, trade: Dict, timestamp: int):
        """
        Handle trade execution
        
        Args:
            trade: Trade details
            timestamp: Trade timestamp
        """
        pass
    
    def on_order_update(self, order: Dict, timestamp: int):
        """
        Handle order status update
        
        Args:
            order: Order details
            timestamp: Update timestamp
        """
        # Default implementation - can be overridden
        self.orders.append({
            'timestamp': timestamp,
            **order
        })
    
    def on_fill(self, fill: Dict, timestamp: int):
        """
        Handle order fill
        
        Args:
            fill: Fill details
            timestamp: Fill timestamp
        """
        # Default implementation
        self.trades.append({
            'timestamp': timestamp,
            **fill
        })
    
    def get_current_position(self) -> float:
        """
        Get current position
        
        Returns:
            Current position quantity
        """
        # Calculate from trades
        position = 0.0
        for trade in self.trades:
            if trade['side'] == 'buy':
                position += trade['quantity']
            else:
                position -= trade['quantity']
        return position
    
    def get_performance_metrics(self) -> Dict:
        """
        Calculate performance metrics
        
        Returns:
            Dictionary of performance metrics
        """
        if not self.trades:
            return {
                'total_trades': 0,
                'total_pnl': 0,
                'win_rate': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0
            }
        
        # Basic metrics
        total_trades = len(self.trades)
        
        # Calculate PnL from trades
        pnl = self._calculate_pnl()
        total_pnl = sum(pnl)
        
        # Win rate
        winning_trades = sum(1 for p in pnl if p > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Sharpe ratio
        if len(pnl) > 1:
            returns = np.array(pnl)
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Max drawdown
        cumulative_pnl = np.cumsum(pnl)
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = cumulative_pnl - running_max
        max_drawdown = abs(np.min(drawdown)) if len(drawdown) > 0 else 0
        
        return {
            'total_trades': total_trades,
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'avg_trade_pnl': total_pnl / total_trades if total_trades > 0 else 0
        }
    
    def _calculate_pnl(self) -> List[float]:
        """Calculate PnL for each trade"""
        pnl = []
        position = 0.0
        avg_price = 0.0
        
        for trade in sorted(self.trades, key=lambda x: x['timestamp']):
            if trade['side'] == 'buy':
                # Update average price
                new_position = position + trade['quantity']
                if new_position != 0:
                    avg_price = (position * avg_price + trade['quantity'] * trade['price']) / new_position
                position = new_position
            else:
                # Calculate PnL on sell
                if position > 0:
                    trade_pnl = min(trade['quantity'], position) * (trade['price'] - avg_price)
                    pnl.append(trade_pnl)
                position -= trade['quantity']
        
        return pnl
    
    def reset(self):
        """Reset strategy state"""
        self.trades = []
        self.orders = []
        self.positions = []
        self.pnl_history = []
        self.current_capital = self.initial_capital
        self.is_initialized = False
        
        # Call child class reset
        self.on_reset()
    
    def on_reset(self):
        """Strategy-specific reset - to be overridden"""
        pass