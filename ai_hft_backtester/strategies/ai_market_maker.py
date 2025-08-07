"""AI-powered market making strategy"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from ..ai.models import LSTMPredictor, MarketMakingPolicy
from ..ai.features import RealtimeFeatureEngine
from ..core.orderbook import OrderBook


@dataclass
class Order:
    """Order representation"""
    order_id: str
    side: str  # 'buy' or 'sell'
    price: float
    quantity: float
    timestamp: int
    status: str = 'pending'


@dataclass
class Position:
    """Position tracking"""
    quantity: float = 0.0
    average_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


class AIMarketMaker:
    """AI-powered market making strategy"""
    
    def __init__(
        self,
        price_predictor: LSTMPredictor,
        policy_network: MarketMakingPolicy,
        feature_engine: RealtimeFeatureEngine,
        risk_params: Dict,
        device: str = 'cpu'
    ):
        """
        Initialize AI market maker
        
        Args:
            price_predictor: Model for price prediction
            policy_network: RL policy for order placement
            feature_engine: Feature extraction engine
            risk_params: Risk management parameters
            device: Computing device (cpu/cuda)
        """
        self.price_predictor = price_predictor.to(device)
        self.policy_network = policy_network.to(device)
        self.feature_engine = feature_engine
        self.risk_params = risk_params
        self.device = device
        
        # State tracking
        self.position = Position()
        self.active_orders: Dict[str, Order] = {}
        self.order_counter = 0
        
        # Performance tracking
        self.trades = []
        self.pnl_history = []
        
        # Risk limits
        self.max_position = risk_params.get('max_position', 1.0)
        self.max_order_size = risk_params.get('max_order_size', 0.1)
        self.stop_loss = risk_params.get('stop_loss', 0.002)
        self.daily_loss_limit = risk_params.get('daily_loss_limit', 0.05)
        
    def on_orderbook_update(
        self,
        orderbook: OrderBook,
        timestamp: int
    ) -> List[Dict]:
        """
        Handle order book update and generate trading signals
        
        Args:
            orderbook: Current order book state
            timestamp: Update timestamp
        
        Returns:
            List of order actions
        """
        actions = []
        
        # Extract features
        orderbook_features = orderbook.get_features()
        features = self.feature_engine.get_features(orderbook_features)
        
        # Get price prediction
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            price_pred = self.price_predictor(features_tensor)
            price_probs = torch.softmax(price_pred, dim=-1)
        
        # Get current state for RL policy
        state = self._get_state(orderbook, features, price_probs)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get action from policy
        action_idx, action_params = self.policy_network.get_action(state_tensor)
        
        # Risk checks
        if self._check_risk_limits():
            # Cancel all orders if risk limit hit
            actions.extend(self._cancel_all_orders())
            return actions
        
        # Generate orders based on action
        best_bid, best_ask = orderbook.get_best_bid_ask()
        if best_bid and best_ask:
            spread = action_params['spread']
            size = min(action_params['size'], self.max_order_size)
            
            # Adjust for position limits
            size = self._adjust_size_for_position(size)
            
            # Place orders
            if abs(self.position.quantity) < self.max_position:
                # Buy order
                buy_price = best_bid - spread / 2
                actions.append({
                    'action': 'place_order',
                    'side': 'buy',
                    'price': buy_price,
                    'quantity': size,
                    'order_type': 'limit'
                })
                
                # Sell order
                sell_price = best_ask + spread / 2
                actions.append({
                    'action': 'place_order',
                    'side': 'sell',
                    'price': sell_price,
                    'quantity': size,
                    'order_type': 'limit'
                })
        
        # Update feature buffers
        if best_bid and best_ask:
            mid_price = (best_bid + best_ask) / 2
            self.feature_engine.update_buffers(mid_price, orderbook.bids[0, 1] + orderbook.asks[0, 1])
        
        return actions
    
    def on_trade(self, trade: Dict, timestamp: int):
        """
        Handle trade execution
        
        Args:
            trade: Trade details
            timestamp: Trade timestamp
        """
        # Update position
        side = trade['side']
        price = trade['price']
        quantity = trade['quantity']
        
        if side == 'buy':
            # Bought
            new_quantity = self.position.quantity + quantity
            self.position.average_price = (
                (self.position.average_price * self.position.quantity + price * quantity) /
                new_quantity
            )
            self.position.quantity = new_quantity
        else:
            # Sold
            if self.position.quantity > 0:
                # Closing long position
                realized = min(quantity, self.position.quantity) * (price - self.position.average_price)
                self.position.realized_pnl += realized
            
            self.position.quantity -= quantity
            if self.position.quantity < 0:
                # Now short
                self.position.average_price = price
        
        # Record trade
        self.trades.append({
            'timestamp': timestamp,
            'side': side,
            'price': price,
            'quantity': quantity,
            'position': self.position.quantity,
            'realized_pnl': self.position.realized_pnl
        })
    
    def _get_state(
        self,
        orderbook: OrderBook,
        features: np.ndarray,
        price_probs: torch.Tensor
    ) -> np.ndarray:
        """
        Construct state vector for RL policy
        
        Returns:
            State vector
        """
        # Market features
        market_state = features[:50]  # Top 50 features
        
        # Price prediction
        pred_state = price_probs.cpu().numpy().flatten()
        
        # Position state
        position_state = np.array([
            self.position.quantity / self.max_position,
            self.position.unrealized_pnl,
            self.position.realized_pnl,
            len(self.active_orders) / 10.0  # Normalized
        ])
        
        # Order book state
        best_bid, best_ask = orderbook.get_best_bid_ask()
        if best_bid and best_ask:
            book_state = np.array([
                (best_ask - best_bid) / best_bid,  # Relative spread
                orderbook.bids[0, 1] / (orderbook.bids[0, 1] + orderbook.asks[0, 1]),  # Volume imbalance
            ])
        else:
            book_state = np.zeros(2)
        
        # Combine all states
        state = np.concatenate([market_state, pred_state, position_state, book_state])
        
        return state
    
    def _check_risk_limits(self) -> bool:
        """
        Check if any risk limits are breached
        
        Returns:
            True if risk limit breached
        """
        # Position limit
        if abs(self.position.quantity) > self.max_position:
            return True
        
        # Stop loss
        if self.position.unrealized_pnl < -self.stop_loss * self.max_position:
            return True
        
        # Daily loss limit
        if self.position.realized_pnl < -self.daily_loss_limit:
            return True
        
        return False
    
    def _adjust_size_for_position(self, size: float) -> float:
        """
        Adjust order size based on current position
        
        Args:
            size: Desired order size
        
        Returns:
            Adjusted size
        """
        # Reduce size as we approach position limit
        position_ratio = abs(self.position.quantity) / self.max_position
        
        if position_ratio > 0.8:
            # Significantly reduce size near limit
            size *= (1 - position_ratio) * 2
        elif position_ratio > 0.5:
            # Moderately reduce size
            size *= (1 - position_ratio * 0.5)
        
        return max(0.01, size)  # Minimum order size
    
    def _cancel_all_orders(self) -> List[Dict]:
        """
        Cancel all active orders
        
        Returns:
            List of cancel actions
        """
        actions = []
        for order_id in list(self.active_orders.keys()):
            actions.append({
                'action': 'cancel_order',
                'order_id': order_id
            })
        return actions
    
    def get_performance_metrics(self) -> Dict:
        """
        Calculate performance metrics
        
        Returns:
            Dictionary of performance metrics
        """
        if not self.trades:
            return {}
        
        # Calculate metrics
        total_trades = len(self.trades)
        winning_trades = sum(1 for t in self.trades if t.get('pnl', 0) > 0)
        
        returns = [t.get('pnl', 0) for t in self.trades]
        
        metrics = {
            'total_trades': total_trades,
            'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
            'total_pnl': self.position.realized_pnl,
            'sharpe_ratio': np.mean(returns) / np.std(returns) if len(returns) > 1 else 0,
            'max_drawdown': self._calculate_max_drawdown(),
            'avg_position': np.mean([abs(t['position']) for t in self.trades])
        }
        
        return metrics
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        if not self.pnl_history:
            return 0.0
        
        cumulative = np.cumsum(self.pnl_history)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        
        return abs(np.min(drawdown)) if len(drawdown) > 0 else 0.0