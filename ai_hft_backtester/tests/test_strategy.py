"""Unit tests for trading strategies"""

import numpy as np
import torch
import pytest
from unittest.mock import Mock, MagicMock
from ..strategies.ai_market_maker import AIMarketMaker, Order, Position
from ..ai.models import LSTMPredictor, MarketMakingPolicy
from ..ai.features import RealtimeFeatureEngine
from ..core.orderbook import OrderBook


class TestPosition:
    """Test Position dataclass"""
    
    def test_initialization(self):
        """Test position initialization"""
        position = Position()
        assert position.quantity == 0.0
        assert position.average_price == 0.0
        assert position.unrealized_pnl == 0.0
        assert position.realized_pnl == 0.0
    
    def test_position_with_values(self):
        """Test position with values"""
        position = Position(
            quantity=1.0,
            average_price=50000.0,
            unrealized_pnl=-100.0,
            realized_pnl=200.0
        )
        assert position.quantity == 1.0
        assert position.average_price == 50000.0
        assert position.unrealized_pnl == -100.0
        assert position.realized_pnl == 200.0


class TestAIMarketMaker:
    """Test AIMarketMaker strategy"""
    
    def setup_method(self):
        """Setup test fixtures"""
        # Create mock models
        self.price_predictor = Mock(spec=LSTMPredictor)
        self.policy_network = Mock(spec=MarketMakingPolicy)
        
        # Create mock feature engine
        self.feature_engine = Mock(spec=RealtimeFeatureEngine)
        self.feature_engine.get_features = Mock(return_value=np.zeros(50))
        self.feature_engine.update_buffers = Mock()
        
        # Risk parameters
        self.risk_params = {
            'max_position': 1.0,
            'max_order_size': 0.1,
            'stop_loss': 0.002,
            'daily_loss_limit': 0.05
        }
        
        # Create strategy
        self.strategy = AIMarketMaker(
            price_predictor=self.price_predictor,
            policy_network=self.policy_network,
            feature_engine=self.feature_engine,
            risk_params=self.risk_params,
            device='cpu'
        )
    
    def test_initialization(self):
        """Test strategy initialization"""
        assert self.strategy.max_position == 1.0
        assert self.strategy.max_order_size == 0.1
        assert self.strategy.stop_loss == 0.002
        assert self.strategy.daily_loss_limit == 0.05
        assert self.strategy.position.quantity == 0.0
        assert len(self.strategy.active_orders) == 0
    
    def test_on_orderbook_update_no_risk_breach(self):
        """Test orderbook update handling without risk breach"""
        # Setup orderbook
        orderbook = OrderBook("BTCUSDT")
        orderbook.bids = np.array([[50000.0, 1.0], [49999.0, 2.0]])
        orderbook.asks = np.array([[50001.0, 1.0], [50002.0, 2.0]])
        
        # Mock model predictions
        price_pred = torch.tensor([[0.2, 0.5, 0.3]])  # Neutral prediction
        self.price_predictor.return_value = price_pred
        
        action_params = {'spread': 0.0002, 'size': 0.05}
        self.policy_network.get_action.return_value = (0, action_params)
        
        # Call strategy
        actions = self.strategy.on_orderbook_update(orderbook, 1000000)
        
        # Should generate buy and sell orders
        assert len(actions) == 2
        
        # Check buy order
        buy_action = next(a for a in actions if a['side'] == 'buy')
        assert buy_action['action'] == 'place_order'
        assert buy_action['price'] == 50000.0 - 0.0001  # best_bid - spread/2
        assert buy_action['quantity'] == 0.05
        
        # Check sell order
        sell_action = next(a for a in actions if a['side'] == 'sell')
        assert sell_action['action'] == 'place_order'
        assert sell_action['price'] == 50001.0 + 0.0001  # best_ask + spread/2
        assert sell_action['quantity'] == 0.05
    
    def test_on_orderbook_update_risk_limit_hit(self):
        """Test orderbook update when risk limit is hit"""
        # Set position at max
        self.strategy.position.quantity = 1.1  # Over max
        
        orderbook = OrderBook("BTCUSDT")
        orderbook.bids = np.array([[50000.0, 1.0]])
        orderbook.asks = np.array([[50001.0, 1.0]])
        
        # Mock predictions
        self.price_predictor.return_value = torch.tensor([[0.1, 0.8, 0.1]])
        self.policy_network.get_action.return_value = (0, {'spread': 0.0002, 'size': 0.1})
        
        # Mock active orders for cancellation
        self.strategy.active_orders = {
            'order_1': Order('order_1', 'buy', 49999.0, 0.1, 1000, 'active'),
            'order_2': Order('order_2', 'sell', 50002.0, 0.1, 1000, 'active')
        }
        
        actions = self.strategy.on_orderbook_update(orderbook, 2000000)
        
        # Should only have cancel actions
        assert all(a['action'] == 'cancel_order' for a in actions)
        assert len(actions) == 2
    
    def test_on_trade_buy(self):
        """Test handling buy trade"""
        # Initial position
        self.strategy.position.quantity = 0.5
        self.strategy.position.average_price = 50000.0
        
        trade = {
            'side': 'buy',
            'price': 49900.0,
            'quantity': 0.3
        }
        
        self.strategy.on_trade(trade, 3000000)
        
        # Check position update
        assert self.strategy.position.quantity == 0.8  # 0.5 + 0.3
        
        # Check average price calculation
        # (0.5 * 50000 + 0.3 * 49900) / 0.8 = 49962.5
        assert self.strategy.position.average_price == pytest.approx(49962.5)
        
        # Check trade recording
        assert len(self.strategy.trades) == 1
        assert self.strategy.trades[0]['side'] == 'buy'
        assert self.strategy.trades[0]['price'] == 49900.0
    
    def test_on_trade_sell_closing_position(self):
        """Test handling sell trade that closes position"""
        # Initial long position
        self.strategy.position.quantity = 0.5
        self.strategy.position.average_price = 50000.0
        self.strategy.position.realized_pnl = 0.0
        
        trade = {
            'side': 'sell',
            'price': 50100.0,
            'quantity': 0.3
        }
        
        self.strategy.on_trade(trade, 4000000)
        
        # Check position update
        assert self.strategy.position.quantity == 0.2  # 0.5 - 0.3
        
        # Check realized PnL
        # Sold 0.3 at 50100, bought at 50000 avg
        # PnL = 0.3 * (50100 - 50000) = 30
        assert self.strategy.position.realized_pnl == 30.0
    
    def test_get_state(self):
        """Test state vector construction"""
        # Setup orderbook
        orderbook = OrderBook("BTCUSDT")
        orderbook.bids = np.array([[50000.0, 1.0]])
        orderbook.asks = np.array([[50001.0, 1.0]])
        
        # Mock features
        features = np.random.randn(60)
        
        # Mock price predictions
        price_probs = torch.tensor([0.3, 0.4, 0.3])
        
        # Get state
        state = self.strategy._get_state(orderbook, features, price_probs)
        
        # Check state composition
        assert len(state) > 50  # Should include all components
        assert isinstance(state, np.ndarray)
    
    def test_check_risk_limits(self):
        """Test risk limit checking"""
        # No breach initially
        assert not self.strategy._check_risk_limits()
        
        # Position limit breach
        self.strategy.position.quantity = 1.5
        assert self.strategy._check_risk_limits()
        
        # Reset and test stop loss breach
        self.strategy.position.quantity = 0.5
        self.strategy.position.unrealized_pnl = -0.003  # Below stop loss
        assert self.strategy._check_risk_limits()
        
        # Reset and test daily loss limit breach
        self.strategy.position.unrealized_pnl = 0.0
        self.strategy.position.realized_pnl = -0.06  # Below daily limit
        assert self.strategy._check_risk_limits()
    
    def test_adjust_size_for_position(self):
        """Test order size adjustment based on position"""
        # Small position - no adjustment
        self.strategy.position.quantity = 0.3
        adjusted = self.strategy._adjust_size_for_position(0.1)
        assert adjusted == 0.1
        
        # Medium position - moderate reduction
        self.strategy.position.quantity = 0.6
        adjusted = self.strategy._adjust_size_for_position(0.1)
        assert adjusted < 0.1
        assert adjusted > 0.05
        
        # Near limit - significant reduction
        self.strategy.position.quantity = 0.9
        adjusted = self.strategy._adjust_size_for_position(0.1)
        assert adjusted < 0.05
        assert adjusted >= 0.01  # Minimum size
    
    def test_get_performance_metrics(self):
        """Test performance metrics calculation"""
        # Add some trades
        self.strategy.trades = [
            {'pnl': 10.0, 'position': 0.5},
            {'pnl': -5.0, 'position': 0.3},
            {'pnl': 15.0, 'position': 0.7},
            {'pnl': -2.0, 'position': 0.4}
        ]
        
        self.strategy.position.realized_pnl = 18.0
        self.strategy.pnl_history = [10.0, 5.0, 20.0, 18.0]
        
        metrics = self.strategy.get_performance_metrics()
        
        assert metrics['total_trades'] == 4
        assert metrics['win_rate'] == 0.5  # 2 wins out of 4
        assert metrics['total_pnl'] == 18.0
        assert metrics['avg_position'] == pytest.approx(0.475)  # Average of positions
        
        # Check Sharpe ratio calculation
        returns = [10.0, -5.0, 15.0, -2.0]
        expected_sharpe = np.mean(returns) / np.std(returns)
        assert metrics['sharpe_ratio'] == pytest.approx(expected_sharpe)


class TestAIMarketMakerIntegration:
    """Integration tests for AIMarketMaker"""
    
    def test_full_trading_cycle(self):
        """Test complete trading cycle"""
        # Create real models (small for testing)
        price_predictor = LSTMPredictor(input_size=50, hidden_size=32, num_layers=1)
        policy_network = MarketMakingPolicy(state_size=60, hidden_size=32, n_actions=9)
        
        # Feature engine
        feature_config = {
            'means': np.zeros(50),
            'stds': np.ones(50),
            'window_sizes': [10, 20]
        }
        feature_engine = RealtimeFeatureEngine(feature_config)
        
        # Create strategy
        strategy = AIMarketMaker(
            price_predictor=price_predictor,
            policy_network=policy_network,
            feature_engine=feature_engine,
            risk_params={'max_position': 1.0, 'stop_loss': 0.01},
            device='cpu'
        )
        
        # Simulate trading
        orderbook = OrderBook("BTCUSDT")
        
        # Add some price history
        for i in range(30):
            price = 50000.0 + i * 10
            feature_engine.update_buffers(price, 10.0)
        
        # Update orderbook
        orderbook.bids = np.array([[50300.0, 1.0], [50299.0, 2.0]])
        orderbook.asks = np.array([[50301.0, 1.0], [50302.0, 2.0]])
        
        # Get trading signals
        actions = strategy.on_orderbook_update(orderbook, 1000000)
        
        # Should generate some actions
        assert len(actions) > 0
        
        # Simulate a trade
        if actions:
            strategy.on_trade({
                'side': 'buy',
                'price': 50300.0,
                'quantity': 0.05
            }, 2000000)
            
            assert strategy.position.quantity == 0.05
            assert len(strategy.trades) == 1