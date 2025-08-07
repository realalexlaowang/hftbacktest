"""Trading environment for reinforcement learning"""

import numpy as np
import gym
from gym import spaces
from typing import Dict, Tuple, Optional
from ..core.orderbook import OrderBook
from ..ai.features import FeatureEngineer


class TradingEnvironment(gym.Env):
    """OpenAI Gym environment for trading"""
    
    def __init__(
        self,
        data_loader,
        initial_balance: float = 10000,
        commission: float = 0.0002,
        max_position: float = 1.0,
        feature_engineer: Optional[FeatureEngineer] = None
    ):
        """
        Initialize trading environment
        
        Args:
            data_loader: Data source for market data
            initial_balance: Starting capital
            commission: Trading commission rate
            max_position: Maximum position size
            feature_engineer: Feature extraction engine
        """
        super().__init__()
        
        self.data_loader = data_loader
        self.initial_balance = initial_balance
        self.commission = commission
        self.max_position = max_position
        self.feature_engineer = feature_engineer or FeatureEngineer()
        
        # State space: features + position info
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(100,),  # Adjust based on feature dimensions
            dtype=np.float32
        )
        
        # Action space: discrete actions
        # 0: Hold, 1: Buy 25%, 2: Buy 50%, 3: Buy 100%
        # 4: Sell 25%, 5: Sell 50%, 6: Sell 100%, 7: Close position
        self.action_space = spaces.Discrete(8)
        
        # Initialize state
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.balance = self.initial_balance
        self.position = 0.0
        self.total_pnl = 0.0
        self.n_trades = 0
        self.current_step = 0
        
        # Reset data
        self.data_loader.reset()
        self.orderbook = OrderBook("BTCUSDT")
        
        # Get initial observation
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment
        
        Args:
            action: Action to take
        
        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Get current market data
        market_data = self.data_loader.get_next()
        if market_data is None:
            return self._get_observation(), 0, True, {'reason': 'data_exhausted'}
        
        # Update orderbook
        self.orderbook.update(market_data['updates'], market_data['timestamp'])
        
        # Get current price
        best_bid, best_ask = self.orderbook.get_best_bid_ask()
        if not best_bid or not best_ask:
            return self._get_observation(), 0, False, {}
        
        mid_price = (best_bid + best_ask) / 2
        
        # Execute action
        old_position = self.position
        old_balance = self.balance
        
        self._execute_action(action, mid_price)
        
        # Calculate reward
        reward = self._calculate_reward(old_position, old_balance, mid_price)
        
        # Check if done
        done = self._is_done()
        
        # Get next observation
        obs = self._get_observation()
        
        # Info dict
        info = {
            'balance': self.balance,
            'position': self.position,
            'total_pnl': self.total_pnl,
            'n_trades': self.n_trades
        }
        
        self.current_step += 1
        
        return obs, reward, done, info
    
    def _execute_action(self, action: int, price: float):
        """Execute trading action"""
        position_change = 0.0
        
        if action == 0:  # Hold
            return
        elif action == 1:  # Buy 25%
            position_change = 0.25 * self.max_position - self.position
        elif action == 2:  # Buy 50%
            position_change = 0.50 * self.max_position - self.position
        elif action == 3:  # Buy 100%
            position_change = self.max_position - self.position
        elif action == 4:  # Sell 25%
            position_change = -0.25 * self.max_position - self.position
        elif action == 5:  # Sell 50%
            position_change = -0.50 * self.max_position - self.position
        elif action == 6:  # Sell 100%
            position_change = -self.max_position - self.position
        elif action == 7:  # Close position
            position_change = -self.position
        
        if abs(position_change) > 0.001:  # Minimum trade size
            # Calculate cost including commission
            cost = abs(position_change) * price * (1 + self.commission)
            
            # Check if we have enough balance
            if position_change > 0 and cost > self.balance:
                # Adjust position change to available balance
                position_change = self.balance / (price * (1 + self.commission))
                cost = self.balance
            
            # Update position and balance
            self.position += position_change
            self.balance -= position_change * price  # Negative for sells
            self.balance -= abs(position_change) * price * self.commission  # Commission
            
            self.n_trades += 1
    
    def _calculate_reward(self, old_position: float, old_balance: float, current_price: float) -> float:
        """Calculate step reward"""
        # Current portfolio value
        current_value = self.balance + self.position * current_price
        old_value = old_balance + old_position * current_price
        
        # PnL for this step
        step_pnl = current_value - old_value
        
        # Reward components
        pnl_reward = step_pnl / self.initial_balance  # Normalized PnL
        
        # Penalize large positions (risk)
        position_penalty = -0.01 * (self.position / self.max_position) ** 2
        
        # Penalize excessive trading
        trade_penalty = -0.001 if self.n_trades > self.current_step / 10 else 0
        
        return pnl_reward + position_penalty + trade_penalty
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        # Get orderbook features
        orderbook_features = self.orderbook.get_features()
        
        # Get additional features from feature engineer
        if self.feature_engineer:
            # This would be expanded with actual feature extraction
            additional_features = np.zeros(50)
        else:
            additional_features = np.zeros(50)
        
        # Add position information
        position_features = np.array([
            self.position / self.max_position,
            self.balance / self.initial_balance,
            self.total_pnl / self.initial_balance,
            self.n_trades / max(1, self.current_step)
        ])
        
        # Combine all features
        observation = np.concatenate([
            orderbook_features,
            additional_features,
            position_features
        ])
        
        # Pad or truncate to match observation space
        if len(observation) < self.observation_space.shape[0]:
            observation = np.pad(observation, (0, self.observation_space.shape[0] - len(observation)))
        else:
            observation = observation[:self.observation_space.shape[0]]
        
        return observation.astype(np.float32)
    
    def _is_done(self) -> bool:
        """Check if episode is done"""
        # Done if out of money
        if self.balance <= 0:
            return True
        
        # Done if max steps reached
        if self.current_step >= 10000:  # Configurable
            return True
        
        return False
    
    def render(self, mode='human'):
        """Render environment state"""
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Balance: ${self.balance:.2f}")
            print(f"Position: {self.position:.4f}")
            print(f"Total PnL: ${self.total_pnl:.2f}")
            print(f"Trades: {self.n_trades}")
            print("-" * 40)