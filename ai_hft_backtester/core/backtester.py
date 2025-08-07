"""Main backtesting engine"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
import logging
from tqdm import tqdm

from .orderbook import OrderBook
from .latency import BinanceLatencyModel
from ..data.loader import DataLoader
from ..strategies.ai_market_maker import AIMarketMaker


logger = logging.getLogger(__name__)


class Backtester:
    """Main backtesting engine with tick-by-tick simulation"""
    
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        exchange: str = "binance",
        data_path: str = "./data",
        latency_config: Dict = None
    ):
        """
        Initialize backtester
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_path: Path to historical data
            latency_config: Latency model configuration
        """
        self.symbol = symbol
        self.exchange = exchange
        self.data_path = data_path
        
        # Initialize components
        self.data_loader = DataLoader(data_path)
        self.latency_model = BinanceLatencyModel(
            location=latency_config.get('location', 'tokyo') if latency_config else 'tokyo'
        )
        self.orderbook = OrderBook(symbol)
        
        # Simulation state
        self.current_time = 0
        self.event_queue = []
        self.order_queue = {}
        self.trades = []
        self.fills = []
        
        # Performance tracking
        self.equity_curve = []
        self.metrics = {}
        
    def run(
        self,
        strategy: AIMarketMaker,
        start_date: str,
        end_date: str,
        initial_capital: float = 10000,
        commission_rate: float = 0.0002,  # Binance maker fee
        progress_bar: bool = True
    ) -> 'BacktestResults':
        """
        Run backtest simulation
        
        Args:
            strategy: Trading strategy instance
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            initial_capital: Starting capital
            commission_rate: Trading commission rate
            progress_bar: Show progress bar
        
        Returns:
            BacktestResults object
        """
        logger.info(f"Starting backtest for {self.symbol} from {start_date} to {end_date}")
        
        # Load data
        orderbook_data, trade_data = self.data_loader.load_data(
            self.symbol, start_date, end_date
        )
        
        # Initialize simulation
        self.capital = initial_capital
        self.commission_rate = commission_rate
        
        # Create event queue
        self._create_event_queue(orderbook_data, trade_data)
        
        # Sort events by timestamp
        self.event_queue.sort(key=lambda x: x['timestamp'])
        
        # Progress tracking
        total_events = len(self.event_queue)
        pbar = tqdm(total=total_events, desc="Backtesting") if progress_bar else None
        
        # Main simulation loop
        for event in self.event_queue:
            self.current_time = event['timestamp']
            
            # Process event based on type
            if event['type'] == 'orderbook_update':
                self._process_orderbook_update(event, strategy)
            elif event['type'] == 'trade':
                self._process_trade(event, strategy)
            elif event['type'] == 'order_arrival':
                self._process_order_arrival(event, strategy)
            
            # Update equity curve
            self._update_equity(strategy)
            
            if pbar:
                pbar.update(1)
        
        if pbar:
            pbar.close()
        
        # Calculate final metrics
        self._calculate_metrics(strategy)
        
        # Create results object
        results = BacktestResults(
            strategy_name=strategy.__class__.__name__,
            symbol=self.symbol,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            final_capital=self.capital,
            trades=self.trades,
            fills=self.fills,
            equity_curve=pd.DataFrame(self.equity_curve),
            metrics=self.metrics
        )
        
        return results
    
    def _create_event_queue(self, orderbook_data: pd.DataFrame, trade_data: pd.DataFrame):
        """Create unified event queue from market data"""
        self.event_queue = []
        
        # Add orderbook updates
        for idx, row in orderbook_data.iterrows():
            # Apply feed latency
            latency = self.latency_model.get_feed_latency()
            arrival_time = row['timestamp'] + int(latency)
            
            self.event_queue.append({
                'type': 'orderbook_update',
                'timestamp': arrival_time,
                'data': row
            })
        
        # Add trades
        for idx, row in trade_data.iterrows():
            # Apply feed latency
            latency = self.latency_model.get_feed_latency()
            arrival_time = row['timestamp'] + int(latency)
            
            self.event_queue.append({
                'type': 'trade',
                'timestamp': arrival_time,
                'data': row
            })
    
    def _process_orderbook_update(self, event: Dict, strategy: AIMarketMaker):
        """Process orderbook update event"""
        data = event['data']
        
        # Update orderbook
        updates = self._parse_orderbook_updates(data)
        self.orderbook.update(updates, event['timestamp'])
        
        # Get strategy signals
        actions = strategy.on_orderbook_update(self.orderbook, event['timestamp'])
        
        # Process actions
        for action in actions:
            if action['action'] == 'place_order':
                self._place_order(action, strategy, event['timestamp'])
            elif action['action'] == 'cancel_order':
                self._cancel_order(action, strategy)
    
    def _process_trade(self, event: Dict, strategy: AIMarketMaker):
        """Process trade event"""
        trade_data = event['data']
        
        # Check if any of our orders were filled
        # This is simplified - in reality would need order matching engine
        if self.order_queue:
            self._check_order_fills(trade_data, strategy, event['timestamp'])
    
    def _place_order(self, action: Dict, strategy: AIMarketMaker, timestamp: int):
        """Place order with latency simulation"""
        # Calculate order latency
        order_size = action['quantity'] / 0.1  # Normalize to typical size
        send_latency, proc_latency = self.latency_model.get_order_latency(
            order_type=1,  # Limit order
            order_size=order_size
        )
        
        # Schedule order arrival at exchange
        arrival_time = timestamp + int(send_latency + proc_latency)
        
        order_id = f"order_{len(self.order_queue)}"
        order = {
            'order_id': order_id,
            'arrival_time': arrival_time,
            'placement_time': timestamp,
            'side': action['side'],
            'price': action['price'],
            'quantity': action['quantity'],
            'status': 'pending'
        }
        
        self.order_queue[order_id] = order
        
        # Schedule order arrival event
        self.event_queue.append({
            'type': 'order_arrival',
            'timestamp': arrival_time,
            'order_id': order_id
        })
    
    def _cancel_order(self, action: Dict, strategy: AIMarketMaker):
        """Process order cancellation"""
        order_id = action.get('order_id')
        if order_id and order_id in self.order_queue:
            order = self.order_queue[order_id]
            order['status'] = 'cancelled'
            del self.order_queue[order_id]
            
            # Remove from strategy's active orders if tracking
            if hasattr(strategy, 'active_orders') and order_id in strategy.active_orders:
                del strategy.active_orders[order_id]
    
    def _check_order_fills(self, trade_data: Dict, strategy: AIMarketMaker, timestamp: int):
        """Check if any orders were filled"""
        trade_price = trade_data['price']
        trade_quantity = trade_data['quantity']
        
        for order_id, order in list(self.order_queue.items()):
            if order['status'] != 'active':
                continue
                
            # Simple fill logic - would be more complex in reality
            if order['side'] == 'buy' and trade_price <= order['price']:
                # Buy order filled
                fill_quantity = min(order['quantity'], trade_quantity * 0.1)  # Partial fill
                self._execute_fill(order, fill_quantity, trade_price, strategy, timestamp)
            elif order['side'] == 'sell' and trade_price >= order['price']:
                # Sell order filled
                fill_quantity = min(order['quantity'], trade_quantity * 0.1)
                self._execute_fill(order, fill_quantity, trade_price, strategy, timestamp)
    
    def _execute_fill(
        self,
        order: Dict,
        quantity: float,
        price: float,
        strategy: AIMarketMaker,
        timestamp: int
    ):
        """Execute order fill"""
        # Calculate commission
        commission = quantity * price * self.commission_rate
        
        # Update capital
        if order['side'] == 'buy':
            self.capital -= quantity * price + commission
        else:
            self.capital += quantity * price - commission
        
        # Record fill
        fill = {
            'timestamp': timestamp,
            'order_id': order['order_id'],
            'side': order['side'],
            'price': price,
            'quantity': quantity,
            'commission': commission
        }
        self.fills.append(fill)
        
        # Notify strategy
        strategy.on_trade(fill, timestamp)
        
        # Update order
        order['quantity'] -= quantity
        if order['quantity'] <= 0:
            del self.order_queue[order['order_id']]
    
    def _update_equity(self, strategy: AIMarketMaker):
        """Update equity curve"""
        # Get current position value
        position_value = 0
        if strategy.position.quantity != 0:
            mid_price = self.orderbook.get_mid_price()
            if mid_price:
                position_value = strategy.position.quantity * mid_price
        
        # Calculate total equity
        equity = self.capital + position_value
        
        self.equity_curve.append({
            'timestamp': self.current_time,
            'equity': equity,
            'capital': self.capital,
            'position_value': position_value,
            'position_size': strategy.position.quantity
        })
    
    def _calculate_metrics(self, strategy: AIMarketMaker):
        """Calculate performance metrics"""
        # Get strategy metrics
        strategy_metrics = strategy.get_performance_metrics()
        
        # Calculate backtester metrics
        equity_df = pd.DataFrame(self.equity_curve)
        returns = equity_df['equity'].pct_change().dropna()
        
        self.metrics = {
            **strategy_metrics,
            'total_return': (self.capital - self.equity_curve[0]['equity']) / self.equity_curve[0]['equity'],
            'annual_return': self._calculate_annual_return(returns),
            'volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'max_drawdown': self._calculate_max_drawdown(equity_df['equity']),
            'total_trades': len(self.fills),
            'total_commission': sum(f['commission'] for f in self.fills)
        }
    
    def _calculate_annual_return(self, returns: pd.Series) -> float:
        """Calculate annualized return"""
        if len(returns) == 0:
            return 0
        
        total_return = (1 + returns).prod() - 1
        n_days = len(returns) / (24 * 60 * 60 / 1000)  # Convert from ms to days
        years = n_days / 365
        
        return (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0
        
        excess_returns = returns - risk_free_rate / 252
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    
    def _calculate_max_drawdown(self, equity: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cummax = equity.expanding().max()
        drawdown = (equity - cummax) / cummax
        return abs(drawdown.min())
    
    def _parse_orderbook_updates(self, data: pd.Series) -> List[Dict]:
        """Parse orderbook update data"""
        updates = []
        
        # Parse bid updates
        if 'bids' in data:
            for bid in data['bids']:
                updates.append({
                    'side': 'bid',
                    'price': bid[0],
                    'quantity': bid[1]
                })
        
        # Parse ask updates
        if 'asks' in data:
            for ask in data['asks']:
                updates.append({
                    'side': 'ask',
                    'price': ask[0],
                    'quantity': ask[1]
                })
        
        return updates
    
    def _process_order_arrival(self, event: Dict, strategy: AIMarketMaker):
        """Process order arrival at exchange"""
        order_id = event['order_id']
        if order_id in self.order_queue:
            self.order_queue[order_id]['status'] = 'active'


class BacktestResults:
    """Container for backtest results"""
    
    def __init__(
        self,
        strategy_name: str,
        symbol: str,
        start_date: str,
        end_date: str,
        initial_capital: float,
        final_capital: float,
        trades: List[Dict],
        fills: List[Dict],
        equity_curve: pd.DataFrame,
        metrics: Dict
    ):
        self.strategy_name = strategy_name
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.final_capital = final_capital
        self.trades = trades
        self.fills = fills
        self.equity_curve = equity_curve
        self.metrics = metrics
    
    def plot_performance(self):
        """Plot performance charts"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Equity curve
        axes[0].plot(self.equity_curve['timestamp'], self.equity_curve['equity'])
        axes[0].set_title('Equity Curve')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Equity')
        
        # Position size
        axes[1].plot(self.equity_curve['timestamp'], self.equity_curve['position_size'])
        axes[1].set_title('Position Size')
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Position')
        axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # Drawdown
        cummax = self.equity_curve['equity'].expanding().max()
        drawdown = (self.equity_curve['equity'] - cummax) / cummax * 100
        axes[2].fill_between(self.equity_curve['timestamp'], drawdown, 0, alpha=0.3, color='red')
        axes[2].set_title('Drawdown %')
        axes[2].set_xlabel('Time')
        axes[2].set_ylabel('Drawdown %')
        
        plt.tight_layout()
        plt.show()
    
    def print_statistics(self):
        """Print performance statistics"""
        print(f"\n=== Backtest Results for {self.strategy_name} ===")
        print(f"Symbol: {self.symbol}")
        print(f"Period: {self.start_date} to {self.end_date}")
        print(f"\nPerformance Metrics:")
        print(f"  Initial Capital: ${self.initial_capital:,.2f}")
        print(f"  Final Capital: ${self.final_capital:,.2f}")
        print(f"  Total Return: {self.metrics.get('total_return', 0)*100:.2f}%")
        print(f"  Annual Return: {self.metrics.get('annual_return', 0)*100:.2f}%")
        print(f"  Sharpe Ratio: {self.metrics.get('sharpe_ratio', 0):.2f}")
        print(f"  Max Drawdown: {self.metrics.get('max_drawdown', 0)*100:.2f}%")
        print(f"\nTrading Statistics:")
        print(f"  Total Trades: {self.metrics.get('total_trades', 0)}")
        print(f"  Win Rate: {self.metrics.get('win_rate', 0)*100:.2f}%")
        print(f"  Avg Position: {self.metrics.get('avg_position', 0):.4f}")
        print(f"  Total Commission: ${self.metrics.get('total_commission', 0):.2f}")
    
    def to_csv(self, filepath: str):
        """Export results to CSV"""
        # Export equity curve
        self.equity_curve.to_csv(f"{filepath}_equity.csv", index=False)
        
        # Export trades
        pd.DataFrame(self.fills).to_csv(f"{filepath}_trades.csv", index=False)
        
        # Export metrics
        pd.Series(self.metrics).to_csv(f"{filepath}_metrics.csv")