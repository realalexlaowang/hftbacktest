"""Core simulation engine with order queue modeling"""

import numpy as np
from numba import jit, types
from typing import Dict, List, Tuple, Optional
import heapq
from dataclasses import dataclass, field


@dataclass
class SimulatedOrder:
    """Order representation in simulation"""
    order_id: str
    timestamp: int
    side: str
    price: float
    quantity: float
    remaining: float
    queue_position: int = 0
    status: str = 'pending'
    
    def __lt__(self, other):
        """For heap operations"""
        if self.side == 'buy':
            return self.price > other.price  # Higher price has priority for buys
        else:
            return self.price < other.price  # Lower price has priority for sells


@jit(nopython=True)
def calculate_queue_position(
    order_price: float,
    order_quantity: float,
    order_timestamp: int,
    existing_orders: np.ndarray,  # [price, quantity, timestamp]
    side: int  # 0 for buy, 1 for sell
) -> int:
    """
    Calculate queue position for a new order
    
    Args:
        order_price: Price of new order
        order_quantity: Size of new order
        order_timestamp: Timestamp of new order
        existing_orders: Array of existing orders at same price level
        side: 0 for buy, 1 for sell
    
    Returns:
        Queue position (0-based)
    """
    position = 0
    
    for i in range(len(existing_orders)):
        if existing_orders[i, 0] == order_price:
            # Same price level
            if existing_orders[i, 2] < order_timestamp:
                # Earlier order has priority
                position += int(existing_orders[i, 1])
            elif existing_orders[i, 2] == order_timestamp:
                # Same timestamp, use pro-rata
                position += int(existing_orders[i, 1] * 0.5)
    
    return position


@jit(nopython=True)
def calculate_fill_probability(
    queue_position: int,
    order_quantity: float,
    trade_quantity: float,
    total_quantity_at_level: float,
    market_impact_factor: float = 0.1
) -> float:
    """
    Calculate probability of order fill based on queue position
    
    Args:
        queue_position: Position in queue
        order_quantity: Size of our order
        trade_quantity: Size of incoming trade
        total_quantity_at_level: Total quantity at price level
        market_impact_factor: Impact of large trades
    
    Returns:
        Fill probability between 0 and 1
    """
    if queue_position == 0:
        # First in queue
        if trade_quantity >= order_quantity:
            return 1.0
        else:
            return trade_quantity / order_quantity
    
    # Calculate how much quantity is ahead
    quantity_ahead = float(queue_position)
    
    # Simple fill model
    if trade_quantity <= quantity_ahead:
        # Trade won't reach our order
        return 0.0
    elif trade_quantity >= quantity_ahead + order_quantity:
        # Full fill
        return 1.0
    else:
        # Partial fill
        remaining_quantity = trade_quantity - quantity_ahead
        fill_ratio = remaining_quantity / order_quantity
        
        # Adjust for market impact
        impact_adjustment = 1.0 - market_impact_factor * (trade_quantity / total_quantity_at_level)
        
        return min(1.0, fill_ratio * impact_adjustment)


@jit(nopython=True)
def calculate_slippage(
    order_price: float,
    market_price: float,
    order_quantity: float,
    market_depth: float,
    volatility: float,
    impact_coefficient: float = 0.1
) -> float:
    """
    Calculate execution slippage
    
    Args:
        order_price: Intended execution price
        market_price: Current market price
        order_quantity: Order size
        market_depth: Available liquidity
        volatility: Current market volatility
        impact_coefficient: Price impact coefficient
    
    Returns:
        Actual execution price after slippage
    """
    # Base slippage from price movement
    base_slippage = abs(market_price - order_price) / order_price
    
    # Size impact
    size_impact = impact_coefficient * (order_quantity / market_depth) ** 2
    
    # Volatility adjustment
    volatility_adjustment = volatility * np.sqrt(order_quantity / market_depth)
    
    # Total slippage
    total_slippage = base_slippage + size_impact + volatility_adjustment
    
    # Apply slippage
    if order_price > market_price:  # Buy order
        execution_price = order_price * (1 + total_slippage)
    else:  # Sell order
        execution_price = order_price * (1 - total_slippage)
    
    return execution_price


class SimulationEngine:
    """High-fidelity market simulation engine"""
    
    def __init__(self, tick_size: float = 0.01, lot_size: float = 0.00001):
        """
        Initialize simulation engine
        
        Args:
            tick_size: Minimum price increment
            lot_size: Minimum quantity increment
        """
        self.tick_size = tick_size
        self.lot_size = lot_size
        
        # Order management
        self.order_id_counter = 0
        self.active_orders: Dict[str, SimulatedOrder] = {}
        self.order_history: List[SimulatedOrder] = []
        
        # Market state
        self.last_trade_price = 0.0
        self.total_volume = 0.0
        self.volatility = 0.0
        
        # Queue tracking
        self.buy_queues: Dict[float, List[SimulatedOrder]] = {}
        self.sell_queues: Dict[float, List[SimulatedOrder]] = {}
        
    def place_order(
        self,
        side: str,
        price: float,
        quantity: float,
        timestamp: int,
        order_type: str = 'limit'
    ) -> str:
        """
        Place a new order in the simulation
        
        Args:
            side: 'buy' or 'sell'
            price: Order price
            quantity: Order quantity
            timestamp: Order timestamp
            order_type: 'limit' or 'market'
        
        Returns:
            Order ID
        """
        # Round to tick/lot size
        price = round(price / self.tick_size) * self.tick_size
        quantity = round(quantity / self.lot_size) * self.lot_size
        
        # Generate order ID
        order_id = f"SIM_{self.order_id_counter}"
        self.order_id_counter += 1
        
        # Calculate queue position
        queue_position = self._calculate_initial_queue_position(side, price, quantity, timestamp)
        
        # Create order
        order = SimulatedOrder(
            order_id=order_id,
            timestamp=timestamp,
            side=side,
            price=price,
            quantity=quantity,
            remaining=quantity,
            queue_position=queue_position,
            status='active'
        )
        
        # Add to active orders
        self.active_orders[order_id] = order
        
        # Add to price queue
        if side == 'buy':
            if price not in self.buy_queues:
                self.buy_queues[price] = []
            self.buy_queues[price].append(order)
            self.buy_queues[price].sort(key=lambda x: x.queue_position)
        else:
            if price not in self.sell_queues:
                self.sell_queues[price] = []
            self.sell_queues[price].append(order)
            self.sell_queues[price].sort(key=lambda x: x.queue_position)
        
        return order_id
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order
        
        Args:
            order_id: ID of order to cancel
        
        Returns:
            True if cancelled, False if not found
        """
        if order_id not in self.active_orders:
            return False
        
        order = self.active_orders[order_id]
        order.status = 'cancelled'
        
        # Remove from queues
        if order.side == 'buy' and order.price in self.buy_queues:
            self.buy_queues[order.price] = [
                o for o in self.buy_queues[order.price] if o.order_id != order_id
            ]
            if not self.buy_queues[order.price]:
                del self.buy_queues[order.price]
        elif order.side == 'sell' and order.price in self.sell_queues:
            self.sell_queues[order.price] = [
                o for o in self.sell_queues[order.price] if o.order_id != order_id
            ]
            if not self.sell_queues[order.price]:
                del self.sell_queues[order.price]
        
        # Move to history
        del self.active_orders[order_id]
        self.order_history.append(order)
        
        return True
    
    def process_trade(
        self,
        price: float,
        quantity: float,
        aggressor_side: str,
        timestamp: int
    ) -> List[Dict]:
        """
        Process a market trade and check for fills
        
        Args:
            price: Trade price
            quantity: Trade quantity
            aggressor_side: Side of aggressor ('buy' or 'sell')
            timestamp: Trade timestamp
        
        Returns:
            List of fill events
        """
        fills = []
        remaining_quantity = quantity
        
        # Update market state
        self.last_trade_price = price
        self.total_volume += quantity
        
        # Determine which queue to check
        if aggressor_side == 'buy':
            # Check sell orders at or below trade price
            prices_to_check = sorted([p for p in self.sell_queues.keys() if p <= price])
        else:
            # Check buy orders at or above trade price
            prices_to_check = sorted([p for p in self.buy_queues.keys() if p >= price], reverse=True)
        
        # Process potential fills
        for check_price in prices_to_check:
            if remaining_quantity <= 0:
                break
            
            queue = self.sell_queues[check_price] if aggressor_side == 'buy' else self.buy_queues[check_price]
            
            # Calculate total quantity at this level
            total_at_level = sum(o.remaining for o in queue)
            
            # Process orders in queue
            for order in queue[:]:  # Copy to allow modification
                if remaining_quantity <= 0:
                    break
                
                # Calculate fill probability
                fill_prob = calculate_fill_probability(
                    order.queue_position,
                    order.remaining,
                    remaining_quantity,
                    total_at_level,
                    0.1  # market impact factor
                )
                
                if np.random.random() < fill_prob:
                    # Calculate fill quantity
                    fill_quantity = min(order.remaining, remaining_quantity * fill_prob)
                    
                    # Calculate slippage
                    execution_price = calculate_slippage(
                        order.price,
                        price,
                        fill_quantity,
                        total_at_level,
                        self.volatility,
                        0.1  # impact coefficient
                    )
                    
                    # Create fill event
                    fill = {
                        'order_id': order.order_id,
                        'timestamp': timestamp,
                        'price': execution_price,
                        'quantity': fill_quantity,
                        'side': order.side
                    }
                    fills.append(fill)
                    
                    # Update order
                    order.remaining -= fill_quantity
                    remaining_quantity -= fill_quantity
                    
                    if order.remaining <= 0:
                        order.status = 'filled'
                        del self.active_orders[order.order_id]
                        self.order_history.append(order)
                        queue.remove(order)
            
            # Clean up empty queues
            if not queue:
                if aggressor_side == 'buy':
                    del self.sell_queues[check_price]
                else:
                    del self.buy_queues[check_price]
        
        return fills
    
    def update_volatility(self, returns: np.ndarray, window: int = 100):
        """Update volatility estimate"""
        if len(returns) >= window:
            self.volatility = np.std(returns[-window:])
    
    def get_order_book_state(self) -> Dict:
        """Get current order book state"""
        bids = []
        asks = []
        
        # Aggregate by price level
        for price in sorted(self.buy_queues.keys(), reverse=True)[:10]:
            total_qty = sum(o.remaining for o in self.buy_queues[price])
            bids.append([price, total_qty])
        
        for price in sorted(self.sell_queues.keys())[:10]:
            total_qty = sum(o.remaining for o in self.sell_queues[price])
            asks.append([price, total_qty])
        
        return {
            'bids': np.array(bids) if bids else np.empty((0, 2)),
            'asks': np.array(asks) if asks else np.empty((0, 2))
        }
    
    def _calculate_initial_queue_position(
        self,
        side: str,
        price: float,
        quantity: float,
        timestamp: int
    ) -> int:
        """Calculate initial queue position for new order"""
        queue = self.buy_queues.get(price, []) if side == 'buy' else self.sell_queues.get(price, [])
        
        if not queue:
            return 0
        
        # Sum quantity of orders already in queue
        position = sum(o.remaining for o in queue)
        
        return int(position)