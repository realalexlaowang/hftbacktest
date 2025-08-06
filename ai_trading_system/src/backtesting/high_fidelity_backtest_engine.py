"""
高保真回测引擎
实现精确的市场微观结构模拟，包括延迟、滑点、排队逻辑等
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import heapq
import time
from datetime import datetime, timedelta
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)

class OrderType(Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderStatus(Enum):
    PENDING = "PENDING"
    PARTIAL_FILLED = "PARTIAL_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

@dataclass
class MarketDataTick:
    """市场数据tick"""
    timestamp: datetime
    symbol: str
    price: float
    volume: float
    side: str  # 'buy' or 'sell'
    
@dataclass
class OrderBookLevel:
    """订单簿档位"""
    price: float
    quantity: float
    order_count: int
    
@dataclass
class OrderBookSnapshot:
    """订单簿快照"""
    timestamp: datetime
    symbol: str
    bids: List[OrderBookLevel]  # 买盘，价格从高到低
    asks: List[OrderBookLevel]  # 卖盘，价格从低到高
    
@dataclass
class BacktestOrder:
    """回测订单"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    submit_time: datetime = field(default_factory=datetime.now)
    fill_time: Optional[datetime] = None
    fees: float = 0.0
    slippage: float = 0.0
    
    # 微观结构信息
    queue_position: int = 0  # 在该价位的排队位置
    maker_taker: str = ""    # "maker" or "taker"
    latency_ms: float = 0.0  # 订单延迟(毫秒)

@dataclass  
class Trade:
    """成交记录"""
    trade_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
    fees: float
    is_maker: bool

@dataclass
class LatencyModel:
    """延迟模型"""
    base_latency_ms: float = 10.0      # 基础延迟
    network_jitter_ms: float = 2.0     # 网络抖动
    exchange_processing_ms: float = 5.0 # 交易所处理延迟
    queue_delay_factor: float = 0.1     # 排队延迟因子
    
    def calculate_latency(self, queue_position: int = 0) -> float:
        """计算实际延迟"""
        base = self.base_latency_ms
        jitter = np.random.normal(0, self.network_jitter_ms)
        processing = self.exchange_processing_ms
        queue_delay = queue_position * self.queue_delay_factor
        
        return max(1.0, base + jitter + processing + queue_delay)

class OrderBook:
    """高保真订单簿模拟"""
    
    def __init__(self, symbol: str, tick_size: float = 0.01):
        self.symbol = symbol
        self.tick_size = tick_size
        
        # 订单簿数据结构：price -> {total_qty, orders: [order_ids]}
        self.bids: Dict[float, Dict] = defaultdict(lambda: {'quantity': 0.0, 'orders': deque(), 'order_count': 0})
        self.asks: Dict[float, Dict] = defaultdict(lambda: {'quantity': 0.0, 'orders': deque(), 'order_count': 0})
        
        # 活跃订单追踪
        self.active_orders: Dict[str, BacktestOrder] = {}
        
        # 最新价格
        self.last_price: float = 0.0
        self.bid_price: float = 0.0
        self.ask_price: float = 0.0
        
    def add_order(self, order: BacktestOrder) -> Tuple[bool, str]:
        """添加订单到订单簿"""
        if order.order_type != OrderType.LIMIT:
            return False, "Only limit orders can be added to order book"
            
        price = self._round_price(order.price)
        
        if order.side == OrderSide.BUY:
            self.bids[price]['quantity'] += order.quantity
            self.bids[price]['orders'].append(order.order_id)
            self.bids[price]['order_count'] += 1
            # 计算排队位置（当前价位的订单数量）
            order.queue_position = len(self.bids[price]['orders'])
        else:
            self.asks[price]['quantity'] += order.quantity  
            self.asks[price]['orders'].append(order.order_id)
            self.asks[price]['order_count'] += 1
            order.queue_position = len(self.asks[price]['orders'])
            
        self.active_orders[order.order_id] = order
        self._update_best_prices()
        
        return True, "Order added successfully"
    
    def remove_order(self, order_id: str) -> bool:
        """从订单簿移除订单"""
        if order_id not in self.active_orders:
            return False
            
        order = self.active_orders[order_id]
        price = self._round_price(order.price)
        
        if order.side == OrderSide.BUY:
            if price in self.bids and order_id in self.bids[price]['orders']:
                self.bids[price]['orders'].remove(order_id)
                self.bids[price]['quantity'] -= (order.quantity - order.filled_quantity)
                self.bids[price]['order_count'] -= 1
                if self.bids[price]['quantity'] <= 0:
                    del self.bids[price]
        else:
            if price in self.asks and order_id in self.asks[price]['orders']:
                self.asks[price]['orders'].remove(order_id)
                self.asks[price]['quantity'] -= (order.quantity - order.filled_quantity)
                self.asks[price]['order_count'] -= 1
                if self.asks[price]['quantity'] <= 0:
                    del self.asks[price]
                    
        del self.active_orders[order_id]
        self._update_best_prices()
        return True
    
    def match_market_order(self, order: BacktestOrder) -> List[Trade]:
        """撮合市价单"""
        trades = []
        remaining_qty = order.quantity
        
        # 选择对手盘
        if order.side == OrderSide.BUY:
            # 买单匹配卖盘，从最低价开始
            price_levels = sorted(self.asks.keys())
        else:
            # 卖单匹配买盘，从最高价开始  
            price_levels = sorted(self.bids.keys(), reverse=True)
            
        for price in price_levels:
            if remaining_qty <= 0:
                break
                
            if order.side == OrderSide.BUY:
                level = self.asks[price]
            else:
                level = self.bids[price]
                
            # 在该价位进行撮合
            level_qty = min(remaining_qty, level['quantity'])
            if level_qty > 0:
                trade = Trade(
                    trade_id=f"trade_{len(trades)}_{int(time.time()*1000)}",
                    order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=level_qty,
                    price=price,
                    timestamp=datetime.now(),
                    fees=0.0,  # 将在后续计算
                    is_maker=False  # 市价单总是taker
                )
                trades.append(trade)
                
                # 更新订单
                order.filled_quantity += level_qty
                order.avg_fill_price = ((order.avg_fill_price * (order.filled_quantity - level_qty)) + 
                                      (price * level_qty)) / order.filled_quantity
                
                # 更新订单簿
                level['quantity'] -= level_qty
                remaining_qty -= level_qty
                
                # 如果该价位完全成交，移除
                if level['quantity'] <= 0:
                    if order.side == OrderSide.BUY:
                        del self.asks[price]
                    else:
                        del self.bids[price]
        
        # 更新订单状态
        if order.filled_quantity >= order.quantity:
            order.status = OrderStatus.FILLED
        elif order.filled_quantity > 0:
            order.status = OrderStatus.PARTIAL_FILLED
            
        order.maker_taker = "taker"
        self._update_best_prices()
        
        return trades
    
    def process_market_data(self, tick: MarketDataTick) -> List[Trade]:
        """处理市场数据，触发订单撮合"""
        trades = []
        self.last_price = tick.price
        
        # 模拟外部流动性对订单簿的影响
        if tick.side == 'buy':
            # 买单消耗卖盘
            trades.extend(self._consume_liquidity(tick, is_buy=True))
        else:
            # 卖单消耗买盘
            trades.extend(self._consume_liquidity(tick, is_buy=False))
            
        self._update_best_prices()
        return trades
    
    def _consume_liquidity(self, tick: MarketDataTick, is_buy: bool) -> List[Trade]:
        """消耗流动性"""
        trades = []
        
        if is_buy:
            # 买单消耗ask side
            levels_to_remove = []
            for price in sorted(self.asks.keys()):
                if price <= tick.price:
                    level = self.asks[price]
                    # 模拟该价位的部分成交
                    consumed_qty = min(level['quantity'], tick.volume * 0.1)  # 假设消耗10%的tick量
                    
                    if consumed_qty > 0:
                        level['quantity'] -= consumed_qty
                        if level['quantity'] <= 0:
                            levels_to_remove.append(price)
                            
            for price in levels_to_remove:
                del self.asks[price]
        else:
            # 卖单消耗bid side
            levels_to_remove = []
            for price in sorted(self.bids.keys(), reverse=True):
                if price >= tick.price:
                    level = self.bids[price]
                    consumed_qty = min(level['quantity'], tick.volume * 0.1)
                    
                    if consumed_qty > 0:
                        level['quantity'] -= consumed_qty
                        if level['quantity'] <= 0:
                            levels_to_remove.append(price)
                            
            for price in levels_to_remove:
                del self.bids[price]
                
        return trades
    
    def _update_best_prices(self):
        """更新最优买卖价"""
        if self.bids:
            self.bid_price = max(self.bids.keys())
        else:
            self.bid_price = 0.0
            
        if self.asks:
            self.ask_price = min(self.asks.keys())
        else:
            self.ask_price = float('inf')
    
    def _round_price(self, price: float) -> float:
        """价格舍入到最小变动单位"""
        return round(price / self.tick_size) * self.tick_size
    
    def get_spread(self) -> float:
        """获取买卖价差"""
        if self.bid_price > 0 and self.ask_price < float('inf'):
            return self.ask_price - self.bid_price
        return 0.0
    
    def get_depth(self, levels: int = 10) -> OrderBookSnapshot:
        """获取订单簿深度"""
        bids = []
        asks = []
        
        # 获取买盘深度
        for price in sorted(self.bids.keys(), reverse=True)[:levels]:
            level = self.bids[price]
            bids.append(OrderBookLevel(
                price=price,
                quantity=level['quantity'],
                order_count=level['order_count']
            ))
            
        # 获取卖盘深度
        for price in sorted(self.asks.keys())[:levels]:
            level = self.asks[price]
            asks.append(OrderBookLevel(
                price=price,
                quantity=level['quantity'], 
                order_count=level['order_count']
            ))
            
        return OrderBookSnapshot(
            timestamp=datetime.now(),
            symbol=self.symbol,
            bids=bids,
            asks=asks
        )

class SlippageModel:
    """滑点模型"""
    
    def __init__(self):
        self.base_spread_ratio = 0.0001  # 基础价差比例
        self.impact_coefficient = 0.001   # 市场冲击系数
        self.temporary_impact_decay = 0.9 # 临时冲击衰减
        
    def calculate_slippage(self, order: BacktestOrder, orderbook: OrderBook, 
                          avg_volume: float) -> float:
        """计算滑点"""
        if order.order_type == OrderType.LIMIT and order.price:
            # 限价单滑点主要来自等待时间和价格变动
            return self._calculate_limit_slippage(order, orderbook)
        else:
            # 市价单滑点来自市场冲击
            return self._calculate_market_slippage(order, orderbook, avg_volume)
    
    def _calculate_limit_slippage(self, order: BacktestOrder, orderbook: OrderBook) -> float:
        """计算限价单滑点"""
        # 如果订单价格偏离市场价格，可能有滑点
        if order.side == OrderSide.BUY:
            if order.price < orderbook.ask_price:
                # 买单价格低于卖一，可能需要等待
                return max(0, (orderbook.ask_price - order.price) / order.price)
        else:
            if order.price > orderbook.bid_price:
                # 卖单价格高于买一，可能需要等待
                return max(0, (order.price - orderbook.bid_price) / order.price)
        return 0.0
    
    def _calculate_market_slippage(self, order: BacktestOrder, orderbook: OrderBook, 
                                 avg_volume: float) -> float:
        """计算市价单滑点"""
        # 基于订单大小和市场深度计算滑点
        market_impact = (order.quantity / avg_volume) * self.impact_coefficient
        
        # 基础价差成本
        spread_cost = orderbook.get_spread() / (2 * orderbook.last_price) if orderbook.last_price > 0 else 0
        
        return spread_cost + market_impact

class FeeModel:
    """手续费模型"""
    
    def __init__(self):
        # 币安费率结构
        self.maker_fee = 0.001   # 0.1% maker费率
        self.taker_fee = 0.001   # 0.1% taker费率
        
        # VIP等级费率（可根据交易量调整）
        self.vip_levels = {
            0: {'maker': 0.001, 'taker': 0.001},
            1: {'maker': 0.0009, 'taker': 0.001},
            2: {'maker': 0.0008, 'taker': 0.001},
            # ... 更多VIP等级
        }
        
    def calculate_fee(self, trade: Trade, vip_level: int = 0) -> float:
        """计算手续费"""
        rates = self.vip_levels.get(vip_level, self.vip_levels[0])
        
        if trade.is_maker:
            fee_rate = rates['maker']
        else:
            fee_rate = rates['taker']
            
        return trade.quantity * trade.price * fee_rate

class HighFidelityBacktestEngine:
    """高保真回测引擎"""
    
    def __init__(self, 
                 initial_balance: float = 100000.0,
                 symbols: List[str] = None,
                 latency_model: LatencyModel = None,
                 commission_model: FeeModel = None):
        
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.symbols = symbols or ['BTCUSDT']
        
        # 模型组件
        self.latency_model = latency_model or LatencyModel()
        self.fee_model = commission_model or FeeModel()
        self.slippage_model = SlippageModel()
        
        # 订单簿和数据
        self.order_books: Dict[str, OrderBook] = {}
        for symbol in self.symbols:
            self.order_books[symbol] = OrderBook(symbol)
            
        # 交易记录
        self.orders: Dict[str, BacktestOrder] = {}
        self.trades: List[Trade] = []
        self.positions: Dict[str, float] = defaultdict(float)
        
        # 性能统计
        self.pnl_history: List[float] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        
        # 事件队列（用于模拟延迟）
        self.event_queue: List[Tuple[datetime, Any]] = []
        
        # 当前时间
        self.current_time: datetime = datetime.now()
        
        # 市场数据统计
        self.volume_stats: Dict[str, List[float]] = defaultdict(list)
        
    def add_market_data(self, tick: MarketDataTick):
        """添加市场数据"""
        self.current_time = tick.timestamp
        
        # 更新成交量统计
        self.volume_stats[tick.symbol].append(tick.volume)
        if len(self.volume_stats[tick.symbol]) > 1000:
            self.volume_stats[tick.symbol].pop(0)  # 保持最近1000个数据点
            
        # 处理订单簿更新
        if tick.symbol in self.order_books:
            trades = self.order_books[tick.symbol].process_market_data(tick)
            self.trades.extend(trades)
            
        # 处理延迟事件
        self._process_delayed_events()
        
        # 更新权益曲线
        self._update_equity()
    
    def submit_order(self, order: BacktestOrder) -> str:
        """提交订单"""
        # 计算延迟
        latency_ms = self.latency_model.calculate_latency()
        order.latency_ms = latency_ms
        
        # 将订单加入延迟队列 - 修复时间戳操作
        if hasattr(self.current_time, 'add_milliseconds'):
            execution_time = self.current_time.add_milliseconds(latency_ms)
        else:
            execution_time = self.current_time + timedelta(milliseconds=latency_ms)
        
        self.event_queue.append((execution_time, 'order_execution', order))
        
        self.orders[order.order_id] = order
        
        logger.info(f"Order submitted: {order.order_id}, latency: {latency_ms:.2f}ms")
        return order.order_id
    
    def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        if order_id not in self.orders:
            return False
            
        order = self.orders[order_id]
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
            return False
            
        # 添加延迟
        latency_ms = self.latency_model.calculate_latency()
        execution_time = self.current_time + timedelta(milliseconds=latency_ms)
        self.event_queue.append((execution_time, 'order_cancellation', order_id))
        
        return True
    
    def _process_delayed_events(self):
        """处理延迟事件"""
        # 处理所有到期的事件
        while self.event_queue and self.event_queue[0][0] <= self.current_time:
            event_time, event_type, event_data = heapq.heappop(self.event_queue)
            
            if event_type == 'order_execution':
                self._execute_order(event_data)
            elif event_type == 'order_cancellation':
                self._cancel_order_immediate(event_data)
    
    def _execute_order(self, order: BacktestOrder):
        """立即执行订单（已考虑延迟）"""
        symbol = order.symbol
        
        if symbol not in self.order_books:
            order.status = OrderStatus.REJECTED
            return
            
        orderbook = self.order_books[symbol]
        
        # 检查资金充足性
        if not self._check_sufficient_funds(order):
            order.status = OrderStatus.REJECTED
            return
            
        if order.order_type == OrderType.MARKET:
            # 市价单立即成交
            trades = orderbook.match_market_order(order)
            for trade in trades:
                self._process_trade(trade, order)
                
        elif order.order_type == OrderType.LIMIT:
            # 限价单加入订单簿
            success, message = orderbook.add_order(order)
            if success:
                order.maker_taker = "maker"
                # 检查是否能立即成交
                if self._can_immediate_fill(order, orderbook):
                    trades = self._try_immediate_fill(order, orderbook)
                    for trade in trades:
                        self._process_trade(trade, order)
            else:
                order.status = OrderStatus.REJECTED
                
        # 计算滑点
        avg_volume = np.mean(self.volume_stats[symbol]) if self.volume_stats[symbol] else 1000.0
        order.slippage = self.slippage_model.calculate_slippage(order, orderbook, avg_volume)
    
    def _process_trade(self, trade: Trade, order: BacktestOrder):
        """处理成交"""
        # 计算手续费
        trade.fees = self.fee_model.calculate_fee(trade)
        
        # 更新持仓
        if trade.side == OrderSide.BUY:
            self.positions[trade.symbol] += trade.quantity
            self.balance -= (trade.quantity * trade.price + trade.fees)
        else:
            self.positions[trade.symbol] -= trade.quantity
            self.balance += (trade.quantity * trade.price - trade.fees)
            
        # 更新订单费用
        order.fees += trade.fees
        
        self.trades.append(trade)
        
        logger.info(f"Trade executed: {trade.trade_id}, "
                   f"price: {trade.price}, qty: {trade.quantity}, "
                   f"fees: {trade.fees:.4f}")
    
    def _check_sufficient_funds(self, order: BacktestOrder) -> bool:
        """检查资金充足性"""
        if order.side == OrderSide.BUY:
            required_balance = order.quantity * (order.price or self.order_books[order.symbol].ask_price)
            return self.balance >= required_balance
        else:
            # 卖出检查持仓
            return self.positions[order.symbol] >= order.quantity
    
    def _can_immediate_fill(self, order: BacktestOrder, orderbook: OrderBook) -> bool:
        """检查限价单是否能立即成交"""
        if order.side == OrderSide.BUY:
            return order.price >= orderbook.ask_price
        else:
            return order.price <= orderbook.bid_price
    
    def _try_immediate_fill(self, order: BacktestOrder, orderbook: OrderBook) -> List[Trade]:
        """尝试立即成交限价单"""
        # 这里简化处理，实际应该更复杂
        if self._can_immediate_fill(order, orderbook):
            return orderbook.match_market_order(order)
        return []
    
    def _cancel_order_immediate(self, order_id: str):
        """立即取消订单"""
        if order_id in self.orders:
            order = self.orders[order_id]
            if order.status == OrderStatus.PENDING:
                order.status = OrderStatus.CANCELLED
                # 从订单簿移除
                if order.symbol in self.order_books:
                    self.order_books[order.symbol].remove_order(order_id)
    
    def _update_equity(self):
        """更新权益曲线"""
        total_equity = self.balance
        
        # 加上持仓价值
        for symbol, position in self.positions.items():
            if symbol in self.order_books and position != 0:
                current_price = self.order_books[symbol].last_price
                total_equity += position * current_price
                
        self.equity_curve.append((self.current_time, total_equity))
        
        # 计算PnL
        pnl = total_equity - self.initial_balance
        self.pnl_history.append(pnl)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取回测性能统计"""
        if not self.equity_curve:
            return {}
            
        equity_values = [eq[1] for eq in self.equity_curve]
        returns = np.diff(equity_values) / equity_values[:-1]
        
        stats = {
            'total_return': (equity_values[-1] - equity_values[0]) / equity_values[0],
            'sharpe_ratio': np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(equity_values),
            'win_rate': self._calculate_win_rate(),
            'total_trades': len(self.trades),
            'total_fees': sum(trade.fees for trade in self.trades),
            'avg_trade_size': np.mean([trade.quantity * trade.price for trade in self.trades]) if self.trades else 0,
            'avg_latency_ms': np.mean([order.latency_ms for order in self.orders.values()]),
            'avg_slippage': np.mean([order.slippage for order in self.orders.values() if order.slippage > 0])
        }
        
        return stats
    
    def _calculate_max_drawdown(self, equity_values: List[float]) -> float:
        """计算最大回撤"""
        peak = equity_values[0]
        max_dd = 0.0
        
        for value in equity_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_dd = max(max_dd, drawdown)
            
        return max_dd
    
    def _calculate_win_rate(self) -> float:
        """计算胜率"""
        if not self.trades:
            return 0.0
            
        profitable_trades = 0
        for trade in self.trades:
            # 简化胜率计算，实际应该基于完整订单的PnL
            if trade.side == OrderSide.BUY:
                # 买入后价格上涨算盈利（简化）
                current_price = self.order_books[trade.symbol].last_price
                if current_price > trade.price:
                    profitable_trades += 1
            else:
                # 卖出后价格下跌算盈利
                current_price = self.order_books[trade.symbol].last_price
                if current_price < trade.price:
                    profitable_trades += 1
                    
        return profitable_trades / len(self.trades)