"""
高级Level-3订单簿实现
提供Market-By-Order精确模拟，支持FIFO队列管理
"""

import numpy as np
import time
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import heapq
import uuid
from datetime import datetime
import numba
from numba import types
from numba.typed import Dict as TypedDict, List as TypedList

from .microsecond_precision import HighPrecisionTimestamp

class OrderType(Enum):
    """订单类型"""
    LIMIT = "LIMIT"
    MARKET = "MARKET"
    STOP_LIMIT = "STOP_LIMIT"
    ICEBERG = "ICEBERG"
    FILL_OR_KILL = "FOK"
    IMMEDIATE_OR_CANCEL = "IOC"

class OrderStatus(Enum):
    """订单状态"""
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

class OrderSide(Enum):
    """订单方向"""
    BUY = "BUY"
    SELL = "SELL"

@dataclass
class Level3Order:
    """Level-3订单 - 包含完整订单信息"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    price: float
    original_quantity: float
    remaining_quantity: float
    filled_quantity: float = 0.0
    timestamp: HighPrecisionTimestamp = field(default_factory=HighPrecisionTimestamp.now)
    sequence_number: int = 0
    client_order_id: str = ""
    account_id: str = ""
    status: OrderStatus = OrderStatus.PENDING
    
    # 高级属性
    time_in_force: str = "GTC"  # GTC, IOC, FOK, DAY
    iceberg_quantity: float = 0.0  # 冰山订单显示数量
    stop_price: float = 0.0  # 止损价格
    execution_priority: int = 0  # 执行优先级
    
    # 性能统计
    queue_position: int = 0  # 队列位置
    estimated_fill_time: float = 0.0  # 预计成交时间(秒)
    cumulative_fees: float = 0.0  # 累计手续费
    
    def __post_init__(self):
        if not self.order_id:
            self.order_id = str(uuid.uuid4())
        if not self.client_order_id:
            self.client_order_id = self.order_id
    
    @property
    def is_buy(self) -> bool:
        return self.side == OrderSide.BUY
    
    @property
    def is_sell(self) -> bool:
        return self.side == OrderSide.SELL
    
    @property
    def is_active(self) -> bool:
        return self.status in [OrderStatus.ACTIVE, OrderStatus.PARTIALLY_FILLED]
    
    @property
    def fill_percentage(self) -> float:
        return self.filled_quantity / self.original_quantity if self.original_quantity > 0 else 0.0

@dataclass
class TradeExecution:
    """成交记录"""
    trade_id: str
    buy_order_id: str
    sell_order_id: str
    symbol: str
    price: float
    quantity: float
    timestamp: HighPrecisionTimestamp
    sequence_number: int
    buyer_fee: float = 0.0
    seller_fee: float = 0.0
    is_buyer_maker: bool = False
    
    def __post_init__(self):
        if not self.trade_id:
            self.trade_id = str(uuid.uuid4())

class Level3OrderBook:
    """高级Level-3订单簿 - Market-By-Order精确模拟"""
    
    def __init__(self, symbol: str, max_price_levels: int = 1000):
        self.symbol = symbol
        self.max_price_levels = max_price_levels
        
        # 核心数据结构
        self.orders: Dict[str, Level3Order] = {}  # 所有活跃订单
        self.bid_levels: Dict[float, deque] = defaultdict(deque)  # 买盘价格层级
        self.ask_levels: Dict[float, deque] = defaultdict(deque)  # 卖盘价格层级
        
        # 索引和映射
        self.order_to_price: Dict[str, float] = {}  # 订单ID到价格映射
        self.price_to_orders: Dict[float, Set[str]] = defaultdict(set)  # 价格到订单集合
        
        # 排序价格列表 (用于快速访问最优价格)
        self.sorted_bid_prices: List[float] = []
        self.sorted_ask_prices: List[float] = []
        
        # 序列号管理
        self.sequence_number = 0
        self.trade_sequence = 0
        
        # 统计信息
        self.total_orders_added = 0
        self.total_orders_cancelled = 0
        self.total_trades_executed = 0
        self.total_volume_traded = 0.0
        
        # 时间戳
        self.last_update_time = HighPrecisionTimestamp.now()
        
        # 性能缓存
        self._best_bid_cache: Optional[float] = None
        self._best_ask_cache: Optional[float] = None
        self._spread_cache: Optional[float] = None
        self._mid_price_cache: Optional[float] = None
    
    def add_order(self, order: Level3Order) -> bool:
        """添加订单到Level-3订单簿"""
        try:
            # 验证订单
            if order.order_id in self.orders:
                return False
            
            if order.remaining_quantity <= 0:
                return False
            
            # 设置序列号
            self.sequence_number += 1
            order.sequence_number = self.sequence_number
            order.status = OrderStatus.ACTIVE
            order.timestamp = HighPrecisionTimestamp.now()
            
            # 处理市价单 - 立即执行
            if order.order_type == OrderType.MARKET:
                return self._execute_market_order(order)
            
            # 处理限价单
            if order.order_type == OrderType.LIMIT:
                return self._add_limit_order(order)
            
            # 处理其他订单类型
            return self._add_special_order(order)
            
        except Exception as e:
            print(f"添加订单失败: {e}")
            return False
    
    def _add_limit_order(self, order: Level3Order) -> bool:
        """添加限价单"""
        # 检查是否能立即成交
        if self._can_immediate_fill(order):
            return self._execute_aggressive_order(order)
        
        # 添加到订单簿
        self.orders[order.order_id] = order
        self.order_to_price[order.order_id] = order.price
        self.price_to_orders[order.price].add(order.order_id)
        
        if order.is_buy:
            self.bid_levels[order.price].append(order.order_id)
            self._update_sorted_bids()
        else:
            self.ask_levels[order.price].append(order.order_id)
            self._update_sorted_asks()
        
        # 计算队列位置
        order.queue_position = self._calculate_queue_position(order)
        order.estimated_fill_time = self._estimate_fill_time(order)
        
        self.total_orders_added += 1
        self._invalidate_cache()
        self.last_update_time = HighPrecisionTimestamp.now()
        
        return True
    
    def _can_immediate_fill(self, order: Level3Order) -> bool:
        """检查订单是否能立即成交"""
        if order.is_buy:
            best_ask = self.get_best_ask()
            return best_ask is not None and order.price >= best_ask
        else:
            best_bid = self.get_best_bid()
            return best_bid is not None and order.price <= best_bid
    
    def _execute_market_order(self, order: Level3Order) -> bool:
        """执行市价单"""
        trades = []
        remaining_qty = order.remaining_quantity
        
        if order.is_buy:
            # 买入市价单，从最低卖价开始匹配
            ask_prices = sorted(self.ask_levels.keys())
            for price in ask_prices:
                if remaining_qty <= 0:
                    break
                remaining_qty = self._match_at_price_level(order, price, remaining_qty, trades)
        else:
            # 卖出市价单，从最高买价开始匹配
            bid_prices = sorted(self.bid_levels.keys(), reverse=True)
            for price in bid_prices:
                if remaining_qty <= 0:
                    break
                remaining_qty = self._match_at_price_level(order, price, remaining_qty, trades)
        
        # 更新订单状态
        order.remaining_quantity = remaining_qty
        order.filled_quantity = order.original_quantity - remaining_qty
        
        if remaining_qty <= 0:
            order.status = OrderStatus.FILLED
        elif order.filled_quantity > 0:
            if order.time_in_force == "IOC":
                order.status = OrderStatus.CANCELLED  # IOC未完全成交部分取消
            else:
                order.status = OrderStatus.PARTIALLY_FILLED
        else:
            order.status = OrderStatus.REJECTED  # 无法成交
        
        return len(trades) > 0
    
    def _execute_aggressive_order(self, order: Level3Order) -> bool:
        """执行aggressive限价单(能立即成交的限价单)"""
        trades = []
        remaining_qty = order.remaining_quantity
        
        if order.is_buy:
            # 买入限价单，匹配价格 <= order.price 的卖单
            ask_prices = [p for p in sorted(self.ask_levels.keys()) if p <= order.price]
            for price in ask_prices:
                if remaining_qty <= 0:
                    break
                remaining_qty = self._match_at_price_level(order, price, remaining_qty, trades)
        else:
            # 卖出限价单，匹配价格 >= order.price 的买单
            bid_prices = [p for p in sorted(self.bid_levels.keys(), reverse=True) if p >= order.price]
            for price in bid_prices:
                if remaining_qty <= 0:
                    break
                remaining_qty = self._match_at_price_level(order, price, remaining_qty, trades)
        
        # 更新订单状态
        order.remaining_quantity = remaining_qty
        order.filled_quantity = order.original_quantity - remaining_qty
        
        if remaining_qty <= 0:
            order.status = OrderStatus.FILLED
        elif order.filled_quantity > 0:
            order.status = OrderStatus.PARTIALLY_FILLED
            # 剩余部分加入订单簿
            if order.time_in_force not in ["IOC", "FOK"]:
                return self._add_limit_order(order)
        
        return len(trades) > 0
    
    def _match_at_price_level(self, aggressive_order: Level3Order, price: float, 
                             remaining_qty: float, trades: List[TradeExecution]) -> float:
        """在特定价格层级匹配订单"""
        if aggressive_order.is_buy:
            order_queue = self.ask_levels[price]
        else:
            order_queue = self.bid_levels[price]
        
        while order_queue and remaining_qty > 0:
            passive_order_id = order_queue[0]  # FIFO: 取队首订单
            passive_order = self.orders.get(passive_order_id)
            
            if not passive_order or not passive_order.is_active:
                order_queue.popleft()  # 移除无效订单
                continue
            
            # 计算成交数量
            trade_qty = min(remaining_qty, passive_order.remaining_quantity)
            
            # 创建成交记录
            trade = self._create_trade(aggressive_order, passive_order, price, trade_qty)
            trades.append(trade)
            
            # 更新订单状态
            passive_order.remaining_quantity -= trade_qty
            passive_order.filled_quantity += trade_qty
            aggressive_order.filled_quantity += trade_qty
            remaining_qty -= trade_qty
            
            # 检查被动订单是否完全成交
            if passive_order.remaining_quantity <= 0:
                passive_order.status = OrderStatus.FILLED
                order_queue.popleft()  # 从队列移除
                self._remove_order_from_indexes(passive_order)
            
            self.total_trades_executed += 1
            self.total_volume_traded += trade_qty
        
        # 清理空的价格层级
        if not order_queue:
            if aggressive_order.is_buy:
                del self.ask_levels[price]
                self._update_sorted_asks()
            else:
                del self.bid_levels[price]
                self._update_sorted_bids()
        
        return remaining_qty
    
    def _create_trade(self, aggressive_order: Level3Order, passive_order: Level3Order,
                     price: float, quantity: float) -> TradeExecution:
        """创建成交记录"""
        self.trade_sequence += 1
        
        # 确定买卖方
        if aggressive_order.is_buy:
            buy_order_id = aggressive_order.order_id
            sell_order_id = passive_order.order_id
            is_buyer_maker = False  # aggressive order是taker
        else:
            buy_order_id = passive_order.order_id
            sell_order_id = aggressive_order.order_id
            is_buyer_maker = True  # passive order是maker
        
        trade = TradeExecution(
            trade_id=f"{self.symbol}_{self.trade_sequence}",
            buy_order_id=buy_order_id,
            sell_order_id=sell_order_id,
            symbol=self.symbol,
            price=price,
            quantity=quantity,
            timestamp=HighPrecisionTimestamp.now(),
            sequence_number=self.trade_sequence,
            is_buyer_maker=is_buyer_maker
        )
        
        return trade
    
    def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        order = self.orders.get(order_id)
        if not order or not order.is_active:
            return False
        
        # 从价格层级移除
        price = self.order_to_price[order_id]
        if order.is_buy:
            queue = self.bid_levels[price]
        else:
            queue = self.ask_levels[price]
        
        # 从队列中移除 (可能不在队首)
        try:
            queue.remove(order_id)
        except ValueError:
            pass  # 订单可能已经被移除
        
        # 清理空的价格层级
        if not queue:
            if order.is_buy:
                del self.bid_levels[price]
                self._update_sorted_bids()
            else:
                del self.ask_levels[price]
                self._update_sorted_asks()
        
        # 更新订单状态
        order.status = OrderStatus.CANCELLED
        self._remove_order_from_indexes(order)
        
        self.total_orders_cancelled += 1
        self._invalidate_cache()
        self.last_update_time = HighPrecisionTimestamp.now()
        
        return True
    
    def modify_order(self, order_id: str, new_price: Optional[float] = None,
                    new_quantity: Optional[float] = None) -> bool:
        """修改订单 - 先取消再重新添加"""
        order = self.orders.get(order_id)
        if not order or not order.is_active:
            return False
        
        # 保存原始信息
        original_order = Level3Order(
            order_id=str(uuid.uuid4()),  # 新订单ID
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            price=new_price if new_price is not None else order.price,
            original_quantity=new_quantity if new_quantity is not None else order.remaining_quantity,
            remaining_quantity=new_quantity if new_quantity is not None else order.remaining_quantity,
            client_order_id=order.client_order_id,
            account_id=order.account_id,
            time_in_force=order.time_in_force
        )
        
        # 取消原订单
        if self.cancel_order(order_id):
            # 添加新订单
            return self.add_order(original_order)
        
        return False
    
    def get_order_queue_position(self, order_id: str) -> int:
        """获取订单在队列中的位置"""
        order = self.orders.get(order_id)
        if not order:
            return -1
        
        price = self.order_to_price[order_id]
        if order.is_buy:
            queue = self.bid_levels[price]
        else:
            queue = self.ask_levels[price]
        
        try:
            return list(queue).index(order_id) + 1  # 1-based position
        except ValueError:
            return -1
    
    def _calculate_queue_position(self, order: Level3Order) -> int:
        """计算订单的队列位置"""
        if order.is_buy:
            queue = self.bid_levels[order.price]
        else:
            queue = self.ask_levels[order.price]
        
        return len(queue)  # 新订单总是排在队尾
    
    def _estimate_fill_time(self, order: Level3Order) -> float:
        """估算订单成交时间"""
        queue_position = order.queue_position
        
        # 简化模型：基于队列位置和历史成交速度
        avg_trade_interval = 0.1  # 假设平均0.1秒一个交易
        return queue_position * avg_trade_interval
    
    def get_best_bid(self) -> Optional[float]:
        """获取最优买价"""
        if self._best_bid_cache is not None:
            return self._best_bid_cache
        
        if self.sorted_bid_prices:
            self._best_bid_cache = self.sorted_bid_prices[0]
            return self._best_bid_cache
        return None
    
    def get_best_ask(self) -> Optional[float]:
        """获取最优卖价"""
        if self._best_ask_cache is not None:
            return self._best_ask_cache
        
        if self.sorted_ask_prices:
            self._best_ask_cache = self.sorted_ask_prices[0]
            return self._best_ask_cache
        return None
    
    def get_spread(self) -> Optional[float]:
        """获取买卖价差"""
        if self._spread_cache is not None:
            return self._spread_cache
        
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid is not None and best_ask is not None:
            self._spread_cache = best_ask - best_bid
            return self._spread_cache
        return None
    
    def get_mid_price(self) -> Optional[float]:
        """获取中间价"""
        if self._mid_price_cache is not None:
            return self._mid_price_cache
        
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid is not None and best_ask is not None:
            self._mid_price_cache = (best_bid + best_ask) / 2.0
            return self._mid_price_cache
        return None
    
    def get_level2_snapshot(self, depth: int = 10) -> Dict[str, List[Tuple[float, float, int]]]:
        """生成Level-2市场深度快照"""
        bids = []
        asks = []
        
        # 买盘
        for i, price in enumerate(self.sorted_bid_prices[:depth]):
            queue = self.bid_levels[price]
            total_quantity = sum(self.orders[oid].remaining_quantity for oid in queue 
                               if oid in self.orders and self.orders[oid].is_active)
            order_count = len([oid for oid in queue if oid in self.orders and self.orders[oid].is_active])
            if total_quantity > 0:
                bids.append((price, total_quantity, order_count))
        
        # 卖盘
        for i, price in enumerate(self.sorted_ask_prices[:depth]):
            queue = self.ask_levels[price]
            total_quantity = sum(self.orders[oid].remaining_quantity for oid in queue 
                               if oid in self.orders and self.orders[oid].is_active)
            order_count = len([oid for oid in queue if oid in self.orders and self.orders[oid].is_active])
            if total_quantity > 0:
                asks.append((price, total_quantity, order_count))
        
        return {
            'bids': bids,
            'asks': asks,
            'timestamp': self.last_update_time.to_microseconds(),
            'sequence': self.sequence_number
        }
    
    def _update_sorted_bids(self):
        """更新排序的买盘价格列表"""
        self.sorted_bid_prices = sorted([p for p in self.bid_levels.keys() if self.bid_levels[p]], 
                                       reverse=True)
    
    def _update_sorted_asks(self):
        """更新排序的卖盘价格列表"""
        self.sorted_ask_prices = sorted([p for p in self.ask_levels.keys() if self.ask_levels[p]])
    
    def _remove_order_from_indexes(self, order: Level3Order):
        """从所有索引中移除订单"""
        if order.order_id in self.orders:
            del self.orders[order.order_id]
        
        if order.order_id in self.order_to_price:
            price = self.order_to_price[order.order_id]
            del self.order_to_price[order.order_id]
            
            if price in self.price_to_orders:
                self.price_to_orders[price].discard(order.order_id)
                if not self.price_to_orders[price]:
                    del self.price_to_orders[price]
    
    def _invalidate_cache(self):
        """清除缓存"""
        self._best_bid_cache = None
        self._best_ask_cache = None
        self._spread_cache = None
        self._mid_price_cache = None
    
    def _add_special_order(self, order: Level3Order) -> bool:
        """处理特殊订单类型"""
        if order.order_type == OrderType.FILL_OR_KILL:
            # FOK: 全部成交或全部取消
            if self._can_fill_completely(order):
                return self._execute_aggressive_order(order)
            else:
                order.status = OrderStatus.CANCELLED
                return False
        
        elif order.order_type == OrderType.IMMEDIATE_OR_CANCEL:
            # IOC: 立即成交，剩余取消
            return self._execute_aggressive_order(order)
        
        elif order.order_type == OrderType.ICEBERG:
            # 冰山订单: 只显示部分数量
            visible_order = Level3Order(
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                order_type=OrderType.LIMIT,
                price=order.price,
                original_quantity=min(order.iceberg_quantity, order.remaining_quantity),
                remaining_quantity=min(order.iceberg_quantity, order.remaining_quantity),
                client_order_id=order.client_order_id,
                account_id=order.account_id
            )
            return self._add_limit_order(visible_order)
        
        return False
    
    def _can_fill_completely(self, order: Level3Order) -> bool:
        """检查订单是否能完全成交"""
        available_quantity = 0.0
        remaining_qty = order.remaining_quantity
        
        if order.is_buy:
            for price in sorted(self.ask_levels.keys()):
                if price > order.price:
                    break
                for order_id in self.ask_levels[price]:
                    passive_order = self.orders.get(order_id)
                    if passive_order and passive_order.is_active:
                        available_quantity += passive_order.remaining_quantity
                        if available_quantity >= remaining_qty:
                            return True
        else:
            for price in sorted(self.bid_levels.keys(), reverse=True):
                if price < order.price:
                    break
                for order_id in self.bid_levels[price]:
                    passive_order = self.orders.get(order_id)
                    if passive_order and passive_order.is_active:
                        available_quantity += passive_order.remaining_quantity
                        if available_quantity >= remaining_qty:
                            return True
        
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取订单簿统计信息"""
        active_orders = len([o for o in self.orders.values() if o.is_active])
        total_bid_quantity = sum(
            sum(self.orders[oid].remaining_quantity for oid in queue if oid in self.orders and self.orders[oid].is_active)
            for queue in self.bid_levels.values()
        )
        total_ask_quantity = sum(
            sum(self.orders[oid].remaining_quantity for oid in queue if oid in self.orders and self.orders[oid].is_active)
            for queue in self.ask_levels.values()
        )
        
        return {
            'symbol': self.symbol,
            'sequence_number': self.sequence_number,
            'active_orders': active_orders,
            'bid_levels': len(self.bid_levels),
            'ask_levels': len(self.ask_levels),
            'total_bid_quantity': total_bid_quantity,
            'total_ask_quantity': total_ask_quantity,
            'best_bid': self.get_best_bid(),
            'best_ask': self.get_best_ask(),
            'spread': self.get_spread(),
            'mid_price': self.get_mid_price(),
            'total_orders_added': self.total_orders_added,
            'total_orders_cancelled': self.total_orders_cancelled,
            'total_trades_executed': self.total_trades_executed,
            'total_volume_traded': self.total_volume_traded,
            'last_update_time': self.last_update_time.to_microseconds()
        }

def benchmark_level3_orderbook():
    """Level-3订单簿性能基准测试"""
    print("📚 Level-3订单簿性能基准测试")
    print("=" * 50)
    
    orderbook = Level3OrderBook("BTCUSDT")
    
    # 测试1: 批量添加订单
    print("\n📊 测试1: 批量添加限价单")
    orders = []
    base_price = 45000.0
    
    start_time = time.perf_counter()
    
    # 添加1000个买单和1000个卖单
    for i in range(1000):
        # 买单
        buy_order = Level3Order(
            order_id=f"buy_{i}",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            price=base_price - i * 0.1,
            original_quantity=1.0 + i * 0.01,
            remaining_quantity=1.0 + i * 0.01
        )
        orderbook.add_order(buy_order)
        
        # 卖单
        sell_order = Level3Order(
            order_id=f"sell_{i}",
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            price=base_price + 10 + i * 0.1,
            original_quantity=1.0 + i * 0.01,
            remaining_quantity=1.0 + i * 0.01
        )
        orderbook.add_order(sell_order)
    
    add_time = time.perf_counter() - start_time
    
    print(f"添加订单数: {orderbook.total_orders_added:,}")
    print(f"添加时间: {add_time:.4f}秒")
    print(f"添加速度: {orderbook.total_orders_added / add_time:,.0f} 订单/秒")
    
    # 测试2: 市价单执行
    print("\n📊 测试2: 市价单执行")
    start_time = time.perf_counter()
    
    # 执行100个市价买单
    for i in range(100):
        market_buy = Level3Order(
            order_id=f"market_buy_{i}",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            price=0.0,  # 市价单价格设为0
            original_quantity=0.5 + i * 0.01,
            remaining_quantity=0.5 + i * 0.01
        )
        orderbook.add_order(market_buy)
    
    market_time = time.perf_counter() - start_time
    
    print(f"市价单执行: {orderbook.total_trades_executed:,}")
    print(f"执行时间: {market_time:.4f}秒")
    print(f"执行速度: {orderbook.total_trades_executed / market_time:,.0f} 交易/秒")
    print(f"总成交量: {orderbook.total_volume_traded:.2f}")
    
    # 测试3: 订单簿快照生成
    print("\n📊 测试3: Level-2快照生成")
    start_time = time.perf_counter()
    
    snapshots = []
    for _ in range(1000):
        snapshot = orderbook.get_level2_snapshot(depth=20)
        snapshots.append(snapshot)
    
    snapshot_time = time.perf_counter() - start_time
    
    print(f"快照生成: {len(snapshots):,} 个")
    print(f"生成时间: {snapshot_time:.4f}秒")
    print(f"生成速度: {len(snapshots) / snapshot_time:,.0f} 快照/秒")
    
    # 测试4: 订单取消
    print("\n📊 测试4: 订单取消")
    start_time = time.perf_counter()
    
    # 取消500个订单
    cancelled_count = 0
    for i in range(0, 1000, 2):  # 每隔一个取消
        if orderbook.cancel_order(f"buy_{i}"):
            cancelled_count += 1
    
    cancel_time = time.perf_counter() - start_time
    
    print(f"取消订单数: {cancelled_count:,}")
    print(f"取消时间: {cancel_time:.4f}秒")
    print(f"取消速度: {cancelled_count / cancel_time:,.0f} 订单/秒")
    
    # 最终统计
    print("\n📈 最终统计")
    stats = orderbook.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")
    
    print("\n✅ Level-3订单簿测试完成!")
    
    return {
        'add_speed': orderbook.total_orders_added / add_time,
        'trade_speed': orderbook.total_trades_executed / market_time if market_time > 0 else 0,
        'snapshot_speed': len(snapshots) / snapshot_time,
        'cancel_speed': cancelled_count / cancel_time,
        'total_orders': orderbook.total_orders_added,
        'total_trades': orderbook.total_trades_executed,
        'total_volume': orderbook.total_volume_traded
    }

if __name__ == "__main__":
    # 运行Level-3订单簿基准测试
    results = benchmark_level3_orderbook()
    
    print(f"\n🎯 Level-3订单簿性能总结:")
    print(f"- 订单添加速度: {results['add_speed']:,.0f} 订单/秒")
    print(f"- 交易执行速度: {results['trade_speed']:,.0f} 交易/秒")
    print(f"- 快照生成速度: {results['snapshot_speed']:,.0f} 快照/秒")
    print(f"- 订单取消速度: {results['cancel_speed']:,.0f} 订单/秒")
    print(f"- 总订单数: {results['total_orders']:,}")
    print(f"- 总交易数: {results['total_trades']:,}")
    print(f"- 总成交量: {results['total_volume']:.2f}")