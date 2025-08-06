"""
多交易所回测引擎
支持跨交易所套利策略和精确的延迟模拟
"""

import asyncio
import numpy as np
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import uuid
from collections import defaultdict, deque
import heapq

from .microsecond_precision import HighPrecisionTimestamp, MicrosecondEventScheduler
from .level3_orderbook_advanced import Level3OrderBook, Level3Order, OrderSide, OrderType, OrderStatus, TradeExecution

class ExchangeType(Enum):
    """交易所类型"""
    BINANCE = "BINANCE"
    COINBASE = "COINBASE"
    HUOBI = "HUOBI"
    OKEX = "OKEX"
    KRAKEN = "KRAKEN"
    BITFINEX = "BITFINEX"

@dataclass
class ExchangeLatencyProfile:
    """交易所延迟配置"""
    name: str
    # 基础延迟 (微秒)
    rest_api_latency: Tuple[float, float]  # (min, max)
    websocket_latency: Tuple[float, float]  # (min, max)
    order_processing_latency: Tuple[float, float]  # (min, max)
    
    # 网络延迟
    network_jitter_std: float = 5.0  # 网络抖动标准差 (微秒)
    packet_loss_rate: float = 0.001  # 丢包率
    
    # 负载相关
    load_factor_base: float = 1.0  # 基础负载因子
    load_factor_std: float = 0.2   # 负载因子标准差
    
    # 交易所特定
    maker_fee: float = 0.001   # Maker手续费
    taker_fee: float = 0.001   # Taker手续费
    min_order_size: float = 0.001  # 最小订单量
    price_precision: int = 2   # 价格精度
    quantity_precision: int = 6  # 数量精度

@dataclass 
class CrossExchangeOrder:
    """跨交易所订单"""
    order_id: str
    exchange_name: str
    local_order: Level3Order
    timestamp: HighPrecisionTimestamp
    status: OrderStatus = OrderStatus.PENDING
    
    # 延迟追踪
    submit_latency_us: float = 0.0
    confirm_latency_us: float = 0.0
    total_latency_us: float = 0.0

@dataclass
class ArbitrageOpportunity:
    """套利机会"""
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    max_quantity: float
    profit_per_unit: float
    profit_percentage: float
    timestamp: HighPrecisionTimestamp
    
    @property
    def total_profit_potential(self) -> float:
        return self.profit_per_unit * self.max_quantity

class ExchangeSimulator(ABC):
    """交易所模拟器抽象基类"""
    
    def __init__(self, exchange_type: ExchangeType, latency_profile: ExchangeLatencyProfile):
        self.exchange_type = exchange_type
        self.name = exchange_type.value
        self.latency_profile = latency_profile
        
        # 核心组件
        self.orderbook = Level3OrderBook(f"{self.name}_BTCUSDT")
        self.event_scheduler = MicrosecondEventScheduler()
        
        # 订单管理
        self.pending_orders: Dict[str, CrossExchangeOrder] = {}
        self.completed_orders: Dict[str, CrossExchangeOrder] = {}
        
        # 统计信息
        self.total_orders_submitted = 0
        self.total_orders_filled = 0
        self.total_volume_traded = 0.0
        self.total_fees_collected = 0.0
        
        # 性能指标
        self.latency_samples: List[float] = []
        self.load_factor_history: List[float] = []
        
    @abstractmethod
    def get_current_latency(self) -> float:
        """获取当前延迟 (微秒)"""
        pass
    
    @abstractmethod
    def calculate_fees(self, price: float, quantity: float, is_maker: bool) -> float:
        """计算手续费"""
        pass
    
    def submit_order(self, order: Level3Order) -> CrossExchangeOrder:
        """提交订单到交易所"""
        # 创建跨交易所订单
        cross_order = CrossExchangeOrder(
            order_id=f"{self.name}_{order.order_id}",
            exchange_name=self.name,
            local_order=order,
            timestamp=HighPrecisionTimestamp.now()
        )
        
        # 计算提交延迟
        submit_latency = self.get_current_latency()
        cross_order.submit_latency_us = submit_latency
        
        # 调度订单处理事件
        self.event_scheduler.schedule_delay_event(
            int(submit_latency),
            "process_order",
            {"cross_order": cross_order}
        )
        
        self.pending_orders[cross_order.order_id] = cross_order
        self.total_orders_submitted += 1
        
        return cross_order
    
    def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        if order_id not in self.pending_orders:
            return False
        
        cross_order = self.pending_orders[order_id]
        cancel_latency = self.get_current_latency()
        
        # 调度取消事件
        self.event_scheduler.schedule_delay_event(
            int(cancel_latency),
            "cancel_order",
            {"order_id": order_id}
        )
        
        return True
    
    def process_pending_events(self, current_time: HighPrecisionTimestamp) -> List[Dict[str, Any]]:
        """处理待处理事件"""
        triggered_events = self.event_scheduler.advance_time_to(current_time)
        processed_events = []
        
        for event_type, event_data in triggered_events:
            if event_type == "process_order":
                result = self._process_order_event(event_data)
                processed_events.append({"type": "order_processed", "data": result})
            
            elif event_type == "cancel_order":
                result = self._process_cancel_event(event_data)
                processed_events.append({"type": "order_cancelled", "data": result})
        
        return processed_events
    
    def _process_order_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理订单事件"""
        cross_order = event_data["cross_order"]
        
        # 添加订单到本地订单簿
        success = self.orderbook.add_order(cross_order.local_order)
        
        if success:
            cross_order.status = OrderStatus.ACTIVE
            # 计算确认延迟
            confirm_latency = self.get_current_latency() * 0.5  # 确认通常更快
            cross_order.confirm_latency_us = confirm_latency
            cross_order.total_latency_us = cross_order.submit_latency_us + confirm_latency
            
            # 记录延迟样本
            self.latency_samples.append(cross_order.total_latency_us)
            
            return {
                "success": True,
                "order_id": cross_order.order_id,
                "latency_us": cross_order.total_latency_us,
                "status": cross_order.status
            }
        else:
            cross_order.status = OrderStatus.REJECTED
            self.completed_orders[cross_order.order_id] = cross_order
            del self.pending_orders[cross_order.order_id]
            
            return {
                "success": False,
                "order_id": cross_order.order_id,
                "reason": "Order rejected by exchange"
            }
    
    def _process_cancel_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理取消事件"""
        order_id = event_data["order_id"]
        
        if order_id in self.pending_orders:
            cross_order = self.pending_orders[order_id]
            
            # 从订单簿取消
            local_order_id = cross_order.local_order.order_id
            success = self.orderbook.cancel_order(local_order_id)
            
            if success:
                cross_order.status = OrderStatus.CANCELLED
                self.completed_orders[order_id] = cross_order
                del self.pending_orders[order_id]
                
                return {
                    "success": True,
                    "order_id": order_id,
                    "status": OrderStatus.CANCELLED
                }
        
        return {
            "success": False,
            "order_id": order_id,
            "reason": "Order not found or already processed"
        }
    
    def get_best_prices(self) -> Tuple[Optional[float], Optional[float]]:
        """获取最优买卖价"""
        return self.orderbook.get_best_bid(), self.orderbook.get_best_ask()
    
    def get_market_depth(self, depth: int = 10) -> Dict[str, Any]:
        """获取市场深度"""
        return self.orderbook.get_level2_snapshot(depth)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取交易所统计信息"""
        avg_latency = np.mean(self.latency_samples) if self.latency_samples else 0.0
        p95_latency = np.percentile(self.latency_samples, 95) if self.latency_samples else 0.0
        
        return {
            "exchange_name": self.name,
            "total_orders_submitted": self.total_orders_submitted,
            "total_orders_filled": self.total_orders_filled,
            "total_volume_traded": self.total_volume_traded,
            "total_fees_collected": self.total_fees_collected,
            "pending_orders": len(self.pending_orders),
            "avg_latency_us": avg_latency,
            "p95_latency_us": p95_latency,
            "orderbook_stats": self.orderbook.get_statistics()
        }

class BinanceSimulator(ExchangeSimulator):
    """Binance交易所模拟器"""
    
    def __init__(self):
        latency_profile = ExchangeLatencyProfile(
            name="Binance",
            rest_api_latency=(50, 150),    # 50-150微秒
            websocket_latency=(10, 30),    # 10-30微秒
            order_processing_latency=(100, 300),  # 100-300微秒
            network_jitter_std=5.0,
            packet_loss_rate=0.0005,
            maker_fee=0.001,  # 0.1%
            taker_fee=0.001,  # 0.1%
            min_order_size=0.00001,
            price_precision=2,
            quantity_precision=6
        )
        super().__init__(ExchangeType.BINANCE, latency_profile)
    
    def get_current_latency(self) -> float:
        """获取当前延迟"""
        # 基础延迟
        base_latency = np.random.uniform(*self.latency_profile.order_processing_latency)
        
        # 网络抖动
        jitter = np.random.normal(0, self.latency_profile.network_jitter_std)
        
        # 负载因子
        load_factor = np.random.normal(
            self.latency_profile.load_factor_base,
            self.latency_profile.load_factor_std
        )
        load_factor = max(0.5, load_factor)  # 最小0.5倍
        
        self.load_factor_history.append(load_factor)
        
        return max(10.0, (base_latency + jitter) * load_factor)  # 最小10微秒
    
    def calculate_fees(self, price: float, quantity: float, is_maker: bool) -> float:
        """计算Binance手续费"""
        notional = price * quantity
        fee_rate = self.latency_profile.maker_fee if is_maker else self.latency_profile.taker_fee
        
        # Binance VIP等级优惠 (简化)
        if notional > 100000:  # 大额交易优惠
            fee_rate *= 0.8
        
        return notional * fee_rate

class CoinbaseSimulator(ExchangeSimulator):
    """Coinbase Pro交易所模拟器"""
    
    def __init__(self):
        latency_profile = ExchangeLatencyProfile(
            name="Coinbase",
            rest_api_latency=(80, 200),    # 较高延迟
            websocket_latency=(15, 50),
            order_processing_latency=(150, 400),
            network_jitter_std=8.0,  # 更大抖动
            packet_loss_rate=0.001,
            maker_fee=0.005,  # 0.5%
            taker_fee=0.005,  # 0.5%
            min_order_size=0.001,
            price_precision=2,
            quantity_precision=8
        )
        super().__init__(ExchangeType.COINBASE, latency_profile)
    
    def get_current_latency(self) -> float:
        """获取当前延迟 - Coinbase特有模式"""
        # 基础延迟
        base_latency = np.random.uniform(*self.latency_profile.order_processing_latency)
        
        # 时间相关的延迟变化 (美国市场时间影响)
        current_hour = time.localtime().tm_hour
        if 9 <= current_hour <= 16:  # 美国交易时间
            time_multiplier = 1.3  # 高峰期延迟增加
        else:
            time_multiplier = 0.8  # 非高峰期延迟降低
        
        # 网络抖动
        jitter = np.random.normal(0, self.latency_profile.network_jitter_std)
        
        # 负载因子
        load_factor = np.random.normal(
            self.latency_profile.load_factor_base * time_multiplier,
            self.latency_profile.load_factor_std
        )
        load_factor = max(0.5, load_factor)
        
        self.load_factor_history.append(load_factor)
        
        return max(15.0, (base_latency + jitter) * load_factor)
    
    def calculate_fees(self, price: float, quantity: float, is_maker: bool) -> float:
        """计算Coinbase手续费"""
        notional = price * quantity
        fee_rate = self.latency_profile.maker_fee if is_maker else self.latency_profile.taker_fee
        
        # Coinbase Pro费率阶梯 (简化)
        if notional > 50000:
            fee_rate *= 0.9
        if notional > 100000:
            fee_rate *= 0.8
        
        return notional * fee_rate

class HuobiSimulator(ExchangeSimulator):
    """Huobi交易所模拟器"""
    
    def __init__(self):
        latency_profile = ExchangeLatencyProfile(
            name="Huobi",
            rest_api_latency=(60, 180),
            websocket_latency=(12, 35),
            order_processing_latency=(120, 350),
            network_jitter_std=6.0,
            packet_loss_rate=0.0008,
            maker_fee=0.002,  # 0.2%
            taker_fee=0.002,  # 0.2%
            min_order_size=0.0001,
            price_precision=2,
            quantity_precision=6
        )
        super().__init__(ExchangeType.HUOBI, latency_profile)
    
    def get_current_latency(self) -> float:
        """获取当前延迟 - Huobi特有模式"""
        base_latency = np.random.uniform(*self.latency_profile.order_processing_latency)
        
        # 亚洲时区影响
        current_hour = (time.localtime().tm_hour + 8) % 24  # 北京时间
        if 9 <= current_hour <= 17:  # 亚洲交易时间
            time_multiplier = 1.2
        else:
            time_multiplier = 0.9
        
        jitter = np.random.normal(0, self.latency_profile.network_jitter_std)
        load_factor = max(0.5, np.random.normal(
            self.latency_profile.load_factor_base * time_multiplier,
            self.latency_profile.load_factor_std
        ))
        
        self.load_factor_history.append(load_factor)
        
        return max(12.0, (base_latency + jitter) * load_factor)
    
    def calculate_fees(self, price: float, quantity: float, is_maker: bool) -> float:
        """计算Huobi手续费"""
        notional = price * quantity
        fee_rate = self.latency_profile.maker_fee if is_maker else self.latency_profile.taker_fee
        
        # Huobi HT抵扣优惠 (简化)
        fee_rate *= 0.9  # 假设使用HT抵扣
        
        return notional * fee_rate

class MultiExchangeEngine:
    """多交易所回测引擎"""
    
    def __init__(self, enabled_exchanges: List[ExchangeType] = None):
        if enabled_exchanges is None:
            enabled_exchanges = [ExchangeType.BINANCE, ExchangeType.COINBASE, ExchangeType.HUOBI]
        
        # 初始化交易所模拟器
        self.exchanges: Dict[str, ExchangeSimulator] = {}
        for exchange_type in enabled_exchanges:
            if exchange_type == ExchangeType.BINANCE:
                self.exchanges[exchange_type.value] = BinanceSimulator()
            elif exchange_type == ExchangeType.COINBASE:
                self.exchanges[exchange_type.value] = CoinbaseSimulator()
            elif exchange_type == ExchangeType.HUOBI:
                self.exchanges[exchange_type.value] = HuobiSimulator()
        
        # 跨交易所延迟矩阵 (微秒)
        self.cross_exchange_latency = {
            ("BINANCE", "COINBASE"): (800, 1200),
            ("BINANCE", "HUOBI"): (600, 1000),
            ("COINBASE", "HUOBI"): (1000, 1500),
        }
        
        # 套利监控
        self.arbitrage_opportunities: List[ArbitrageOpportunity] = []
        self.arbitrage_threshold = 0.001  # 0.1% 最小套利阈值
        
        # 统计信息
        self.total_arbitrage_opportunities = 0
        self.total_arbitrage_profit = 0.0
        self.cross_exchange_orders: Dict[str, List[CrossExchangeOrder]] = defaultdict(list)
        
        # 事件调度
        self.global_scheduler = MicrosecondEventScheduler()
        
    def add_initial_liquidity(self, exchange_name: str, base_price: float = 45000.0, 
                             levels: int = 100, quantity_per_level: float = 1.0):
        """为交易所添加初始流动性"""
        exchange = self.exchanges.get(exchange_name)
        if not exchange:
            return False
        
        # 添加买盘和卖盘
        for i in range(levels):
            # 买盘
            bid_price = base_price - (i + 1) * 0.1
            bid_order = Level3Order(
                order_id=f"{exchange_name}_initial_bid_{i}",
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                price=bid_price,
                original_quantity=quantity_per_level,
                remaining_quantity=quantity_per_level
            )
            exchange.orderbook.add_order(bid_order)
            
            # 卖盘
            ask_price = base_price + (i + 1) * 0.1
            ask_order = Level3Order(
                order_id=f"{exchange_name}_initial_ask_{i}",
                symbol="BTCUSDT",
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                price=ask_price,
                original_quantity=quantity_per_level,
                remaining_quantity=quantity_per_level
            )
            exchange.orderbook.add_order(ask_order)
        
        return True
    
    def submit_order_to_exchange(self, exchange_name: str, order: Level3Order) -> Optional[CrossExchangeOrder]:
        """向指定交易所提交订单"""
        exchange = self.exchanges.get(exchange_name)
        if not exchange:
            return None
        
        cross_order = exchange.submit_order(order)
        self.cross_exchange_orders[exchange_name].append(cross_order)
        
        return cross_order
    
    def submit_arbitrage_orders(self, opportunity: ArbitrageOpportunity, 
                               quantity: float) -> Tuple[Optional[CrossExchangeOrder], Optional[CrossExchangeOrder]]:
        """提交套利订单对"""
        if quantity > opportunity.max_quantity:
            quantity = opportunity.max_quantity
        
        # 创建买单 (低价交易所)
        buy_order = Level3Order(
            order_id=f"arb_buy_{uuid.uuid4().hex[:8]}",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,  # 市价单快速成交
            price=opportunity.buy_price,
            original_quantity=quantity,
            remaining_quantity=quantity
        )
        
        # 创建卖单 (高价交易所)
        sell_order = Level3Order(
            order_id=f"arb_sell_{uuid.uuid4().hex[:8]}",
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            price=opportunity.sell_price,
            original_quantity=quantity,
            remaining_quantity=quantity
        )
        
        # 计算跨交易所延迟
        cross_latency = self._get_cross_exchange_latency(
            opportunity.buy_exchange, 
            opportunity.sell_exchange
        )
        
        # 提交买单
        buy_cross_order = self.submit_order_to_exchange(opportunity.buy_exchange, buy_order)
        
        # 延迟提交卖单 (模拟跨交易所协调延迟)
        self.global_scheduler.schedule_delay_event(
            int(cross_latency),
            "submit_sell_order",
            {
                "exchange_name": opportunity.sell_exchange,
                "order": sell_order,
                "related_buy_order": buy_cross_order.order_id if buy_cross_order else None
            }
        )
        
        return buy_cross_order, None  # 卖单稍后提交
    
    def detect_arbitrage_opportunities(self, min_profit_threshold: float = None) -> List[ArbitrageOpportunity]:
        """检测套利机会"""
        if min_profit_threshold is None:
            min_profit_threshold = self.arbitrage_threshold
        
        opportunities = []
        exchange_names = list(self.exchanges.keys())
        current_time = HighPrecisionTimestamp.now()
        
        # 比较每对交易所
        for i in range(len(exchange_names)):
            for j in range(i + 1, len(exchange_names)):
                exchange_a = exchange_names[i]
                exchange_b = exchange_names[j]
                
                # 获取最优价格
                bid_a, ask_a = self.exchanges[exchange_a].get_best_prices()
                bid_b, ask_b = self.exchanges[exchange_b].get_best_prices()
                
                if not all([bid_a, ask_a, bid_b, ask_b]):
                    continue
                
                # 检查A买B卖的套利机会
                if bid_b > ask_a:
                    profit_per_unit = bid_b - ask_a
                    profit_percentage = profit_per_unit / ask_a
                    
                    if profit_percentage >= min_profit_threshold:
                        # 获取可交易量
                        depth_a = self.exchanges[exchange_a].get_market_depth(5)
                        depth_b = self.exchanges[exchange_b].get_market_depth(5)
                        
                        max_quantity = min(
                            sum(qty for _, qty, _ in depth_a.get('asks', [])[:3]),  # A交易所卖盘
                            sum(qty for _, qty, _ in depth_b.get('bids', [])[:3])   # B交易所买盘
                        )
                        
                        if max_quantity > 0:
                            opportunity = ArbitrageOpportunity(
                                buy_exchange=exchange_a,
                                sell_exchange=exchange_b,
                                buy_price=ask_a,
                                sell_price=bid_b,
                                max_quantity=max_quantity,
                                profit_per_unit=profit_per_unit,
                                profit_percentage=profit_percentage,
                                timestamp=current_time
                            )
                            opportunities.append(opportunity)
                
                # 检查B买A卖的套利机会
                if bid_a > ask_b:
                    profit_per_unit = bid_a - ask_b
                    profit_percentage = profit_per_unit / ask_b
                    
                    if profit_percentage >= min_profit_threshold:
                        depth_a = self.exchanges[exchange_a].get_market_depth(5)
                        depth_b = self.exchanges[exchange_b].get_market_depth(5)
                        
                        max_quantity = min(
                            sum(qty for _, qty, _ in depth_b.get('asks', [])[:3]),
                            sum(qty for _, qty, _ in depth_a.get('bids', [])[:3])
                        )
                        
                        if max_quantity > 0:
                            opportunity = ArbitrageOpportunity(
                                buy_exchange=exchange_b,
                                sell_exchange=exchange_a,
                                buy_price=ask_b,
                                sell_price=bid_a,
                                max_quantity=max_quantity,
                                profit_per_unit=profit_per_unit,
                                profit_percentage=profit_percentage,
                                timestamp=current_time
                            )
                            opportunities.append(opportunity)
        
        # 按盈利潜力排序
        opportunities.sort(key=lambda x: x.total_profit_potential, reverse=True)
        
        self.arbitrage_opportunities.extend(opportunities)
        self.total_arbitrage_opportunities += len(opportunities)
        
        return opportunities
    
    def _get_cross_exchange_latency(self, exchange_a: str, exchange_b: str) -> float:
        """获取跨交易所延迟"""
        key = (exchange_a, exchange_b)
        reverse_key = (exchange_b, exchange_a)
        
        if key in self.cross_exchange_latency:
            latency_range = self.cross_exchange_latency[key]
        elif reverse_key in self.cross_exchange_latency:
            latency_range = self.cross_exchange_latency[reverse_key]
        else:
            # 默认跨交易所延迟
            latency_range = (500, 1000)
        
        return np.random.uniform(*latency_range)
    
    def process_global_events(self, current_time: HighPrecisionTimestamp) -> Dict[str, List[Dict[str, Any]]]:
        """处理全局事件"""
        all_events = {}
        
        # 处理全局调度器事件
        global_events = self.global_scheduler.advance_time_to(current_time)
        
        for event_type, event_data in global_events:
            if event_type == "submit_sell_order":
                exchange_name = event_data["exchange_name"]
                order = event_data["order"]
                
                sell_cross_order = self.submit_order_to_exchange(exchange_name, order)
                if sell_cross_order:
                    all_events.setdefault("arbitrage_orders", []).append({
                        "type": "sell_order_submitted",
                        "order_id": sell_cross_order.order_id,
                        "exchange": exchange_name,
                        "related_buy_order": event_data.get("related_buy_order")
                    })
        
        # 处理各交易所事件
        for exchange_name, exchange in self.exchanges.items():
            exchange_events = exchange.process_pending_events(current_time)
            if exchange_events:
                all_events[exchange_name] = exchange_events
        
        return all_events
    
    def get_consolidated_orderbook(self, depth: int = 10) -> Dict[str, Any]:
        """获取合并订单簿"""
        all_bids = []
        all_asks = []
        
        for exchange_name, exchange in self.exchanges.items():
            depth_data = exchange.get_market_depth(depth)
            
            # 添加交易所标识
            for price, qty, count in depth_data.get('bids', []):
                all_bids.append((price, qty, count, exchange_name))
            
            for price, qty, count in depth_data.get('asks', []):
                all_asks.append((price, qty, count, exchange_name))
        
        # 排序合并
        all_bids.sort(key=lambda x: x[0], reverse=True)  # 按价格降序
        all_asks.sort(key=lambda x: x[0])                # 按价格升序
        
        return {
            'bids': all_bids[:depth],
            'asks': all_asks[:depth],
            'timestamp': HighPrecisionTimestamp.now().to_microseconds()
        }
    
    def get_multi_exchange_statistics(self) -> Dict[str, Any]:
        """获取多交易所统计信息"""
        stats = {
            'total_exchanges': len(self.exchanges),
            'total_arbitrage_opportunities': self.total_arbitrage_opportunities,
            'total_arbitrage_profit': self.total_arbitrage_profit,
            'exchanges': {}
        }
        
        total_orders = 0
        total_volume = 0.0
        all_latencies = []
        
        for exchange_name, exchange in self.exchanges.items():
            exchange_stats = exchange.get_statistics()
            stats['exchanges'][exchange_name] = exchange_stats
            
            total_orders += exchange_stats['total_orders_submitted']
            total_volume += exchange_stats['total_volume_traded']
            
            if exchange.latency_samples:
                all_latencies.extend(exchange.latency_samples)
        
        # 全局统计
        stats['global'] = {
            'total_orders_submitted': total_orders,
            'total_volume_traded': total_volume,
            'avg_latency_us': np.mean(all_latencies) if all_latencies else 0,
            'p95_latency_us': np.percentile(all_latencies, 95) if all_latencies else 0,
            'p99_latency_us': np.percentile(all_latencies, 99) if all_latencies else 0
        }
        
        return stats

def benchmark_multi_exchange_engine():
    """多交易所引擎性能基准测试"""
    print("🌐 多交易所引擎性能基准测试")
    print("=" * 50)
    
    # 创建多交易所引擎
    engine = MultiExchangeEngine([
        ExchangeType.BINANCE,
        ExchangeType.COINBASE,
        ExchangeType.HUOBI
    ])
    
    # 初始化流动性
    print("\n📊 初始化流动性...")
    for exchange_name in engine.exchanges.keys():
        engine.add_initial_liquidity(exchange_name, base_price=45000.0, levels=50)
    
    print(f"初始化完成，支持交易所: {list(engine.exchanges.keys())}")
    
    # 测试1: 套利机会检测
    print("\n📊 测试1: 套利机会检测")
    start_time = time.perf_counter()
    
    # 人为制造价差进行测试
    binance_buy = Level3Order(
        order_id="test_buy",
        symbol="BTCUSDT", 
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=45010.0,  # 高价买入
        original_quantity=10.0,
        remaining_quantity=10.0
    )
    engine.exchanges["BINANCE"].orderbook.add_order(binance_buy)
    
    # 检测套利机会
    opportunities = engine.detect_arbitrage_opportunities(min_profit_threshold=0.0001)
    detection_time = time.perf_counter() - start_time
    
    print(f"检测到套利机会: {len(opportunities)} 个")
    print(f"检测时间: {detection_time * 1000:.2f} 毫秒")
    
    if opportunities:
        best_opp = opportunities[0]
        print(f"最佳机会: {best_opp.buy_exchange} -> {best_opp.sell_exchange}")
        print(f"利润率: {best_opp.profit_percentage * 100:.4f}%")
        print(f"利润潜力: ${best_opp.total_profit_potential:.2f}")
    
    # 测试2: 跨交易所订单提交
    print("\n📊 测试2: 跨交易所订单提交")
    start_time = time.perf_counter()
    
    submitted_orders = []
    for i in range(100):
        for exchange_name in engine.exchanges.keys():
            order = Level3Order(
                order_id=f"test_order_{exchange_name}_{i}",
                symbol="BTCUSDT",
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                order_type=OrderType.LIMIT,
                price=45000.0 + (i % 10 - 5) * 0.1,
                original_quantity=0.1 + i * 0.01,
                remaining_quantity=0.1 + i * 0.01
            )
            
            cross_order = engine.submit_order_to_exchange(exchange_name, order)
            if cross_order:
                submitted_orders.append(cross_order)
    
    submission_time = time.perf_counter() - start_time
    
    print(f"提交订单数: {len(submitted_orders)}")
    print(f"提交时间: {submission_time:.4f}秒")
    print(f"提交速度: {len(submitted_orders) / submission_time:,.0f} 订单/秒")
    
    # 测试3: 事件处理性能
    print("\n📊 测试3: 事件处理性能")
    start_time = time.perf_counter()
    
    # 推进时间以处理待处理事件
    current_time = HighPrecisionTimestamp.now()
    future_time = current_time.add_milliseconds(1000)  # 1秒后
    
    all_events = engine.process_global_events(future_time)
    processing_time = time.perf_counter() - start_time
    
    total_events = sum(len(events) for events in all_events.values())
    
    print(f"处理事件数: {total_events}")
    print(f"处理时间: {processing_time * 1000:.2f} 毫秒")
    if total_events > 0:
        print(f"处理速度: {total_events / processing_time:,.0f} 事件/秒")
    
    # 测试4: 合并订单簿生成
    print("\n📊 测试4: 合并订单簿生成")
    start_time = time.perf_counter()
    
    consolidated_books = []
    for _ in range(1000):
        book = engine.get_consolidated_orderbook(depth=20)
        consolidated_books.append(book)
    
    consolidation_time = time.perf_counter() - start_time
    
    print(f"合并订单簿生成: {len(consolidated_books):,} 次")
    print(f"生成时间: {consolidation_time:.4f}秒")
    print(f"生成速度: {len(consolidated_books) / consolidation_time:,.0f} 次/秒")
    
    # 最终统计
    print("\n📈 最终统计")
    stats = engine.get_multi_exchange_statistics()
    
    print(f"总交易所数: {stats['total_exchanges']}")
    print(f"总套利机会: {stats['total_arbitrage_opportunities']}")
    print(f"全局订单数: {stats['global']['total_orders_submitted']}")
    print(f"全局交易量: {stats['global']['total_volume_traded']:.2f}")
    print(f"平均延迟: {stats['global']['avg_latency_us']:.2f} 微秒")
    print(f"P95延迟: {stats['global']['p95_latency_us']:.2f} 微秒")
    
    for exchange_name, exchange_stats in stats['exchanges'].items():
        print(f"\n{exchange_name}:")
        print(f"  订单数: {exchange_stats['total_orders_submitted']}")
        print(f"  成交量: {exchange_stats['total_volume_traded']:.2f}")
        print(f"  平均延迟: {exchange_stats['avg_latency_us']:.2f} 微秒")
    
    print("\n✅ 多交易所引擎测试完成!")
    
    return {
        'arbitrage_detection_time_ms': detection_time * 1000,
        'order_submission_speed': len(submitted_orders) / submission_time,
        'event_processing_speed': total_events / processing_time if total_events > 0 else 0,
        'consolidation_speed': len(consolidated_books) / consolidation_time,
        'total_arbitrage_opportunities': len(opportunities),
        'global_stats': stats['global']
    }

if __name__ == "__main__":
    # 运行多交易所引擎基准测试
    results = benchmark_multi_exchange_engine()
    
    print(f"\n🎯 多交易所引擎性能总结:")
    print(f"- 套利检测速度: {results['arbitrage_detection_time_ms']:.2f} 毫秒")
    print(f"- 订单提交速度: {results['order_submission_speed']:,.0f} 订单/秒")
    print(f"- 事件处理速度: {results['event_processing_speed']:,.0f} 事件/秒")
    print(f"- 合并速度: {results['consolidation_speed']:,.0f} 次/秒")
    print(f"- 套利机会数: {results['total_arbitrage_opportunities']}")
    print(f"- 全局平均延迟: {results['global_stats']['avg_latency_us']:.2f} 微秒")