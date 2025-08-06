"""
统一交易接口
提供回测和实盘交易的一致性API，确保策略代码无需修改即可在两种环境间切换
"""

import asyncio
import numpy as np
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any, Union, Protocol, runtime_checkable
from dataclasses import dataclass, field
from enum import Enum
import uuid
from datetime import datetime
import json

from .microsecond_precision import HighPrecisionTimestamp
from .level3_orderbook_advanced import Level3Order, OrderSide, OrderType, OrderStatus, TradeExecution
from .high_fidelity_backtest_engine import HighFidelityBacktestEngine
from .multi_exchange_engine import MultiExchangeEngine, ExchangeType

# 统一数据结构
@dataclass
class UnifiedMarketData:
    """统一市场数据结构"""
    symbol: str
    timestamp: HighPrecisionTimestamp
    price: float
    volume: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    spread: Optional[float] = None
    source: str = "unknown"
    
    @property
    def mid_price(self) -> Optional[float]:
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2.0
        return self.price

@dataclass
class UnifiedOrderRequest:
    """统一订单请求"""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    client_order_id: Optional[str] = None
    time_in_force: str = "GTC"
    
    # 高级参数
    stop_price: Optional[float] = None
    iceberg_quantity: Optional[float] = None
    strategy_id: Optional[str] = None
    
    def __post_init__(self):
        if not self.client_order_id:
            self.client_order_id = f"unified_{uuid.uuid4().hex[:12]}"

@dataclass
class UnifiedOrderResponse:
    """统一订单响应"""
    order_id: str
    client_order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    status: OrderStatus
    original_quantity: float
    filled_quantity: float
    remaining_quantity: float
    avg_fill_price: float = 0.0
    timestamp: HighPrecisionTimestamp = field(default_factory=HighPrecisionTimestamp.now)
    
    # 执行统计
    latency_us: float = 0.0
    total_fees: float = 0.0
    fills: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def fill_percentage(self) -> float:
        return self.filled_quantity / self.original_quantity if self.original_quantity > 0 else 0.0
    
    @property
    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILLED
    
    @property
    def is_active(self) -> bool:
        return self.status in [OrderStatus.ACTIVE, OrderStatus.PARTIALLY_FILLED]

@dataclass
class UnifiedPortfolioInfo:
    """统一投资组合信息"""
    timestamp: HighPrecisionTimestamp
    total_value: float
    cash_balance: float
    positions: Dict[str, float]  # symbol -> quantity
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    @property
    def total_pnl(self) -> float:
        return self.unrealized_pnl + self.realized_pnl

@dataclass 
class UnifiedPerformanceMetrics:
    """统一性能指标"""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    avg_trade_return: float
    volatility: float
    
    # 交易相关指标
    avg_latency_us: float = 0.0
    avg_slippage_bps: float = 0.0
    total_fees: float = 0.0
    
    # 风险指标
    var_95: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0

@runtime_checkable
class TradingInterface(Protocol):
    """统一交易接口协议"""
    
    async def submit_order(self, order_request: UnifiedOrderRequest) -> UnifiedOrderResponse:
        """提交订单"""
        ...
    
    async def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        ...
    
    async def get_order_status(self, order_id: str) -> Optional[UnifiedOrderResponse]:
        """获取订单状态"""
        ...
    
    async def get_market_data(self, symbol: str) -> Optional[UnifiedMarketData]:
        """获取市场数据"""
        ...
    
    async def get_portfolio_info(self) -> UnifiedPortfolioInfo:
        """获取投资组合信息"""
        ...
    
    async def get_performance_metrics(self) -> UnifiedPerformanceMetrics:
        """获取性能指标"""
        ...
    
    def is_market_open(self) -> bool:
        """检查市场是否开放"""
        ...

class BacktestTradingInterface:
    """回测交易接口实现"""
    
    def __init__(self, backtest_engine: HighFidelityBacktestEngine):
        self.engine = backtest_engine
        self.order_history: Dict[str, UnifiedOrderResponse] = {}
        self.active_orders: Dict[str, UnifiedOrderResponse] = {}
        
        # 性能跟踪
        self.trade_returns: List[float] = []
        self.equity_history: List[Tuple[HighPrecisionTimestamp, float]] = []
        self.latency_history: List[float] = []
        self.slippage_history: List[float] = []
    
    async def submit_order(self, order_request: UnifiedOrderRequest) -> UnifiedOrderResponse:
        """提交订单到回测引擎"""
        start_time = time.perf_counter_ns()
        
        # 转换为Level3Order
        level3_order = Level3Order(
            order_id=order_request.client_order_id,
            symbol=order_request.symbol,
            side=order_request.side,
            order_type=order_request.order_type,
            price=order_request.price or 0.0,
            original_quantity=order_request.quantity,
            remaining_quantity=order_request.quantity,
            time_in_force=order_request.time_in_force,
            stop_price=order_request.stop_price or 0.0,
            iceberg_quantity=order_request.iceberg_quantity or 0.0
        )
        
        # 提交到回测引擎
        success = self.engine.submit_order(level3_order)
        
        # 计算执行延迟
        execution_time_ns = time.perf_counter_ns() - start_time
        latency_us = execution_time_ns / 1000.0
        self.latency_history.append(latency_us)
        
        # 创建统一响应
        response = UnifiedOrderResponse(
            order_id=level3_order.order_id,
            client_order_id=order_request.client_order_id,
            symbol=order_request.symbol,
            side=order_request.side,
            order_type=order_request.order_type,
            status=OrderStatus.ACTIVE if success else OrderStatus.REJECTED,
            original_quantity=order_request.quantity,
            filled_quantity=0.0,
            remaining_quantity=order_request.quantity,
            latency_us=latency_us
        )
        
        if success:
            self.active_orders[response.order_id] = response
        
        self.order_history[response.order_id] = response
        return response
    
    async def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        if order_id not in self.active_orders:
            return False
        
        success = self.engine.cancel_order(order_id)
        
        if success and order_id in self.active_orders:
            self.active_orders[order_id].status = OrderStatus.CANCELLED
            del self.active_orders[order_id]
        
        return success
    
    async def get_order_status(self, order_id: str) -> Optional[UnifiedOrderResponse]:
        """获取订单状态"""
        return self.order_history.get(order_id)
    
    async def get_market_data(self, symbol: str) -> Optional[UnifiedMarketData]:
        """获取市场数据"""
        # 从回测引擎获取当前市场数据
        current_data = self.engine.get_current_market_data(symbol)
        
        if current_data:
            return UnifiedMarketData(
                symbol=symbol,
                timestamp=HighPrecisionTimestamp.now(),
                price=current_data.get('price', 0.0),
                volume=current_data.get('volume', 0.0),
                bid=current_data.get('bid'),
                ask=current_data.get('ask'),
                source="backtest"
            )
        
        return None
    
    async def get_portfolio_info(self) -> UnifiedPortfolioInfo:
        """获取投资组合信息"""
        portfolio_data = self.engine.get_portfolio_summary()
        
        return UnifiedPortfolioInfo(
            timestamp=HighPrecisionTimestamp.now(),
            total_value=portfolio_data.get('total_value', 0.0),
            cash_balance=portfolio_data.get('cash_balance', 0.0),
            positions=portfolio_data.get('positions', {}),
            unrealized_pnl=portfolio_data.get('unrealized_pnl', 0.0),
            realized_pnl=portfolio_data.get('realized_pnl', 0.0)
        )
    
    async def get_performance_metrics(self) -> UnifiedPerformanceMetrics:
        """获取性能指标"""
        # 从回测引擎获取性能数据
        performance = self.engine.get_performance_stats()
        
        # 计算额外指标
        avg_latency = np.mean(self.latency_history) if self.latency_history else 0.0
        avg_slippage = np.mean(self.slippage_history) if self.slippage_history else 0.0
        
        return UnifiedPerformanceMetrics(
            total_return=performance.get('total_return', 0.0),
            sharpe_ratio=performance.get('sharpe_ratio', 0.0),
            max_drawdown=performance.get('max_drawdown', 0.0),
            win_rate=performance.get('win_rate', 0.0),
            total_trades=performance.get('total_trades', 0),
            avg_trade_return=performance.get('avg_trade_return', 0.0),
            volatility=performance.get('volatility', 0.0),
            avg_latency_us=avg_latency,
            avg_slippage_bps=avg_slippage * 10000,
            total_fees=performance.get('total_fees', 0.0)
        )
    
    def is_market_open(self) -> bool:
        """检查市场是否开放 - 回测中始终开放"""
        return True

class LiveTradingInterface:
    """实盘交易接口实现"""
    
    def __init__(self, exchange_client: Any, risk_manager: Any = None):
        """
        初始化实盘交易接口
        
        Args:
            exchange_client: 交易所客户端 (如CCXT)
            risk_manager: 风险管理器
        """
        self.exchange_client = exchange_client
        self.risk_manager = risk_manager
        
        # 订单管理
        self.order_history: Dict[str, UnifiedOrderResponse] = {}
        self.active_orders: Dict[str, UnifiedOrderResponse] = {}
        
        # 性能跟踪
        self.trade_returns: List[float] = []
        self.latency_history: List[float] = []
        self.slippage_history: List[float] = []
        
        # 实时数据缓存
        self.market_data_cache: Dict[str, UnifiedMarketData] = {}
        self.last_portfolio_update = HighPrecisionTimestamp.now()
        
    async def submit_order(self, order_request: UnifiedOrderRequest) -> UnifiedOrderResponse:
        """提交订单到实盘交易所"""
        start_time = time.perf_counter_ns()
        
        # 风险检查
        if self.risk_manager:
            risk_check = await self._perform_risk_check(order_request)
            if not risk_check["approved"]:
                return UnifiedOrderResponse(
                    order_id="",
                    client_order_id=order_request.client_order_id,
                    symbol=order_request.symbol,
                    side=order_request.side,
                    order_type=order_request.order_type,
                    status=OrderStatus.REJECTED,
                    original_quantity=order_request.quantity,
                    filled_quantity=0.0,
                    remaining_quantity=order_request.quantity
                )
        
        try:
            # 构建交易所订单参数
            order_params = {
                'symbol': order_request.symbol,
                'type': order_request.order_type.value.lower(),
                'side': order_request.side.value.lower(),
                'amount': order_request.quantity,
                'params': {
                    'clientOrderId': order_request.client_order_id,
                    'timeInForce': order_request.time_in_force
                }
            }
            
            if order_request.price is not None:
                order_params['price'] = order_request.price
            
            if order_request.stop_price is not None:
                order_params['params']['stopPrice'] = order_request.stop_price
            
            # 提交订单到交易所
            exchange_response = await self.exchange_client.create_order(**order_params)
            
            # 计算执行延迟
            execution_time_ns = time.perf_counter_ns() - start_time
            latency_us = execution_time_ns / 1000.0
            self.latency_history.append(latency_us)
            
            # 创建统一响应
            response = UnifiedOrderResponse(
                order_id=exchange_response['id'],
                client_order_id=order_request.client_order_id,
                symbol=order_request.symbol,
                side=order_request.side,
                order_type=order_request.order_type,
                status=self._parse_order_status(exchange_response['status']),
                original_quantity=float(exchange_response['amount']),
                filled_quantity=float(exchange_response.get('filled', 0)),
                remaining_quantity=float(exchange_response.get('remaining', exchange_response['amount'])),
                avg_fill_price=float(exchange_response.get('average', 0) or 0),
                latency_us=latency_us,
                total_fees=float(exchange_response.get('fee', {}).get('cost', 0) or 0)
            )
            
            if response.is_active:
                self.active_orders[response.order_id] = response
            
            self.order_history[response.order_id] = response
            return response
            
        except Exception as e:
            # 处理错误
            print(f"订单提交失败: {e}")
            return UnifiedOrderResponse(
                order_id="",
                client_order_id=order_request.client_order_id,
                symbol=order_request.symbol,
                side=order_request.side,
                order_type=order_request.order_type,
                status=OrderStatus.REJECTED,
                original_quantity=order_request.quantity,
                filled_quantity=0.0,
                remaining_quantity=order_request.quantity,
                latency_us=(time.perf_counter_ns() - start_time) / 1000.0
            )
    
    async def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        if order_id not in self.active_orders:
            return False
        
        try:
            order_info = self.active_orders[order_id]
            await self.exchange_client.cancel_order(order_id, order_info.symbol)
            
            # 更新订单状态
            order_info.status = OrderStatus.CANCELLED
            del self.active_orders[order_id]
            
            return True
            
        except Exception as e:
            print(f"取消订单失败: {e}")
            return False
    
    async def get_order_status(self, order_id: str) -> Optional[UnifiedOrderResponse]:
        """获取订单状态"""
        if order_id in self.order_history:
            # 如果是活跃订单，更新状态
            if order_id in self.active_orders:
                try:
                    order_info = self.active_orders[order_id]
                    exchange_order = await self.exchange_client.fetch_order(order_id, order_info.symbol)
                    
                    # 更新状态
                    order_info.status = self._parse_order_status(exchange_order['status'])
                    order_info.filled_quantity = float(exchange_order.get('filled', 0))
                    order_info.remaining_quantity = float(exchange_order.get('remaining', order_info.original_quantity))
                    order_info.avg_fill_price = float(exchange_order.get('average', 0) or 0)
                    
                    if not order_info.is_active:
                        del self.active_orders[order_id]
                    
                except Exception as e:
                    print(f"获取订单状态失败: {e}")
            
            return self.order_history[order_id]
        
        return None
    
    async def get_market_data(self, symbol: str) -> Optional[UnifiedMarketData]:
        """获取市场数据"""
        try:
            ticker = await self.exchange_client.fetch_ticker(symbol)
            
            market_data = UnifiedMarketData(
                symbol=symbol,
                timestamp=HighPrecisionTimestamp.now(),
                price=float(ticker['last']),
                volume=float(ticker['baseVolume']),
                bid=float(ticker['bid']) if ticker['bid'] else None,
                ask=float(ticker['ask']) if ticker['ask'] else None,
                source="live"
            )
            
            # 计算价差
            if market_data.bid and market_data.ask:
                market_data.spread = market_data.ask - market_data.bid
            
            # 缓存数据
            self.market_data_cache[symbol] = market_data
            
            return market_data
            
        except Exception as e:
            print(f"获取市场数据失败: {e}")
            # 返回缓存数据
            return self.market_data_cache.get(symbol)
    
    async def get_portfolio_info(self) -> UnifiedPortfolioInfo:
        """获取投资组合信息"""
        try:
            balance = await self.exchange_client.fetch_balance()
            
            # 提取持仓信息
            positions = {}
            total_value = 0.0
            
            for currency, info in balance.items():
                if currency in ['free', 'used', 'total']:
                    continue
                
                total_amount = float(info.get('total', 0))
                if total_amount > 0:
                    positions[currency] = total_amount
                    # 这里简化处理，实际应该获取当前价格计算价值
                    if currency == 'USDT':
                        total_value += total_amount
            
            return UnifiedPortfolioInfo(
                timestamp=HighPrecisionTimestamp.now(),
                total_value=total_value,
                cash_balance=float(balance.get('USDT', {}).get('free', 0)),
                positions=positions
            )
            
        except Exception as e:
            print(f"获取投资组合信息失败: {e}")
            return UnifiedPortfolioInfo(
                timestamp=HighPrecisionTimestamp.now(),
                total_value=0.0,
                cash_balance=0.0,
                positions={}
            )
    
    async def get_performance_metrics(self) -> UnifiedPerformanceMetrics:
        """获取性能指标"""
        # 实盘交易的性能指标需要基于历史交易数据计算
        # 这里提供简化实现
        
        total_trades = len([o for o in self.order_history.values() if o.is_filled])
        avg_latency = np.mean(self.latency_history) if self.latency_history else 0.0
        total_fees = sum(o.total_fees for o in self.order_history.values())
        
        return UnifiedPerformanceMetrics(
            total_return=0.0,  # 需要根据初始资金和当前价值计算
            sharpe_ratio=0.0,  # 需要历史收益数据计算
            max_drawdown=0.0,  # 需要权益曲线计算
            win_rate=0.0,      # 需要盈亏交易统计
            total_trades=total_trades,
            avg_trade_return=0.0,
            volatility=0.0,
            avg_latency_us=avg_latency,
            total_fees=total_fees
        )
    
    def is_market_open(self) -> bool:
        """检查市场是否开放 - 加密货币市场24/7开放"""
        return True
    
    def _parse_order_status(self, exchange_status: str) -> OrderStatus:
        """解析交易所订单状态"""
        status_mapping = {
            'open': OrderStatus.ACTIVE,
            'closed': OrderStatus.FILLED,
            'canceled': OrderStatus.CANCELLED,
            'cancelled': OrderStatus.CANCELLED,
            'rejected': OrderStatus.REJECTED,
            'pending': OrderStatus.PENDING,
            'partial': OrderStatus.PARTIALLY_FILLED,
            'partially_filled': OrderStatus.PARTIALLY_FILLED
        }
        
        return status_mapping.get(exchange_status.lower(), OrderStatus.PENDING)
    
    async def _perform_risk_check(self, order_request: UnifiedOrderRequest) -> Dict[str, Any]:
        """执行风险检查"""
        if not self.risk_manager:
            return {"approved": True}
        
        # 这里应该调用风险管理器的检查方法
        # 简化实现
        return {"approved": True, "reason": "Risk check passed"}

class UnifiedStrategy:
    """统一策略基类 - 可以在回测和实盘环境中运行"""
    
    def __init__(self, trading_interface: TradingInterface, config: Dict[str, Any] = None):
        self.trading = trading_interface
        self.config = config or {}
        
        # 策略状态
        self.positions: Dict[str, float] = {}
        self.active_orders: Dict[str, UnifiedOrderResponse] = {}
        self.performance_history: List[float] = []
        
        # 策略参数
        self.risk_limit = self.config.get('risk_limit', 0.02)  # 2%风险限制
        self.position_size = self.config.get('position_size', 1000.0)  # USDT
        
    async def initialize(self):
        """策略初始化"""
        pass
    
    async def on_market_data(self, market_data: UnifiedMarketData):
        """处理市场数据"""
        pass
    
    async def on_order_update(self, order: UnifiedOrderResponse):
        """处理订单更新"""
        if order.order_id in self.active_orders:
            if not order.is_active:
                del self.active_orders[order.order_id]
    
    async def submit_order(self, order_request: UnifiedOrderRequest) -> Optional[UnifiedOrderResponse]:
        """提交订单（包含风险检查）"""
        # 检查风险限制
        if not await self._check_risk_limits(order_request):
            print(f"订单被风险控制拒绝: {order_request.client_order_id}")
            return None
        
        # 提交订单
        response = await self.trading.submit_order(order_request)
        
        if response.is_active:
            self.active_orders[response.order_id] = response
        
        return response
    
    async def cancel_all_orders(self) -> int:
        """取消所有活跃订单"""
        cancelled_count = 0
        
        for order_id in list(self.active_orders.keys()):
            if await self.trading.cancel_order(order_id):
                cancelled_count += 1
                del self.active_orders[order_id]
        
        return cancelled_count
    
    async def get_current_portfolio(self) -> UnifiedPortfolioInfo:
        """获取当前投资组合"""
        return await self.trading.get_portfolio_info()
    
    async def get_strategy_performance(self) -> UnifiedPerformanceMetrics:
        """获取策略性能"""
        return await self.trading.get_performance_metrics()
    
    async def _check_risk_limits(self, order_request: UnifiedOrderRequest) -> bool:
        """检查风险限制"""
        portfolio = await self.trading.get_portfolio_info()
        
        # 检查仓位限制
        order_value = order_request.quantity * (order_request.price or 0)
        if order_value > self.position_size:
            return False
        
        # 检查总风险敞口
        total_exposure = abs(order_value)
        for position in portfolio.positions.values():
            total_exposure += abs(position)
        
        if total_exposure > portfolio.total_value * self.risk_limit:
            return False
        
        return True

class SimpleMovingAverageStrategy(UnifiedStrategy):
    """简单移动平均策略示例"""
    
    def __init__(self, trading_interface: TradingInterface, config: Dict[str, Any] = None):
        super().__init__(trading_interface, config)
        
        # 策略参数
        self.short_window = self.config.get('short_window', 10)
        self.long_window = self.config.get('long_window', 30)
        self.symbol = self.config.get('symbol', 'BTCUSDT')
        
        # 价格历史
        self.price_history: List[float] = []
        self.signal_history: List[str] = []
        
    async def on_market_data(self, market_data: UnifiedMarketData):
        """处理市场数据更新"""
        if market_data.symbol != self.symbol:
            return
        
        # 更新价格历史
        self.price_history.append(market_data.price)
        
        # 保持历史长度
        if len(self.price_history) > self.long_window * 2:
            self.price_history = self.price_history[-self.long_window * 2:]
        
        # 计算移动平均
        if len(self.price_history) >= self.long_window:
            short_ma = np.mean(self.price_history[-self.short_window:])
            long_ma = np.mean(self.price_history[-self.long_window:])
            
            # 生成信号
            current_signal = "HOLD"
            if short_ma > long_ma * 1.001:  # 0.1%阈值
                current_signal = "BUY"
            elif short_ma < long_ma * 0.999:
                current_signal = "SELL"
            
            # 检查信号变化
            if len(self.signal_history) == 0 or self.signal_history[-1] != current_signal:
                await self._execute_signal(current_signal, market_data)
            
            self.signal_history.append(current_signal)
            
            # 保持信号历史长度
            if len(self.signal_history) > 100:
                self.signal_history = self.signal_history[-100:]
    
    async def _execute_signal(self, signal: str, market_data: UnifiedMarketData):
        """执行交易信号"""
        if signal == "HOLD":
            return
        
        # 取消现有订单
        await self.cancel_all_orders()
        
        # 计算订单数量
        portfolio = await self.get_current_portfolio()
        current_position = portfolio.positions.get(self.symbol.replace('USDT', ''), 0.0)
        
        if signal == "BUY" and current_position <= 0:
            # 买入信号且当前无多头仓位
            quantity = self.position_size / market_data.price
            
            order_request = UnifiedOrderRequest(
                symbol=self.symbol,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=quantity,
                strategy_id="sma_strategy"
            )
            
            response = await self.submit_order(order_request)
            if response:
                print(f"🟢 BUY Signal: {response.order_id}, Quantity: {quantity:.6f}")
        
        elif signal == "SELL" and current_position > 0:
            # 卖出信号且当前有多头仓位
            quantity = abs(current_position)
            
            order_request = UnifiedOrderRequest(
                symbol=self.symbol,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=quantity,
                strategy_id="sma_strategy"
            )
            
            response = await self.submit_order(order_request)
            if response:
                print(f"🔴 SELL Signal: {response.order_id}, Quantity: {quantity:.6f}")

async def demo_unified_interface():
    """演示统一接口的使用"""
    print("🔗 统一交易接口演示")
    print("=" * 50)
    
    # 创建回测引擎
    backtest_engine = HighFidelityBacktestEngine(initial_balance=100000.0)
    
    # 创建回测交易接口
    backtest_interface = BacktestTradingInterface(backtest_engine)
    
    print("📊 回测环境测试")
    print("-" * 30)
    
    # 创建策略配置
    strategy_config = {
        'short_window': 5,
        'long_window': 15,
        'symbol': 'BTCUSDT',
        'position_size': 5000.0
    }
    
    # 创建策略实例
    strategy = SimpleMovingAverageStrategy(backtest_interface, strategy_config)
    await strategy.initialize()
    
    # 模拟市场数据
    base_price = 45000.0
    prices = []
    
    # 生成趋势数据
    for i in range(50):
        # 上升趋势 + 噪音
        trend = i * 0.5
        noise = np.random.normal(0, 10)
        price = base_price + trend + noise
        prices.append(price)
        
        # 创建市场数据
        market_data = UnifiedMarketData(
            symbol='BTCUSDT',
            timestamp=HighPrecisionTimestamp.now(),
            price=price,
            volume=100.0 + np.random.uniform(0, 50),
            bid=price - 0.5,
            ask=price + 0.5,
            source="backtest"
        )
        
        # 发送到策略
        await strategy.on_market_data(market_data)
        
        # 每10个数据点输出一次状态
        if i % 10 == 0:
            portfolio = await strategy.get_current_portfolio()
            print(f"Step {i}: Price={price:.2f}, Portfolio Value=${portfolio.total_value:.2f}")
    
    print(f"\n📈 回测结果")
    
    # 获取最终性能
    portfolio = await strategy.get_current_portfolio()
    performance = await strategy.get_strategy_performance()
    
    print(f"初始资金: $100,000.00")
    print(f"最终价值: ${portfolio.total_value:.2f}")
    print(f"总收益: ${portfolio.total_pnl:.2f}")
    print(f"收益率: {(portfolio.total_value / 100000 - 1) * 100:.2f}%")
    print(f"总交易数: {performance.total_trades}")
    print(f"平均延迟: {performance.avg_latency_us:.2f} 微秒")
    
    # 测试订单提交性能
    print(f"\n📊 性能测试")
    print("-" * 30)
    
    start_time = time.perf_counter()
    
    # 提交100个测试订单
    test_orders = []
    for i in range(100):
        order_request = UnifiedOrderRequest(
            symbol='BTCUSDT',
            side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=0.001,
            price=45000.0 + i
        )
        
        response = await backtest_interface.submit_order(order_request)
        test_orders.append(response)
    
    submission_time = time.perf_counter() - start_time
    
    print(f"提交订单数: {len(test_orders)}")
    print(f"提交时间: {submission_time:.4f}秒")
    print(f"提交速度: {len(test_orders) / submission_time:,.0f} 订单/秒")
    
    # 统计订单状态
    active_count = sum(1 for o in test_orders if o.is_active)
    rejected_count = sum(1 for o in test_orders if o.status == OrderStatus.REJECTED)
    
    print(f"活跃订单: {active_count}")
    print(f"被拒订单: {rejected_count}")
    
    print("\n✅ 统一接口演示完成!")
    
    return {
        'total_orders': len(test_orders),
        'submission_speed': len(test_orders) / submission_time,
        'strategy_performance': {
            'total_return': (portfolio.total_value / 100000 - 1) * 100,
            'total_trades': performance.total_trades,
            'avg_latency_us': performance.avg_latency_us
        }
    }

if __name__ == "__main__":
    # 运行统一接口演示
    import asyncio
    
    async def main():
        results = await demo_unified_interface()
        
        print(f"\n🎯 统一接口性能总结:")
        print(f"- 订单提交速度: {results['submission_speed']:,.0f} 订单/秒")
        print(f"- 策略收益率: {results['strategy_performance']['total_return']:.2f}%")
        print(f"- 策略交易数: {results['strategy_performance']['total_trades']}")
        print(f"- 平均延迟: {results['strategy_performance']['avg_latency_us']:.2f} 微秒")
    
    asyncio.run(main())