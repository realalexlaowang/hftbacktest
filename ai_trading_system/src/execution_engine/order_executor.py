"""
智能订单执行引擎 - 负责订单路由、执行优化和滑点控制
"""
import asyncio
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import ccxt
from loguru import logger
import redis
import json
from kafka import KafkaProducer

from config.config import get_config
from ai_engine.trading_ai import TradingSignal
from data_pipeline.market_data_collector import MarketData, OrderBookData
from risk_management.risk_manager import Position


class OrderType(Enum):
    """订单类型"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TWAP = "twap"  # 时间加权平均价格
    VWAP = "vwap"  # 成交量加权平均价格


class OrderStatus(Enum):
    """订单状态"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL_FILLED = "partial_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderSide(Enum):
    """订单方向"""
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """订单结构"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"  # GTC, IOC, FOK
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    fee: float = 0.0
    timestamp: int = None
    exchange: str = "binance"
    strategy_id: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = int(time.time() * 1000)


@dataclass
class Fill:
    """成交记录"""
    fill_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    fee: float
    timestamp: int
    exchange: str


@dataclass
class ExecutionReport:
    """执行报告"""
    order_id: str
    execution_time_ms: float  # 执行耗时
    slippage: float  # 滑点
    market_impact: float  # 市场冲击
    execution_quality: float  # 执行质量评分 0-1
    vwap_comparison: float  # 与VWAP的比较
    success: bool
    error_message: Optional[str] = None


class SmartOrderRouter:
    """智能订单路由器"""
    
    def __init__(self):
        self.config = get_config()
        self.exchanges = {}
        
    async def initialize(self):
        """初始化交易所连接"""
        # 币安
        if self.config.exchange.binance_api_key:
            self.exchanges['binance'] = ccxt.binance({
                'apiKey': self.config.exchange.binance_api_key,
                'secret': self.config.exchange.binance_secret_key,
                'sandbox': self.config.exchange.binance_testnet,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot'
                }
            })
        
        logger.info("智能订单路由器初始化完成")
    
    async def select_best_exchange(self, symbol: str, side: OrderSide, 
                                 quantity: float) -> Tuple[str, float]:
        """选择最佳交易所"""
        best_exchange = "binance"
        best_price = 0.0
        
        try:
            for exchange_name, exchange in self.exchanges.items():
                # 获取订单簿
                orderbook = await self._get_orderbook(exchange, symbol)
                if not orderbook:
                    continue
                
                # 计算预期成交价格
                expected_price = self._calculate_expected_price(
                    orderbook, side, quantity
                )
                
                # 选择最优价格
                if side == OrderSide.BUY:
                    if best_price == 0.0 or expected_price < best_price:
                        best_price = expected_price
                        best_exchange = exchange_name
                else:
                    if expected_price > best_price:
                        best_price = expected_price
                        best_exchange = exchange_name
                        
        except Exception as e:
            logger.error(f"选择最佳交易所错误: {e}")
        
        return best_exchange, best_price
    
    async def _get_orderbook(self, exchange, symbol: str) -> Optional[Dict]:
        """获取订单簿"""
        try:
            orderbook = await asyncio.get_event_loop().run_in_executor(
                None, exchange.fetch_order_book, symbol
            )
            return orderbook
        except Exception as e:
            logger.error(f"获取订单簿错误: {e}")
            return None
    
    def _calculate_expected_price(self, orderbook: Dict, side: OrderSide, 
                                quantity: float) -> float:
        """计算预期成交价格"""
        if side == OrderSide.BUY:
            asks = orderbook['asks']
            remaining_qty = quantity
            total_cost = 0.0
            
            for price, size in asks:
                if remaining_qty <= 0:
                    break
                
                fill_qty = min(remaining_qty, size)
                total_cost += fill_qty * price
                remaining_qty -= fill_qty
            
            return total_cost / quantity if quantity > 0 else 0
        else:
            bids = orderbook['bids']
            remaining_qty = quantity
            total_value = 0.0
            
            for price, size in bids:
                if remaining_qty <= 0:
                    break
                
                fill_qty = min(remaining_qty, size)
                total_value += fill_qty * price
                remaining_qty -= fill_qty
            
            return total_value / quantity if quantity > 0 else 0


class AlgorithmicExecutor:
    """算法执行器"""
    
    def __init__(self, order_router: SmartOrderRouter):
        self.router = order_router
        self.config = get_config()
        
    async def execute_twap(self, order: Order, duration_minutes: int = 30) -> List[Order]:
        """TWAP (时间加权平均价格) 执行"""
        child_orders = []
        slice_count = min(duration_minutes, 60)  # 最多60个切片
        slice_quantity = order.quantity / slice_count
        slice_interval = (duration_minutes * 60) / slice_count  # 秒
        
        try:
            for i in range(slice_count):
                # 创建子订单
                child_order = Order(
                    order_id=f"{order.order_id}_twap_{i}",
                    symbol=order.symbol,
                    side=order.side,
                    order_type=OrderType.MARKET,
                    quantity=slice_quantity,
                    exchange=order.exchange,
                    strategy_id=order.strategy_id
                )
                
                # 执行子订单
                await self._execute_market_order(child_order)
                child_orders.append(child_order)
                
                # 等待下一个切片
                if i < slice_count - 1:
                    await asyncio.sleep(slice_interval)
                    
        except Exception as e:
            logger.error(f"TWAP执行错误: {e}")
        
        return child_orders
    
    async def execute_vwap(self, order: Order, historical_volume: List[float]) -> List[Order]:
        """VWAP (成交量加权平均价格) 执行"""
        child_orders = []
        total_volume = sum(historical_volume)
        
        if total_volume <= 0:
            # 退化为均匀分割
            return await self.execute_twap(order, 30)
        
        try:
            for i, volume in enumerate(historical_volume):
                if volume <= 0:
                    continue
                
                # 根据历史成交量比例分配数量
                slice_quantity = order.quantity * (volume / total_volume)
                
                if slice_quantity < 0.01:  # 最小数量过滤
                    continue
                
                child_order = Order(
                    order_id=f"{order.order_id}_vwap_{i}",
                    symbol=order.symbol,
                    side=order.side,
                    order_type=OrderType.MARKET,
                    quantity=slice_quantity,
                    exchange=order.exchange,
                    strategy_id=order.strategy_id
                )
                
                await self._execute_market_order(child_order)
                child_orders.append(child_order)
                
                # 短暂延迟
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"VWAP执行错误: {e}")
        
        return child_orders
    
    async def _execute_market_order(self, order: Order) -> bool:
        """执行市价单"""
        try:
            exchange = self.router.exchanges.get(order.exchange)
            if not exchange:
                logger.error(f"交易所 {order.exchange} 未找到")
                return False
            
            # 提交订单
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                exchange.create_market_order,
                order.symbol,
                order.side.value,
                order.quantity
            )
            
            # 更新订单状态
            order.status = OrderStatus.FILLED
            order.filled_quantity = result.get('filled', 0)
            order.avg_fill_price = result.get('average', 0)
            order.fee = result.get('fee', {}).get('cost', 0)
            
            logger.info(f"市价单执行成功: {order.order_id}")
            return True
            
        except Exception as e:
            logger.error(f"市价单执行失败: {e}")
            order.status = OrderStatus.REJECTED
            return False


class OrderExecutor:
    """订单执行引擎"""
    
    def __init__(self):
        self.config = get_config()
        self.router = SmartOrderRouter()
        self.algo_executor = AlgorithmicExecutor(self.router)
        self.redis_client = None
        self.kafka_producer = None
        self.pending_orders: Dict[str, Order] = {}
        self.execution_reports: List[ExecutionReport] = []
        
    async def initialize(self):
        """初始化"""
        await self.router.initialize()
        
        # Redis连接
        self.redis_client = redis.Redis(
            host=self.config.redis.host,
            port=self.config.redis.port,
            password=self.config.redis.password,
            db=self.config.redis.db,
            decode_responses=True
        )
        
        # Kafka生产者
        self.kafka_producer = KafkaProducer(
            bootstrap_servers=self.config.kafka.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None
        )
        
        logger.info("订单执行引擎初始化完成")
    
    async def submit_order(self, signal: TradingSignal, position_size: float, 
                         market_data: MarketData) -> Optional[Order]:
        """提交订单"""
        try:
            # 创建订单
            order = Order(
                order_id=str(uuid.uuid4()),
                symbol=signal.symbol,
                side=OrderSide.BUY if signal.signal_type == 'BUY' else OrderSide.SELL,
                order_type=OrderType.MARKET,  # 默认使用市价单
                quantity=position_size,
                strategy_id=signal.strategy_name
            )
            
            # 选择最佳交易所
            best_exchange, expected_price = await self.router.select_best_exchange(
                order.symbol, order.side, order.quantity
            )
            order.exchange = best_exchange
            
            # 滑点检查
            if not self._check_slippage(expected_price, market_data.close):
                logger.warning(f"滑点过大，取消订单: {order.order_id}")
                return None
            
            # 执行订单
            start_time = time.time()
            success = await self._execute_order(order)
            execution_time = (time.time() - start_time) * 1000
            
            # 生成执行报告
            if success:
                execution_report = self._generate_execution_report(
                    order, execution_time, expected_price, market_data.close
                )
                self.execution_reports.append(execution_report)
                
                # 发送到Kafka
                await self._send_execution_report(execution_report)
            
            return order
            
        except Exception as e:
            logger.error(f"提交订单错误: {e}")
            return None
    
    def _check_slippage(self, expected_price: float, market_price: float) -> bool:
        """检查滑点"""
        if expected_price <= 0 or market_price <= 0:
            return False
        
        slippage = abs(expected_price - market_price) / market_price
        return slippage <= self.config.trading.slippage_tolerance
    
    async def _execute_order(self, order: Order) -> bool:
        """执行订单"""
        try:
            if order.order_type == OrderType.MARKET:
                return await self.algo_executor._execute_market_order(order)
            elif order.order_type == OrderType.TWAP:
                child_orders = await self.algo_executor.execute_twap(order)
                return len(child_orders) > 0
            elif order.order_type == OrderType.VWAP:
                # 获取历史成交量数据
                historical_volume = await self._get_historical_volume(order.symbol)
                child_orders = await self.algo_executor.execute_vwap(order, historical_volume)
                return len(child_orders) > 0
            else:
                return await self._execute_limit_order(order)
                
        except Exception as e:
            logger.error(f"订单执行错误: {e}")
            return False
    
    async def _execute_limit_order(self, order: Order) -> bool:
        """执行限价单"""
        try:
            exchange = self.router.exchanges.get(order.exchange)
            if not exchange:
                return False
            
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                exchange.create_limit_order,
                order.symbol,
                order.side.value,
                order.quantity,
                order.price
            )
            
            order.status = OrderStatus.SUBMITTED
            self.pending_orders[order.order_id] = order
            
            # 启动订单监控
            asyncio.create_task(self._monitor_order(order))
            
            return True
            
        except Exception as e:
            logger.error(f"限价单执行失败: {e}")
            return False
    
    async def _monitor_order(self, order: Order):
        """监控订单状态"""
        timeout = self.config.trading.order_timeout
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                exchange = self.router.exchanges.get(order.exchange)
                status = await asyncio.get_event_loop().run_in_executor(
                    None, exchange.fetch_order, order.order_id, order.symbol
                )
                
                # 更新订单状态
                if status['status'] == 'closed':
                    order.status = OrderStatus.FILLED
                    order.filled_quantity = status['filled']
                    order.avg_fill_price = status['average']
                    break
                elif status['status'] == 'canceled':
                    order.status = OrderStatus.CANCELLED
                    break
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"订单监控错误: {e}")
                break
        
        # 超时取消订单
        if order.status == OrderStatus.SUBMITTED:
            await self._cancel_order(order)
    
    async def _cancel_order(self, order: Order) -> bool:
        """取消订单"""
        try:
            exchange = self.router.exchanges.get(order.exchange)
            if exchange:
                await asyncio.get_event_loop().run_in_executor(
                    None, exchange.cancel_order, order.order_id, order.symbol
                )
                order.status = OrderStatus.CANCELLED
                logger.info(f"订单已取消: {order.order_id}")
                return True
        except Exception as e:
            logger.error(f"取消订单失败: {e}")
        
        return False
    
    async def _get_historical_volume(self, symbol: str) -> List[float]:
        """获取历史成交量数据"""
        try:
            # 从Redis获取历史数据
            data = await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.get, f"volume_history:{symbol}"
            )
            
            if data:
                return json.loads(data)
            else:
                # 返回默认分布
                return [1.0] * 24  # 24小时均匀分布
                
        except Exception as e:
            logger.error(f"获取历史成交量错误: {e}")
            return [1.0] * 24
    
    def _generate_execution_report(self, order: Order, execution_time: float,
                                 expected_price: float, market_price: float) -> ExecutionReport:
        """生成执行报告"""
        # 计算滑点
        if order.avg_fill_price > 0 and market_price > 0:
            if order.side == OrderSide.BUY:
                slippage = (order.avg_fill_price - market_price) / market_price
            else:
                slippage = (market_price - order.avg_fill_price) / market_price
        else:
            slippage = 0.0
        
        # 计算市场冲击 (简化版本)
        market_impact = abs(slippage) * 0.5  # 假设滑点的一半是市场冲击
        
        # 计算执行质量评分
        execution_quality = self._calculate_execution_quality(
            execution_time, slippage, market_impact
        )
        
        # 与VWAP比较 (简化版本)
        vwap_comparison = 0.0  # 需要实际VWAP数据
        
        return ExecutionReport(
            order_id=order.order_id,
            execution_time_ms=execution_time,
            slippage=slippage,
            market_impact=market_impact,
            execution_quality=execution_quality,
            vwap_comparison=vwap_comparison,
            success=order.status == OrderStatus.FILLED
        )
    
    def _calculate_execution_quality(self, execution_time: float, 
                                   slippage: float, market_impact: float) -> float:
        """计算执行质量评分"""
        # 时间惩罚 (超过100ms开始惩罚)
        time_penalty = max(0, (execution_time - 100) / 1000)
        
        # 滑点惩罚
        slippage_penalty = abs(slippage) * 10
        
        # 市场冲击惩罚
        impact_penalty = market_impact * 5
        
        # 计算总评分 (1为满分)
        score = 1.0 - (time_penalty + slippage_penalty + impact_penalty)
        return max(0.0, min(1.0, score))
    
    async def _send_execution_report(self, report: ExecutionReport):
        """发送执行报告到Kafka"""
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.kafka_producer.send,
                "execution_reports",
                key=report.order_id,
                value=asdict(report)
            )
        except Exception as e:
            logger.error(f"发送执行报告错误: {e}")
    
    async def get_order_status(self, order_id: str) -> Optional[Order]:
        """获取订单状态"""
        return self.pending_orders.get(order_id)
    
    async def get_execution_stats(self) -> Dict[str, float]:
        """获取执行统计"""
        if not self.execution_reports:
            return {}
        
        recent_reports = self.execution_reports[-100:]  # 最近100个报告
        
        avg_execution_time = sum(r.execution_time_ms for r in recent_reports) / len(recent_reports)
        avg_slippage = sum(abs(r.slippage) for r in recent_reports) / len(recent_reports)
        avg_quality = sum(r.execution_quality for r in recent_reports) / len(recent_reports)
        success_rate = sum(1 for r in recent_reports if r.success) / len(recent_reports)
        
        return {
            'avg_execution_time_ms': avg_execution_time,
            'avg_slippage': avg_slippage,
            'avg_execution_quality': avg_quality,
            'success_rate': success_rate,
            'total_orders': len(recent_reports)
        }
    
    async def cleanup(self):
        """清理资源"""
        if self.kafka_producer:
            self.kafka_producer.close()
        
        if self.redis_client:
            await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.close
            )
        
        logger.info("订单执行引擎资源清理完成")


# 使用示例
async def main():
    executor = OrderExecutor()
    await executor.initialize()
    
    # 模拟交易信号
    from ai_engine.trading_ai import TradingSignal
    signal = TradingSignal(
        symbol="BTCUSDT",
        timestamp=int(time.time() * 1000),
        signal_type="BUY",
        confidence=0.8,
        predicted_price=51000.0,
        strategy_name="test",
        features={},
        risk_score=0.3
    )
    
    market_data = MarketData(
        symbol="BTCUSDT",
        timestamp=int(time.time() * 1000),
        open=50000.0,
        high=50100.0,
        low=49900.0,
        close=50050.0,
        volume=1000.0
    )
    
    # 提交订单
    order = await executor.submit_order(signal, 0.01, market_data)
    if order:
        logger.info(f"订单提交成功: {order}")
    
    # 获取执行统计
    stats = await executor.get_execution_stats()
    logger.info(f"执行统计: {stats}")
    
    await executor.cleanup()


if __name__ == "__main__":
    asyncio.run(main())