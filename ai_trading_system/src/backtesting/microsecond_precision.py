"""
微秒级精度时间处理和高精度市场数据结构
为高频交易提供精确的时间模拟
"""

import numpy as np
import time
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import struct
import numba
from numba import types
import pandas as pd

class TimeUnit(Enum):
    """时间单位枚举"""
    NANOSECOND = 1
    MICROSECOND = 1000
    MILLISECOND = 1_000_000
    SECOND = 1_000_000_000

@dataclass
class HighPrecisionTimestamp:
    """高精度时间戳 - 纳秒级精度"""
    nanoseconds_since_epoch: int
    
    @classmethod
    def now(cls) -> 'HighPrecisionTimestamp':
        """获取当前时间戳"""
        return cls(time.time_ns())
    
    @classmethod
    def from_datetime(cls, dt: datetime) -> 'HighPrecisionTimestamp':
        """从datetime对象创建"""
        return cls(int(dt.timestamp() * 1_000_000_000))
    
    @classmethod
    def from_microseconds(cls, microseconds: int) -> 'HighPrecisionTimestamp':
        """从微秒时间戳创建"""
        return cls(microseconds * 1000)
    
    @classmethod
    def from_milliseconds(cls, milliseconds: int) -> 'HighPrecisionTimestamp':
        """从毫秒时间戳创建"""
        return cls(milliseconds * 1_000_000)
    
    def to_datetime(self) -> datetime:
        """转换为datetime对象"""
        return datetime.fromtimestamp(self.nanoseconds_since_epoch / 1_000_000_000, tz=timezone.utc)
    
    def to_microseconds(self) -> int:
        """转换为微秒时间戳"""
        return self.nanoseconds_since_epoch // 1000
    
    def to_milliseconds(self) -> int:
        """转换为毫秒时间戳"""
        return self.nanoseconds_since_epoch // 1_000_000
    
    def add_nanoseconds(self, nanoseconds: int) -> 'HighPrecisionTimestamp':
        """添加纳秒"""
        return HighPrecisionTimestamp(self.nanoseconds_since_epoch + nanoseconds)
    
    def add_microseconds(self, microseconds: int) -> 'HighPrecisionTimestamp':
        """添加微秒"""
        return self.add_nanoseconds(microseconds * 1000)
    
    def add_milliseconds(self, milliseconds: int) -> 'HighPrecisionTimestamp':
        """添加毫秒"""
        return self.add_nanoseconds(milliseconds * 1_000_000)
    
    def difference_nanoseconds(self, other: 'HighPrecisionTimestamp') -> int:
        """计算纳秒级时间差"""
        return self.nanoseconds_since_epoch - other.nanoseconds_since_epoch
    
    def difference_microseconds(self, other: 'HighPrecisionTimestamp') -> float:
        """计算微秒级时间差"""
        return self.difference_nanoseconds(other) / 1000.0
    
    def __lt__(self, other: 'HighPrecisionTimestamp') -> bool:
        return self.nanoseconds_since_epoch < other.nanoseconds_since_epoch
    
    def __le__(self, other: 'HighPrecisionTimestamp') -> bool:
        return self.nanoseconds_since_epoch <= other.nanoseconds_since_epoch
    
    def __gt__(self, other: 'HighPrecisionTimestamp') -> bool:
        return self.nanoseconds_since_epoch > other.nanoseconds_since_epoch
    
    def __ge__(self, other: 'HighPrecisionTimestamp') -> bool:
        return self.nanoseconds_since_epoch >= other.nanoseconds_since_epoch
    
    def __eq__(self, other: 'HighPrecisionTimestamp') -> bool:
        return self.nanoseconds_since_epoch == other.nanoseconds_since_epoch

# 定义Numba兼容的数据类型
tick_dtype = np.dtype([
    ('timestamp_ns', 'i8'),      # 纳秒时间戳
    ('exchange_timestamp_ns', 'i8'),  # 交易所时间戳
    ('receive_timestamp_ns', 'i8'),   # 接收时间戳
    ('symbol_id', 'i4'),         # 符号ID (数值编码)
    ('price', 'f8'),             # 价格
    ('quantity', 'f8'),          # 数量
    ('side', 'i1'),              # 买卖方向 (1=buy, -1=sell, 0=unknown)
    ('trade_id', 'i8'),          # 交易ID
    ('sequence_number', 'i8'),    # 序列号
    ('flags', 'i2')              # 标志位 (用于标记特殊事件)
])

orderbook_level_dtype = np.dtype([
    ('timestamp_ns', 'i8'),      # 时间戳
    ('price', 'f8'),             # 价格
    ('quantity', 'f8'),          # 数量
    ('order_count', 'i4'),       # 订单数
    ('side', 'i1'),              # 买卖方向 (1=bid, -1=ask)
    ('level', 'i2')              # 档位 (0=最优, 1=次优, ...)
])

@numba.jit(nopython=True, cache=True)
def calculate_timestamp_latency(send_time_ns: int, receive_time_ns: int) -> float:
    """计算时间戳延迟 (微秒)"""
    return (receive_time_ns - send_time_ns) / 1000.0

@numba.jit(nopython=True, cache=True)
def filter_ticks_by_time_range(tick_array, start_time_ns: int, end_time_ns: int):
    """按时间范围过滤tick数据"""
    mask = (tick_array['timestamp_ns'] >= start_time_ns) & (tick_array['timestamp_ns'] <= end_time_ns)
    return tick_array[mask]

@numba.jit(nopython=True, cache=True)
def calculate_price_changes(prices, timestamps_ns):
    """计算价格变化 - 考虑时间权重"""
    n = len(prices)
    if n <= 1:
        return np.zeros(n)
    
    price_changes = np.zeros(n)
    for i in range(1, n):
        time_delta_us = (timestamps_ns[i] - timestamps_ns[i-1]) / 1000.0
        price_change = (prices[i] - prices[i-1]) / prices[i-1]
        
        # 时间标准化 (每微秒的价格变化)
        if time_delta_us > 0:
            price_changes[i] = price_change / time_delta_us * 1_000_000  # 每秒变化率
        else:
            price_changes[i] = 0.0
    
    return price_changes

@numba.jit(nopython=True, cache=True)
def calculate_volume_weighted_price(prices, quantities, timestamps_ns, window_us: int = 1000000):
    """计算时间窗口内的成交量加权平均价格"""
    n = len(prices)
    vwap = np.zeros(n)
    
    for i in range(n):
        current_time = timestamps_ns[i]
        window_start = current_time - window_us * 1000  # 转换为纳秒
        
        # 找到窗口内的数据
        total_value = 0.0
        total_volume = 0.0
        
        for j in range(max(0, i - 1000), i + 1):  # 限制搜索范围提高性能
            if j < n and timestamps_ns[j] >= window_start:
                total_value += prices[j] * quantities[j]
                total_volume += quantities[j]
        
        vwap[i] = total_value / total_volume if total_volume > 0 else prices[i]
    
    return vwap

@numba.jit(nopython=True, cache=True)
def detect_price_anomalies(prices, timestamps_ns, z_threshold: float = 3.0):
    """检测价格异常 - 基于Z-score"""
    n = len(prices)
    if n < 10:
        return np.zeros(n, dtype=np.bool_)
    
    anomalies = np.zeros(n, dtype=np.bool_)
    window_size = min(100, n // 2)
    
    for i in range(window_size, n):
        # 计算滚动窗口的统计量
        window_prices = prices[i - window_size:i]
        mean_price = np.mean(window_prices)
        std_price = np.std(window_prices)
        
        if std_price > 0:
            z_score = abs(prices[i] - mean_price) / std_price
            anomalies[i] = z_score > z_threshold
    
    return anomalies

class HighFrequencyDataProcessor:
    """高频数据处理器"""
    
    def __init__(self, symbol: str, buffer_size: int = 1_000_000):
        self.symbol = symbol
        self.buffer_size = buffer_size
        
        # 预分配内存缓冲区
        self.tick_buffer = np.zeros(buffer_size, dtype=tick_dtype)
        self.orderbook_buffer = np.zeros(buffer_size * 20, dtype=orderbook_level_dtype)  # 20档深度
        
        # 当前位置指针
        self.tick_pointer = 0
        self.orderbook_pointer = 0
        
        # 符号ID映射
        self.symbol_to_id = {symbol: 1}
        self.id_to_symbol = {1: symbol}
        
        # 性能统计
        self.processed_ticks = 0
        self.processing_time_ns = 0
        
    def add_tick(self, timestamp: HighPrecisionTimestamp, price: float, quantity: float, 
                 side: str, trade_id: int = 0, exchange_timestamp: Optional[HighPrecisionTimestamp] = None,
                 receive_timestamp: Optional[HighPrecisionTimestamp] = None) -> bool:
        """添加tick数据"""
        if self.tick_pointer >= self.buffer_size:
            return False  # 缓冲区已满
        
        start_time = time.time_ns()
        
        # 设置默认时间戳
        if exchange_timestamp is None:
            exchange_timestamp = timestamp
        if receive_timestamp is None:
            receive_timestamp = HighPrecisionTimestamp.now()
        
        # 转换side
        side_value = 1 if side.upper() == 'BUY' else (-1 if side.upper() == 'SELL' else 0)
        
        # 填充tick数据
        tick = self.tick_buffer[self.tick_pointer]
        tick['timestamp_ns'] = timestamp.nanoseconds_since_epoch
        tick['exchange_timestamp_ns'] = exchange_timestamp.nanoseconds_since_epoch
        tick['receive_timestamp_ns'] = receive_timestamp.nanoseconds_since_epoch
        tick['symbol_id'] = self.symbol_to_id.get(self.symbol, 0)
        tick['price'] = price
        tick['quantity'] = quantity
        tick['side'] = side_value
        tick['trade_id'] = trade_id
        tick['sequence_number'] = self.processed_ticks
        tick['flags'] = 0
        
        self.tick_pointer += 1
        self.processed_ticks += 1
        
        # 更新性能统计
        self.processing_time_ns += time.time_ns() - start_time
        
        return True
    
    def add_orderbook_level(self, timestamp: HighPrecisionTimestamp, price: float, 
                           quantity: float, order_count: int, side: str, level: int) -> bool:
        """添加订单簿档位数据"""
        if self.orderbook_pointer >= len(self.orderbook_buffer):
            return False
        
        side_value = 1 if side.upper() == 'BID' else -1
        
        level_data = self.orderbook_buffer[self.orderbook_pointer]
        level_data['timestamp_ns'] = timestamp.nanoseconds_since_epoch
        level_data['price'] = price
        level_data['quantity'] = quantity
        level_data['order_count'] = order_count
        level_data['side'] = side_value
        level_data['level'] = level
        
        self.orderbook_pointer += 1
        return True
    
    def get_ticks_in_range(self, start_time: HighPrecisionTimestamp, 
                          end_time: HighPrecisionTimestamp) -> np.ndarray:
        """获取时间范围内的tick数据"""
        if self.tick_pointer == 0:
            return np.array([], dtype=tick_dtype)
        
        current_data = self.tick_buffer[:self.tick_pointer]
        return filter_ticks_by_time_range(
            current_data,
            start_time.nanoseconds_since_epoch,
            end_time.nanoseconds_since_epoch
        )
    
    def calculate_microsecond_metrics(self, window_microseconds: int = 1000) -> Dict[str, Any]:
        """计算微秒级指标"""
        if self.tick_pointer < 2:
            return {}
        
        current_data = self.tick_buffer[:self.tick_pointer]
        prices = current_data['price']
        quantities = current_data['quantity']
        timestamps = current_data['timestamp_ns']
        
        # 计算各种微秒级指标
        price_changes = calculate_price_changes(prices, timestamps)
        vwap = calculate_volume_weighted_price(prices, quantities, timestamps, window_microseconds)
        anomalies = detect_price_anomalies(prices, timestamps)
        
        # 计算延迟统计
        latencies_us = np.zeros(len(current_data))
        for i in range(len(current_data)):
            latencies_us[i] = calculate_timestamp_latency(
                current_data[i]['exchange_timestamp_ns'],
                current_data[i]['receive_timestamp_ns']
            )
        
        return {
            'total_ticks': self.tick_pointer,
            'price_range': (float(np.min(prices)), float(np.max(prices))),
            'volume_total': float(np.sum(quantities)),
            'avg_latency_us': float(np.mean(latencies_us)),
            'max_latency_us': float(np.max(latencies_us)),
            'price_volatility': float(np.std(price_changes[price_changes != 0])) if np.any(price_changes != 0) else 0.0,
            'anomaly_count': int(np.sum(anomalies)),
            'last_vwap': float(vwap[-1]) if len(vwap) > 0 else 0.0,
            'processing_speed_ticks_per_sec': self.processed_ticks / (self.processing_time_ns / 1_000_000_000) if self.processing_time_ns > 0 else 0
        }
    
    def export_to_pandas(self, start_idx: int = 0, end_idx: Optional[int] = None) -> pd.DataFrame:
        """导出为Pandas DataFrame"""
        if end_idx is None:
            end_idx = self.tick_pointer
        
        data = self.tick_buffer[start_idx:end_idx]
        
        df = pd.DataFrame({
            'timestamp': [HighPrecisionTimestamp(ts).to_datetime() for ts in data['timestamp_ns']],
            'exchange_timestamp': [HighPrecisionTimestamp(ts).to_datetime() for ts in data['exchange_timestamp_ns']],
            'receive_timestamp': [HighPrecisionTimestamp(ts).to_datetime() for ts in data['receive_timestamp_ns']],
            'price': data['price'],
            'quantity': data['quantity'],
            'side': ['BUY' if s == 1 else ('SELL' if s == -1 else 'UNKNOWN') for s in data['side']],
            'trade_id': data['trade_id'],
            'sequence_number': data['sequence_number'],
            'latency_us': [(data[i]['receive_timestamp_ns'] - data[i]['exchange_timestamp_ns']) / 1000.0 
                          for i in range(len(data))]
        })
        
        return df
    
    def reset_buffers(self):
        """重置缓冲区"""
        self.tick_pointer = 0
        self.orderbook_pointer = 0
        self.processed_ticks = 0
        self.processing_time_ns = 0

class MicrosecondEventScheduler:
    """微秒级事件调度器"""
    
    def __init__(self):
        self.events: List[Tuple[HighPrecisionTimestamp, str, Any]] = []
        self.current_time = HighPrecisionTimestamp.now()
        
    def schedule_event(self, timestamp: HighPrecisionTimestamp, event_type: str, event_data: Any):
        """调度事件"""
        self.events.append((timestamp, event_type, event_data))
        # 保持时间排序
        self.events.sort(key=lambda x: x[0].nanoseconds_since_epoch)
    
    def schedule_delay_event(self, delay_microseconds: int, event_type: str, event_data: Any):
        """调度延迟事件"""
        target_time = self.current_time.add_microseconds(delay_microseconds)
        self.schedule_event(target_time, event_type, event_data)
    
    def advance_time_to(self, target_time: HighPrecisionTimestamp) -> List[Tuple[str, Any]]:
        """推进时间并返回触发的事件"""
        triggered_events = []
        
        while self.events and self.events[0][0] <= target_time:
            event_time, event_type, event_data = self.events.pop(0)
            triggered_events.append((event_type, event_data))
        
        self.current_time = target_time
        return triggered_events
    
    def advance_time_by_microseconds(self, microseconds: int) -> List[Tuple[str, Any]]:
        """按微秒推进时间"""
        target_time = self.current_time.add_microseconds(microseconds)
        return self.advance_time_to(target_time)
    
    def get_next_event_time(self) -> Optional[HighPrecisionTimestamp]:
        """获取下一个事件时间"""
        return self.events[0][0] if self.events else None
    
    def get_pending_events_count(self) -> int:
        """获取待处理事件数量"""
        return len(self.events)

def benchmark_microsecond_precision():
    """微秒精度性能基准测试"""
    print("⏱️  微秒级精度性能基准测试")
    print("=" * 50)
    
    # 测试1: 时间戳操作性能
    print("\n📊 测试1: 时间戳操作")
    n_operations = 1_000_000
    
    start_time = time.perf_counter()
    timestamps = []
    for _ in range(n_operations):
        ts = HighPrecisionTimestamp.now()
        timestamps.append(ts)
    timestamp_creation_time = time.perf_counter() - start_time
    
    print(f"时间戳创建: {n_operations:,} 次")
    print(f"总时间: {timestamp_creation_time:.4f}秒")
    print(f"平均时间: {(timestamp_creation_time / n_operations) * 1_000_000:.2f} 微秒/次")
    print(f"创建速度: {n_operations / timestamp_creation_time:,.0f} 次/秒")
    
    # 测试2: 高频数据处理
    print("\n📊 测试2: 高频数据处理")
    processor = HighFrequencyDataProcessor("BTCUSDT", buffer_size=n_operations)
    
    start_time = time.perf_counter()
    base_timestamp = HighPrecisionTimestamp.now()
    
    for i in range(100000):  # 10万个tick
        timestamp = base_timestamp.add_microseconds(i)
        price = 45000.0 + np.random.normal(0, 10)
        quantity = np.random.uniform(0.1, 10.0)
        side = 'BUY' if np.random.random() > 0.5 else 'SELL'
        
        processor.add_tick(timestamp, price, quantity, side, trade_id=i)
    
    processing_time = time.perf_counter() - start_time
    
    print(f"Tick处理: {processor.processed_ticks:,} 个")
    print(f"处理时间: {processing_time:.4f}秒")
    print(f"处理速度: {processor.processed_ticks / processing_time:,.0f} tick/秒")
    print(f"平均处理时间: {(processing_time / processor.processed_ticks) * 1_000_000:.2f} 微秒/tick")
    
    # 测试3: 微秒级指标计算
    print("\n📊 测试3: 微秒级指标计算")
    start_time = time.perf_counter()
    metrics = processor.calculate_microsecond_metrics(window_microseconds=1000)
    metrics_time = time.perf_counter() - start_time
    
    print(f"指标计算时间: {metrics_time * 1000:.2f} 毫秒")
    print(f"数据点数: {metrics['total_ticks']:,}")
    print(f"平均延迟: {metrics['avg_latency_us']:.2f} 微秒")
    print(f"价格波动率: {metrics['price_volatility']:.6f}")
    print(f"异常点数: {metrics['anomaly_count']}")
    
    # 测试4: 事件调度器
    print("\n📊 测试4: 事件调度器")
    scheduler = MicrosecondEventScheduler()
    
    start_time = time.perf_counter()
    # 调度大量微秒级事件
    for i in range(10000):
        delay = np.random.randint(1, 1000)  # 1-1000微秒延迟
        scheduler.schedule_delay_event(delay, "test_event", {"id": i})
    
    scheduling_time = time.perf_counter() - start_time
    
    print(f"事件调度: {scheduler.get_pending_events_count():,} 个事件")
    print(f"调度时间: {scheduling_time * 1000:.2f} 毫秒")
    print(f"调度速度: {scheduler.get_pending_events_count() / scheduling_time:,.0f} 事件/秒")
    
    # 测试事件处理
    start_time = time.perf_counter()
    target_time = scheduler.current_time.add_microseconds(2000)  # 推进2000微秒
    triggered_events = scheduler.advance_time_to(target_time)
    event_processing_time = time.perf_counter() - start_time
    
    print(f"触发事件: {len(triggered_events):,} 个")
    print(f"处理时间: {event_processing_time * 1000:.2f} 毫秒")
    
    print("\n✅ 微秒精度测试完成!")
    
    return {
        'timestamp_creation_speed': n_operations / timestamp_creation_time,
        'tick_processing_speed': processor.processed_ticks / processing_time,
        'avg_tick_latency_us': (processing_time / processor.processed_ticks) * 1_000_000,
        'event_scheduling_speed': scheduler.get_pending_events_count() / scheduling_time,
        'metrics_calculation_time_ms': metrics_time * 1000
    }

def create_synthetic_hft_data(duration_seconds: int = 60, frequency_hz: int = 1000) -> HighFrequencyDataProcessor:
    """创建合成高频交易数据"""
    print(f"📈 生成 {duration_seconds}秒 {frequency_hz}Hz 的合成高频数据...")
    
    total_ticks = duration_seconds * frequency_hz
    processor = HighFrequencyDataProcessor("BTCUSDT", buffer_size=total_ticks * 2)
    
    base_price = 45000.0
    base_timestamp = HighPrecisionTimestamp.now()
    microsecond_interval = 1_000_000 // frequency_hz  # 微秒间隔
    
    for i in range(total_ticks):
        # 生成价格随机游走
        price_change = np.random.normal(0, 0.1)  # 0.1的价格变化
        base_price += price_change
        base_price = max(30000, min(60000, base_price))  # 限制价格范围
        
        # 生成时间戳
        timestamp = base_timestamp.add_microseconds(i * microsecond_interval)
        
        # 添加少量时间抖动 (模拟网络延迟)
        jitter_us = np.random.randint(-50, 51)  # ±50微秒抖动
        exchange_timestamp = timestamp.add_microseconds(jitter_us)
        receive_timestamp = timestamp.add_microseconds(jitter_us + np.random.randint(10, 100))
        
        # 生成成交量和方向
        quantity = np.random.exponential(2.0)  # 指数分布成交量
        side = 'BUY' if np.random.random() > 0.5 else 'SELL'
        
        processor.add_tick(
            timestamp=timestamp,
            price=base_price,
            quantity=quantity,
            side=side,
            trade_id=i,
            exchange_timestamp=exchange_timestamp,
            receive_timestamp=receive_timestamp
        )
        
        # 每10000个tick报告进度
        if (i + 1) % 10000 == 0:
            print(f"  生成进度: {i + 1:,}/{total_ticks:,} ({(i + 1)/total_ticks*100:.1f}%)")
    
    print(f"✅ 合成数据生成完成: {processor.processed_ticks:,} 个tick")
    return processor

if __name__ == "__main__":
    # 运行微秒精度基准测试
    benchmark_results = benchmark_microsecond_precision()
    
    print(f"\n🎯 微秒精度性能总结:")
    print(f"- 时间戳创建速度: {benchmark_results['timestamp_creation_speed']:,.0f} 次/秒")
    print(f"- Tick处理速度: {benchmark_results['tick_processing_speed']:,.0f} tick/秒")
    print(f"- 平均Tick延迟: {benchmark_results['avg_tick_latency_us']:.2f} 微秒")
    print(f"- 事件调度速度: {benchmark_results['event_scheduling_speed']:,.0f} 事件/秒")
    print(f"- 指标计算时间: {benchmark_results['metrics_calculation_time_ms']:.2f} 毫秒")
    
    # 生成示例数据
    print(f"\n🔬 生成示例高频数据...")
    hft_data = create_synthetic_hft_data(duration_seconds=10, frequency_hz=1000)
    
    # 计算并显示指标
    metrics = hft_data.calculate_microsecond_metrics()
    print(f"\n📊 数据质量指标:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print(f"\n🎉 微秒级精度系统就绪！")