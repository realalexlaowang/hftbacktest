"""
高级回测模型
包含精细的市场微观结构模拟：流动性、排队位置、市场冲击、Maker/Taker逻辑等
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import math
from scipy.stats import norm
from collections import deque
import random

@dataclass
class LiquidityProfile:
    """流动性档案"""
    symbol: str
    avg_daily_volume: float
    avg_spread_bps: float      # 平均价差(基点)
    depth_at_bbo: float        # 最优价位深度
    depth_decay_rate: float    # 深度衰减率
    volatility: float          # 波动率
    
class TimeOfDayEffect:
    """时段效应模型"""
    
    def __init__(self):
        # 不同时段的流动性系数（UTC时间）
        self.liquidity_multipliers = {
            # 亚洲时段 (00:00-08:00 UTC)
            0: 0.6, 1: 0.5, 2: 0.5, 3: 0.4, 4: 0.4, 5: 0.5, 6: 0.7, 7: 0.8,
            # 欧洲时段 (08:00-16:00 UTC)  
            8: 1.0, 9: 1.2, 10: 1.3, 11: 1.4, 12: 1.3, 13: 1.2, 14: 1.1, 15: 1.0,
            # 美洲时段 (16:00-24:00 UTC)
            16: 1.1, 17: 1.3, 18: 1.4, 19: 1.5, 20: 1.4, 21: 1.2, 22: 1.0, 23: 0.8
        }
        
        # 波动率系数
        self.volatility_multipliers = {
            0: 0.8, 1: 0.7, 2: 0.6, 3: 0.5, 4: 0.6, 5: 0.7, 6: 0.9, 7: 1.0,
            8: 1.1, 9: 1.3, 10: 1.4, 11: 1.3, 12: 1.2, 13: 1.3, 14: 1.4, 15: 1.2,
            16: 1.2, 17: 1.4, 18: 1.5, 19: 1.6, 20: 1.4, 21: 1.2, 22: 1.0, 23: 0.9
        }
    
    def get_liquidity_factor(self, hour: int) -> float:
        """获取流动性因子"""
        return self.liquidity_multipliers.get(hour, 1.0)
    
    def get_volatility_factor(self, hour: int) -> float:
        """获取波动率因子"""
        return self.volatility_multipliers.get(hour, 1.0)

class MarketImpactModel:
    """市场冲击模型"""
    
    def __init__(self, liquidity_profile: LiquidityProfile):
        self.profile = liquidity_profile
        self.time_effect = TimeOfDayEffect()
        
    def calculate_permanent_impact(self, order_size: float, avg_volume: float) -> float:
        """计算永久冲击 - Almgren & Chriss模型"""
        # 参与率 = 订单大小 / 平均成交量
        participation_rate = order_size / avg_volume
        
        # 永久冲击 = η * σ * (V/ADV)^γ
        # η: 流动性系数, σ: 波动率, V: 订单量, ADV: 平均成交量, γ: 冲击指数
        eta = 0.142  # 典型的流动性系数
        gamma = 0.6  # 冲击指数
        
        permanent_impact = eta * self.profile.volatility * (participation_rate ** gamma)
        return permanent_impact
    
    def calculate_temporary_impact(self, order_size: float, avg_volume: float, 
                                 execution_time_seconds: float) -> float:
        """计算临时冲击"""
        participation_rate = order_size / avg_volume
        
        # 临时冲击随执行时间衰减
        # temporary_impact = β * σ * (V/ADV)^α * exp(-κ*t)
        beta = 0.156   # 临时冲击系数
        alpha = 0.6    # 临时冲击指数
        kappa = 0.1    # 衰减率
        
        base_impact = beta * self.profile.volatility * (participation_rate ** alpha)
        time_decay = math.exp(-kappa * execution_time_seconds / 60)  # 转换为分钟
        
        return base_impact * time_decay
    
    def calculate_total_impact(self, order_size: float, avg_volume: float,
                             execution_time_seconds: float, hour: int) -> Tuple[float, float]:
        """计算总市场冲击"""
        # 时段调整
        liquidity_factor = self.time_effect.get_liquidity_factor(hour)
        volatility_factor = self.time_effect.get_volatility_factor(hour)
        
        # 调整订单大小和波动率
        adjusted_order_size = order_size / liquidity_factor
        adjusted_volatility = self.profile.volatility * volatility_factor
        
        # 临时保存原始波动率
        original_volatility = self.profile.volatility
        self.profile.volatility = adjusted_volatility
        
        permanent = self.calculate_permanent_impact(adjusted_order_size, avg_volume)
        temporary = self.calculate_temporary_impact(adjusted_order_size, avg_volume, execution_time_seconds)
        
        # 恢复原始波动率
        self.profile.volatility = original_volatility
        
        return permanent, temporary

class QueuePositionModel:
    """排队位置模型"""
    
    def __init__(self):
        self.arrival_rate = 2.0      # 每秒订单到达率
        self.cancellation_rate = 0.5  # 取消率
        self.fill_rate = 0.3         # 成交率
        
    def estimate_queue_position(self, price_level: float, best_price: float, 
                              order_count_ahead: int) -> int:
        """估算排队位置"""
        # 基础排队位置
        base_position = order_count_ahead
        
        # 价格偏离程度影响排队优先级
        price_deviation = abs(price_level - best_price) / best_price
        
        # 价格越偏离，排队位置越靠后
        deviation_penalty = int(price_deviation * 1000)  # 转换为整数偏移
        
        return max(1, base_position + deviation_penalty)
    
    def calculate_fill_probability(self, queue_position: int, time_horizon_seconds: float,
                                 market_order_flow_rate: float) -> float:
        """计算成交概率"""
        # 基于泊松过程的排队模型
        # λ = 市场订单流到达率
        # μ = 队列前进率
        
        # 队列前进率取决于市场订单流和取消率
        queue_advancement_rate = market_order_flow_rate * self.fill_rate + self.cancellation_rate
        
        # 在给定时间内，队列位置减少到0的概率
        # P(成交) = 1 - P(排队位置 > 0)
        expected_advancement = queue_advancement_rate * time_horizon_seconds
        
        if expected_advancement >= queue_position:
            return min(0.95, expected_advancement / queue_position)  # 最大95%概率
        else:
            # 泊松分布累计概率
            return 1 - math.exp(-expected_advancement) * sum(
                (expected_advancement ** k) / math.factorial(k) 
                for k in range(queue_position)
            )
    
    def estimate_fill_time(self, queue_position: int, market_order_flow_rate: float) -> float:
        """估算成交时间（秒）"""
        if queue_position <= 0:
            return 0.0
            
        # 期望等待时间 = 排队位置 / (队列前进率)
        queue_advancement_rate = market_order_flow_rate * self.fill_rate + self.cancellation_rate
        
        if queue_advancement_rate > 0:
            expected_wait_time = queue_position / queue_advancement_rate
            return max(1.0, expected_wait_time)  # 至少1秒
        else:
            return float('inf')  # 无法成交

class MakerTakerLogic:
    """Maker/Taker 逻辑模型"""
    
    def __init__(self, fee_model):
        self.fee_model = fee_model
        
    def determine_maker_taker(self, order_price: float, market_price: float,
                            order_side: str, orderbook_state: Dict) -> Tuple[str, float]:
        """确定Maker/Taker状态和费率"""
        
        best_bid = orderbook_state.get('best_bid', 0)
        best_ask = orderbook_state.get('best_ask', float('inf'))
        
        if order_side.upper() == 'BUY':
            if order_price < best_ask:
                # 买单价格低于最优卖价 -> Maker
                return 'maker', self.fee_model.maker_fee
            else:
                # 买单价格高于或等于最优卖价 -> Taker
                return 'taker', self.fee_model.taker_fee
        else:  # SELL
            if order_price > best_bid:
                # 卖单价格高于最优买价 -> Maker
                return 'maker', self.fee_model.maker_fee
            else:
                # 卖单价格低于或等于最优买价 -> Taker
                return 'taker', self.fee_model.taker_fee
    
    def calculate_rebate(self, trade_volume: float, is_maker: bool, 
                        vip_level: int = 0) -> float:
        """计算返佣"""
        if not is_maker:
            return 0.0
            
        # VIP等级返佣表
        rebate_rates = {
            0: 0.0,      # 普通用户无返佣
            1: 0.00005,  # VIP1: 0.005%
            2: 0.0001,   # VIP2: 0.01%
            3: 0.00015,  # VIP3: 0.015%
        }
        
        rebate_rate = rebate_rates.get(vip_level, 0.0)
        return trade_volume * rebate_rate

class OrderBookStateModel:
    """订单簿状态模型"""
    
    def __init__(self, symbol: str, tick_size: float = 0.01):
        self.symbol = symbol
        self.tick_size = tick_size
        
        # 订单簿状态
        self.bid_levels: Dict[float, Dict] = {}  # price -> {qty, order_count, refresh_rate}
        self.ask_levels: Dict[float, Dict] = {}
        
        # 动态参数
        self.spread_target_bps = 10  # 目标价差(基点)
        self.depth_refresh_rate = 0.1  # 深度刷新率
        
    def update_from_market_data(self, timestamp, price: float, volume: float, side: str):
        """从市场数据更新订单簿状态"""
        # 模拟基于价格和成交量的订单簿重构
        spread_bps = self.spread_target_bps * (1 + np.random.normal(0, 0.2))
        spread = price * spread_bps / 10000
        
        # 设置最优买卖价
        best_bid = price - spread / 2
        best_ask = price + spread / 2
        
        # 更新深度
        self._update_depth_levels(best_bid, best_ask, volume)
    
    def _update_depth_levels(self, best_bid: float, best_ask: float, reference_volume: float):
        """更新深度档位"""
        # 清空旧数据
        self.bid_levels.clear()
        self.ask_levels.clear()
        
        # 生成买盘深度（从高到低）
        for i in range(10):  # 10档深度
            price = self._round_to_tick(best_bid - i * self.tick_size)
            # 深度随距离衰减
            decay_factor = math.exp(-i * 0.3)
            quantity = reference_volume * decay_factor * (0.5 + np.random.random() * 0.5)
            order_count = max(1, int(quantity / (reference_volume * 0.1)))
            
            self.bid_levels[price] = {
                'quantity': quantity,
                'order_count': order_count,
                'refresh_rate': self.depth_refresh_rate
            }
        
        # 生成卖盘深度（从低到高）
        for i in range(10):
            price = self._round_to_tick(best_ask + i * self.tick_size)
            decay_factor = math.exp(-i * 0.3)
            quantity = reference_volume * decay_factor * (0.5 + np.random.random() * 0.5)
            order_count = max(1, int(quantity / (reference_volume * 0.1)))
            
            self.ask_levels[price] = {
                'quantity': quantity,
                'order_count': order_count,
                'refresh_rate': self.depth_refresh_rate
            }
    
    def _round_to_tick(self, price: float) -> float:
        """舍入到最小变动单位"""
        return round(price / self.tick_size) * self.tick_size
    
    def get_market_state(self) -> Dict:
        """获取当前市场状态"""
        if not self.bid_levels or not self.ask_levels:
            return {}
            
        best_bid = max(self.bid_levels.keys())
        best_ask = min(self.ask_levels.keys())
        
        return {
            'best_bid': best_bid,
            'best_ask': best_ask,
            'spread': best_ask - best_bid,
            'spread_bps': (best_ask - best_bid) / ((best_ask + best_bid) / 2) * 10000,
            'bid_depth': sum(level['quantity'] for level in self.bid_levels.values()),
            'ask_depth': sum(level['quantity'] for level in self.ask_levels.values()),
            'total_bid_orders': sum(level['order_count'] for level in self.bid_levels.values()),
            'total_ask_orders': sum(level['order_count'] for level in self.ask_levels.values())
        }

class AdvancedLatencyModel:
    """高级延迟模型"""
    
    def __init__(self):
        # 延迟组件（毫秒）
        self.components = {
            'network_base': 5.0,       # 基础网络延迟
            'network_jitter': 2.0,     # 网络抖动
            'exchange_gateway': 3.0,    # 交易所网关处理
            'risk_check': 2.0,         # 风控检查
            'order_routing': 1.0,      # 订单路由
            'matching_engine': 0.5,    # 撮合引擎
            'ack_return': 1.0          # 确认返回
        }
        
        # 负载相关延迟
        self.load_factors = {
            'low': 1.0,      # 低负载时延迟正常
            'medium': 1.5,   # 中等负载时延迟增加50%
            'high': 2.5,     # 高负载时延迟增加150%
            'extreme': 5.0   # 极高负载时延迟增加400%
        }
        
    def calculate_latency(self, market_load: str = 'medium', order_type: str = 'limit',
                         order_size_percentile: float = 0.5) -> float:
        """计算综合延迟"""
        
        # 基础延迟
        base_latency = sum(self.components.values())
        
        # 负载调整
        load_multiplier = self.load_factors.get(market_load, 1.0)
        
        # 订单类型调整
        type_multiplier = {
            'market': 0.8,    # 市价单处理更快
            'limit': 1.0,     # 限价单标准处理
            'stop': 1.2,      # 止损单需要额外检查
            'iceberg': 1.5    # 冰山单需要特殊处理
        }.get(order_type, 1.0)
        
        # 订单大小调整（大订单需要额外风控）
        size_multiplier = 1.0 + order_size_percentile * 0.5
        
        # 随机抖动
        jitter = np.random.normal(0, 1.0)
        
        total_latency = base_latency * load_multiplier * type_multiplier * size_multiplier + jitter
        
        return max(1.0, total_latency)  # 最小1ms延迟
    
    def simulate_network_conditions(self) -> str:
        """模拟网络状况"""
        conditions = ['low', 'medium', 'high', 'extreme']
        probabilities = [0.4, 0.4, 0.15, 0.05]  # 大部分时间网络状况良好
        
        return np.random.choice(conditions, p=probabilities)

class SlippageCalculator:
    """高级滑点计算器"""
    
    def __init__(self, liquidity_profile: LiquidityProfile):
        self.profile = liquidity_profile
        self.impact_model = MarketImpactModel(liquidity_profile)
        
    def calculate_execution_slippage(self, order_size: float, execution_style: str,
                                   market_conditions: Dict) -> Dict[str, float]:
        """计算执行滑点"""
        
        avg_volume = market_conditions.get('avg_volume', self.profile.avg_daily_volume)
        current_spread = market_conditions.get('spread_bps', self.profile.avg_spread_bps)
        volatility = market_conditions.get('volatility', self.profile.volatility)
        hour = market_conditions.get('hour', 12)
        
        # 基础价差成本
        spread_cost = current_spread / 10000 / 2  # 一半价差
        
        # 市场冲击
        permanent_impact, temporary_impact = self.impact_model.calculate_total_impact(
            order_size, avg_volume, 60, hour  # 假设60秒执行
        )
        
        # 执行风格调整
        style_adjustments = {
            'aggressive': {
                'spread_multiplier': 1.0,      # 吃掉全部价差
                'impact_multiplier': 1.2,      # 冲击增加20%
                'timing_cost': 0.0001         # 时机成本
            },
            'passive': {
                'spread_multiplier': 0.2,      # 只承担小部分价差
                'impact_multiplier': 0.8,      # 冲击减少20%
                'timing_cost': 0.0003         # 较高时机成本
            },
            'opportunistic': {
                'spread_multiplier': 0.5,      # 中等价差成本
                'impact_multiplier': 1.0,      # 标准冲击
                'timing_cost': 0.0002         # 中等时机成本
            }
        }
        
        adjustment = style_adjustments.get(execution_style, style_adjustments['opportunistic'])
        
        # 计算各项成本
        adjusted_spread_cost = spread_cost * adjustment['spread_multiplier']
        adjusted_impact = (permanent_impact + temporary_impact) * adjustment['impact_multiplier']
        timing_cost = adjustment['timing_cost']
        
        # 波动率成本（市场波动导致的额外成本）
        volatility_cost = volatility * 0.1 * np.random.random()  # 随机波动成本
        
        total_slippage = adjusted_spread_cost + adjusted_impact + timing_cost + volatility_cost
        
        return {
            'spread_cost': adjusted_spread_cost,
            'market_impact': adjusted_impact,
            'timing_cost': timing_cost,
            'volatility_cost': volatility_cost,
            'total_slippage': total_slippage,
            'slippage_bps': total_slippage * 10000
        }
    
    def estimate_optimal_execution_time(self, order_size: float, 
                                      target_slippage_bps: float) -> float:
        """估算最优执行时间"""
        # 使用Almgren-Chriss最优执行模型
        # 最优执行时间平衡市场冲击和时机风险
        
        participation_rate = order_size / self.profile.avg_daily_volume
        
        # 风险厌恶参数
        risk_aversion = 0.5
        
        # 最优执行时间 (分钟)
        optimal_time_minutes = math.sqrt(
            (participation_rate * self.profile.volatility) / 
            (risk_aversion * target_slippage_bps / 10000)
        ) * 60
        
        return max(1.0, min(optimal_time_minutes * 60, 3600))  # 1秒到1小时之间

# 使用示例和测试函数
def create_btc_liquidity_profile() -> LiquidityProfile:
    """创建BTC流动性档案"""
    return LiquidityProfile(
        symbol='BTCUSDT',
        avg_daily_volume=50000.0,  # 5万BTC日均成交量
        avg_spread_bps=1.0,        # 1基点平均价差
        depth_at_bbo=100.0,        # 最优价位100BTC深度
        depth_decay_rate=0.3,      # 深度衰减率
        volatility=0.03            # 3%日波动率
    )