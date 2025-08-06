"""
风险管理和资金管理系统 - 实时风险控制和仓位管理
"""
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from loguru import logger
import redis
import json

from config.config import get_config
from ai_engine.trading_ai import TradingSignal
from data_pipeline.market_data_collector import MarketData


class RiskLevel(Enum):
    """风险级别"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Position:
    """持仓信息"""
    symbol: str
    side: str  # 'long', 'short'
    size: float  # 持仓数量
    entry_price: float  # 开仓价格
    current_price: float  # 当前价格
    unrealized_pnl: float  # 未实现盈亏
    realized_pnl: float = 0.0  # 已实现盈亏
    timestamp: int = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = int(time.time() * 1000)


@dataclass
class RiskMetrics:
    """风险指标"""
    total_exposure: float  # 总敞口
    leverage: float  # 杠杆倍数
    var_1d: float  # 1日风险价值
    max_drawdown: float  # 最大回撤
    sharpe_ratio: float  # 夏普比率
    win_rate: float  # 胜率
    risk_level: RiskLevel  # 风险级别
    margin_ratio: float  # 保证金比率
    timestamp: int = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = int(time.time() * 1000)


@dataclass
class RiskLimit:
    """风险限制"""
    max_position_size: float  # 最大单仓位
    max_total_exposure: float  # 最大总敞口
    max_daily_loss: float  # 最大日亏损
    max_drawdown: float  # 最大回撤
    min_margin_ratio: float  # 最小保证金比率
    stop_loss_pct: float = 0.02  # 止损比例 2%
    take_profit_pct: float = 0.06  # 止盈比例 6%


class PortfolioManager:
    """投资组合管理器"""
    
    def __init__(self):
        self.config = get_config()
        self.positions: Dict[str, Position] = {}
        self.redis_client = None
        self.cash_balance = 100000.0  # 初始资金
        self.total_value = self.cash_balance
        self.pnl_history = []
        
    async def initialize(self):
        """初始化"""
        self.redis_client = redis.Redis(
            host=self.config.redis.host,
            port=self.config.redis.port,
            password=self.config.redis.password,
            db=self.config.redis.db,
            decode_responses=True
        )
        logger.info("投资组合管理器初始化完成")
    
    async def update_position(self, symbol: str, size: float, price: float, side: str) -> bool:
        """更新持仓"""
        try:
            if symbol in self.positions:
                # 更新现有持仓
                position = self.positions[symbol]
                
                if side == position.side:
                    # 加仓
                    total_size = position.size + size
                    avg_price = (position.entry_price * position.size + price * size) / total_size
                    position.size = total_size
                    position.entry_price = avg_price
                else:
                    # 减仓或反向开仓
                    if size >= position.size:
                        # 平仓并反向开仓
                        realized_pnl = self._calculate_pnl(position, price)
                        position.realized_pnl += realized_pnl
                        self.cash_balance += realized_pnl
                        
                        # 反向开仓
                        position.side = side
                        position.size = size - position.size
                        position.entry_price = price
                    else:
                        # 部分平仓
                        close_ratio = size / position.size
                        realized_pnl = self._calculate_pnl(position, price) * close_ratio
                        position.realized_pnl += realized_pnl
                        position.size -= size
                        self.cash_balance += realized_pnl
            else:
                # 新建持仓
                self.positions[symbol] = Position(
                    symbol=symbol,
                    side=side,
                    size=size,
                    entry_price=price,
                    current_price=price,
                    unrealized_pnl=0.0
                )
            
            # 更新现金余额
            order_value = size * price
            if side == 'long':
                self.cash_balance -= order_value
            else:
                self.cash_balance += order_value  # 做空获得现金
            
            # 缓存到Redis
            await self._cache_positions()
            
            return True
            
        except Exception as e:
            logger.error(f"更新持仓错误: {e}")
            return False
    
    def _calculate_pnl(self, position: Position, current_price: float) -> float:
        """计算盈亏"""
        if position.side == 'long':
            return (current_price - position.entry_price) * position.size
        else:
            return (position.entry_price - current_price) * position.size
    
    async def update_market_data(self, market_data: MarketData):
        """更新市场数据并计算未实现盈亏"""
        symbol = market_data.symbol
        if symbol in self.positions:
            position = self.positions[symbol]
            position.current_price = market_data.close
            position.unrealized_pnl = self._calculate_pnl(position, market_data.close)
            
            # 更新总价值
            self._update_total_value()
    
    def _update_total_value(self):
        """更新总价值"""
        unrealized_total = sum([pos.unrealized_pnl for pos in self.positions.values()])
        self.total_value = self.cash_balance + unrealized_total
    
    async def _cache_positions(self):
        """缓存持仓信息到Redis"""
        try:
            positions_data = {symbol: asdict(pos) for symbol, pos in self.positions.items()}
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.redis_client.setex(
                    "portfolio:positions",
                    3600,
                    json.dumps(positions_data)
                )
            )
        except Exception as e:
            logger.error(f"缓存持仓错误: {e}")
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """获取持仓信息"""
        return self.positions.get(symbol)
    
    def get_total_exposure(self) -> float:
        """获取总敞口"""
        return sum([abs(pos.size * pos.current_price) for pos in self.positions.values()])
    
    def get_leverage(self) -> float:
        """获取杠杆倍数"""
        exposure = self.get_total_exposure()
        return exposure / self.total_value if self.total_value > 0 else 0


class RiskManager:
    """风险管理器"""
    
    def __init__(self, portfolio_manager: PortfolioManager):
        self.config = get_config()
        self.portfolio = portfolio_manager
        self.risk_limits = self._initialize_risk_limits()
        self.daily_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_value = 100000.0
        self.trade_history = []
        
    def _initialize_risk_limits(self) -> RiskLimit:
        """初始化风险限制"""
        return RiskLimit(
            max_position_size=self.config.trading.max_position_size,
            max_total_exposure=self.config.trading.max_position_size * 3,  # 3倍单仓位
            max_daily_loss=self.config.trading.max_daily_loss,
            max_drawdown=self.config.trading.max_drawdown,
            min_margin_ratio=0.1  # 10%最小保证金比率
        )
    
    async def validate_signal(self, signal: TradingSignal, 
                            market_data: MarketData) -> Tuple[bool, str, float]:
        """
        验证交易信号
        返回: (是否通过, 原因, 建议仓位大小)
        """
        try:
            # 1. 基础风险检查
            if not await self._basic_risk_check():
                return False, "基础风险检查未通过", 0.0
            
            # 2. 仓位大小检查
            position_size = self._calculate_position_size(signal, market_data)
            if position_size <= 0:
                return False, "计算的仓位大小无效", 0.0
            
            # 3. 单仓位限制检查
            if not self._check_position_limit(signal.symbol, position_size, market_data.close):
                return False, "超过单仓位限制", 0.0
            
            # 4. 总敞口检查
            if not self._check_exposure_limit(position_size, market_data.close):
                return False, "超过总敞口限制", 0.0
            
            # 5. 日亏损检查
            if not self._check_daily_loss_limit():
                return False, "超过日亏损限制", 0.0
            
            # 6. 最大回撤检查
            if not self._check_drawdown_limit():
                return False, "超过最大回撤限制", 0.0
            
            # 7. 信号质量检查
            if not self._check_signal_quality(signal):
                return False, "信号质量不足", 0.0
            
            # 8. 市场状态检查
            if not self._check_market_conditions(market_data):
                return False, "市场条件不适合交易", 0.0
            
            return True, "风险检查通过", position_size
            
        except Exception as e:
            logger.error(f"风险验证错误: {e}")
            return False, f"风险验证错误: {e}", 0.0
    
    async def _basic_risk_check(self) -> bool:
        """基础风险检查"""
        # 检查资金是否充足
        if self.portfolio.cash_balance <= 0:
            return False
        
        # 检查杠杆是否过高
        leverage = self.portfolio.get_leverage()
        if leverage > 5.0:  # 最大5倍杠杆
            return False
        
        return True
    
    def _calculate_position_size(self, signal: TradingSignal, market_data: MarketData) -> float:
        """计算仓位大小"""
        try:
            # 基础仓位 (根据总资金的百分比)
            base_size = self.portfolio.total_value * 0.1  # 10%基础仓位
            
            # 根据信号置信度调整
            confidence_multiplier = signal.confidence
            
            # 根据风险评分调整 (风险越高，仓位越小)
            risk_multiplier = max(0.1, 1.0 - signal.risk_score)
            
            # 根据波动率调整
            volatility = signal.features.get('volatility', 0.02)
            volatility_multiplier = max(0.5, 1.0 - volatility * 10)
            
            # 计算最终仓位大小
            position_value = base_size * confidence_multiplier * risk_multiplier * volatility_multiplier
            position_size = position_value / market_data.close
            
            # 确保仓位不为负
            return max(0, position_size)
            
        except Exception as e:
            logger.error(f"计算仓位大小错误: {e}")
            return 0.0
    
    def _check_position_limit(self, symbol: str, size: float, price: float) -> bool:
        """检查单仓位限制"""
        position_value = size * price
        current_position = self.portfolio.get_position(symbol)
        
        if current_position:
            total_value = abs(current_position.size * price) + position_value
        else:
            total_value = position_value
        
        return total_value <= self.risk_limits.max_position_size
    
    def _check_exposure_limit(self, size: float, price: float) -> bool:
        """检查总敞口限制"""
        new_exposure = size * price
        current_exposure = self.portfolio.get_total_exposure()
        total_exposure = current_exposure + new_exposure
        
        return total_exposure <= self.risk_limits.max_total_exposure
    
    def _check_daily_loss_limit(self) -> bool:
        """检查日亏损限制"""
        return abs(self.daily_pnl) < self.risk_limits.max_daily_loss
    
    def _check_drawdown_limit(self) -> bool:
        """检查最大回撤限制"""
        return self.max_drawdown < self.risk_limits.max_drawdown
    
    def _check_signal_quality(self, signal: TradingSignal) -> bool:
        """检查信号质量"""
        # 置信度过滤
        if signal.confidence < self.config.trading.prediction_threshold:
            return False
        
        # 风险评分过滤
        if signal.risk_score > 0.8:
            return False
        
        return True
    
    def _check_market_conditions(self, market_data: MarketData) -> bool:
        """检查市场条件"""
        # 检查是否有足够的成交量
        if market_data.volume <= 0:
            return False
        
        # 检查价格是否异常
        if market_data.close <= 0:
            return False
        
        # 可以添加更多市场条件检查
        # 例如：波动率检查、流动性检查等
        
        return True
    
    async def update_daily_pnl(self):
        """更新日盈亏"""
        total_unrealized = sum([pos.unrealized_pnl for pos in self.portfolio.positions.values()])
        total_realized = sum([pos.realized_pnl for pos in self.portfolio.positions.values()])
        self.daily_pnl = total_unrealized + total_realized
        
        # 更新最大回撤
        current_value = self.portfolio.total_value
        if current_value > self.peak_value:
            self.peak_value = current_value
        
        drawdown = (self.peak_value - current_value) / self.peak_value
        self.max_drawdown = max(self.max_drawdown, drawdown)
    
    async def get_risk_metrics(self) -> RiskMetrics:
        """获取风险指标"""
        await self.update_daily_pnl()
        
        # 计算VaR (简化版本)
        portfolio_volatility = self._calculate_portfolio_volatility()
        var_1d = self.portfolio.total_value * portfolio_volatility * 2.33  # 99% VaR
        
        # 计算夏普比率 (简化版本)
        if len(self.trade_history) > 0:
            returns = [trade['pnl'] / self.portfolio.total_value for trade in self.trade_history[-30:]]
            if len(returns) > 1:
                sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
        
        # 计算胜率
        if len(self.trade_history) > 0:
            winning_trades = sum(1 for trade in self.trade_history if trade['pnl'] > 0)
            win_rate = winning_trades / len(self.trade_history)
        else:
            win_rate = 0
        
        # 确定风险级别
        risk_level = self._assess_risk_level()
        
        return RiskMetrics(
            total_exposure=self.portfolio.get_total_exposure(),
            leverage=self.portfolio.get_leverage(),
            var_1d=var_1d,
            max_drawdown=self.max_drawdown,
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate,
            risk_level=risk_level,
            margin_ratio=self.portfolio.cash_balance / self.portfolio.total_value
        )
    
    def _calculate_portfolio_volatility(self) -> float:
        """计算投资组合波动率"""
        if len(self.portfolio.pnl_history) < 2:
            return 0.02  # 默认2%日波动率
        
        returns = []
        for i in range(1, len(self.portfolio.pnl_history)):
            prev_value = self.portfolio.pnl_history[i-1]
            curr_value = self.portfolio.pnl_history[i]
            if prev_value > 0:
                ret = (curr_value - prev_value) / prev_value
                returns.append(ret)
        
        if len(returns) > 0:
            return np.std(returns)
        else:
            return 0.02
    
    def _assess_risk_level(self) -> RiskLevel:
        """评估风险级别"""
        risk_score = 0
        
        # 杠杆风险
        leverage = self.portfolio.get_leverage()
        if leverage > 3:
            risk_score += 3
        elif leverage > 2:
            risk_score += 2
        elif leverage > 1:
            risk_score += 1
        
        # 回撤风险
        if self.max_drawdown > 0.15:
            risk_score += 3
        elif self.max_drawdown > 0.1:
            risk_score += 2
        elif self.max_drawdown > 0.05:
            risk_score += 1
        
        # 敞口风险
        exposure_ratio = self.portfolio.get_total_exposure() / self.portfolio.total_value
        if exposure_ratio > 0.8:
            risk_score += 2
        elif exposure_ratio > 0.6:
            risk_score += 1
        
        # 确定风险级别
        if risk_score >= 6:
            return RiskLevel.CRITICAL
        elif risk_score >= 4:
            return RiskLevel.HIGH
        elif risk_score >= 2:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def add_trade_record(self, symbol: str, side: str, size: float, price: float, pnl: float):
        """添加交易记录"""
        trade_record = {
            'symbol': symbol,
            'side': side,
            'size': size,
            'price': price,
            'pnl': pnl,
            'timestamp': int(time.time() * 1000)
        }
        self.trade_history.append(trade_record)
        
        # 保持最近1000条记录
        if len(self.trade_history) > 1000:
            self.trade_history = self.trade_history[-1000:]
    
    async def emergency_stop(self) -> bool:
        """紧急停止交易"""
        try:
            logger.warning("触发紧急停止！")
            
            # 平掉所有持仓
            for symbol, position in self.portfolio.positions.items():
                logger.warning(f"紧急平仓: {symbol}, 仓位: {position.size}")
                # 这里应该调用订单执行引擎进行平仓
                # await order_executor.close_position(symbol)
            
            return True
            
        except Exception as e:
            logger.error(f"紧急停止错误: {e}")
            return False


# 使用示例
async def main():
    portfolio = PortfolioManager()
    await portfolio.initialize()
    
    risk_manager = RiskManager(portfolio)
    
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
    
    # 验证信号
    is_valid, reason, size = await risk_manager.validate_signal(signal, market_data)
    logger.info(f"信号验证结果: {is_valid}, 原因: {reason}, 仓位大小: {size}")
    
    # 获取风险指标
    risk_metrics = await risk_manager.get_risk_metrics()
    logger.info(f"风险指标: {risk_metrics}")


if __name__ == "__main__":
    asyncio.run(main())