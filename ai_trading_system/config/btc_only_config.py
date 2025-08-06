"""
BTC/USDT 单一品种交易优化配置
专门针对比特币交易进行优化的配置文件
"""
from pydantic import BaseSettings, Field
from typing import Dict, List, Optional


class BTCTradingConfig(BaseSettings):
    """BTC专用交易配置"""
    
    # 基础配置
    symbol: str = "BTCUSDT"
    base_currency: str = "BTC"
    quote_currency: str = "USDT"
    
    # 风险管理 - 针对BTC波动性调整
    max_position_size: float = Field(default=50000.0, description="最大仓位(USDT)")
    max_daily_loss: float = Field(default=5000.0, description="日最大亏损(USDT)")
    max_drawdown: float = Field(default=0.08, description="最大回撤8%")
    position_sizing_base: float = Field(default=0.05, description="基础仓位比例5%")
    
    # BTC特定的技术指标参数
    rsi_period: int = 14
    rsi_oversold: float = 25  # BTC超卖线
    rsi_overbought: float = 75  # BTC超买线
    
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    
    sma_short: int = 20
    sma_long: int = 50
    
    # BTC价格区间配置
    price_levels: Dict[str, float] = {
        "support_1": 40000.0,    # 主要支撑位
        "support_2": 35000.0,    # 次要支撑位
        "resistance_1": 50000.0, # 主要阻力位
        "resistance_2": 55000.0, # 次要阻力位
    }
    
    # 交易时间配置（UTC时间）
    trading_hours: Dict[str, List[int]] = {
        "active_hours": [8, 9, 10, 11, 12, 13, 14, 15, 16, 17],  # 活跃交易时间
        "avoid_hours": [22, 23, 0, 1, 2, 3, 4, 5],  # 避免交易时间（流动性差）
    }
    
    # BTC特定的止损止盈
    stop_loss_pct: float = 0.03  # 3% 止损
    take_profit_pct: float = 0.08  # 8% 止盈
    trailing_stop_pct: float = 0.02  # 2% 移动止损
    
    # 仓位管理策略
    pyramid_levels: int = 3  # 金字塔加仓层数
    pyramid_ratio: float = 0.5  # 每层仓位递减比例
    
    # 市场状态识别
    volatility_threshold_low: float = 0.02  # 低波动率阈值
    volatility_threshold_high: float = 0.06  # 高波动率阈值
    volume_threshold: float = 1.5  # 成交量倍数阈值
    
    # AI模型参数 - BTC专用
    prediction_threshold: float = 0.65  # 提高置信度阈值
    model_retrain_hours: int = 24  # 每24小时重训练模型
    lookback_periods: int = 168  # 7天数据用于训练
    
    # 执行优化
    order_size_min: float = 10.0  # 最小订单金额(USDT)
    slippage_tolerance: float = 0.0005  # 0.05% 滑点容忍度
    execution_delay_ms: int = 50  # 执行延迟(毫秒)
    
    class Config:
        env_prefix = "BTC_"


class BTCDataConfig(BaseSettings):
    """BTC数据收集优化配置"""
    
    # 数据收集频率
    tick_interval_ms: int = 100  # 100ms收集一次tick数据
    kline_intervals: List[str] = ["1m", "5m", "15m", "1h", "4h", "1d"]
    
    # 订单簿深度
    orderbook_depth: int = 20  # 20档订单簿
    orderbook_update_ms: int = 100  # 100ms更新频率
    
    # 历史数据
    history_days: int = 30  # 保留30天历史数据
    
    # 数据质量控制
    price_change_threshold: float = 0.1  # 价格异常变动阈值10%
    volume_anomaly_threshold: float = 5.0  # 成交量异常倍数
    
    class Config:
        env_prefix = "BTC_DATA_"


class BTCMonitoringConfig(BaseSettings):
    """BTC交易监控配置"""
    
    # 关键指标监控
    key_metrics: List[str] = [
        "btc_price",
        "position_size", 
        "unrealized_pnl",
        "daily_pnl",
        "drawdown",
        "sharpe_ratio",
        "win_rate"
    ]
    
    # 告警阈值
    alert_thresholds: Dict[str, float] = {
        "price_drop_5min": 0.02,  # 5分钟跌幅超过2%
        "volume_spike": 3.0,      # 成交量激增3倍
        "position_loss": 0.05,    # 持仓亏损5%
        "system_latency": 200,    # 系统延迟超过200ms
    }
    
    # 通知设置
    notification_channels: List[str] = ["webhook", "email", "sms"]
    critical_alerts_only: bool = False
    
    class Config:
        env_prefix = "BTC_MONITOR_"


def get_btc_optimized_config():
    """获取BTC优化配置"""
    return {
        "trading": BTCTradingConfig(),
        "data": BTCDataConfig(), 
        "monitoring": BTCMonitoringConfig()
    }