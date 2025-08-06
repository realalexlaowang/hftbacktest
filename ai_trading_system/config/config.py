"""
秒级AI交易系统配置文件
"""
from pydantic import BaseSettings, Field
from typing import Dict, List, Optional
import os


class RedisConfig(BaseSettings):
    """Redis配置"""
    host: str = Field(default="localhost", env="REDIS_HOST")
    port: int = Field(default=6379, env="REDIS_PORT")
    password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    db: int = Field(default=0, env="REDIS_DB")
    max_connections: int = Field(default=20, env="REDIS_MAX_CONNECTIONS")
    
    class Config:
        env_prefix = "REDIS_"


class KafkaConfig(BaseSettings):
    """Kafka配置"""
    bootstrap_servers: List[str] = Field(
        default=["localhost:9092"], 
        env="KAFKA_BOOTSTRAP_SERVERS"
    )
    security_protocol: str = Field(default="PLAINTEXT", env="KAFKA_SECURITY_PROTOCOL")
    sasl_mechanism: Optional[str] = Field(default=None, env="KAFKA_SASL_MECHANISM")
    sasl_username: Optional[str] = Field(default=None, env="KAFKA_SASL_USERNAME")
    sasl_password: Optional[str] = Field(default=None, env="KAFKA_SASL_PASSWORD")
    
    # 主题配置
    market_data_topic: str = "market_data"
    news_topic: str = "news_data"
    signal_topic: str = "trading_signals"
    order_topic: str = "orders"
    
    class Config:
        env_prefix = "KAFKA_"


class ClickHouseConfig(BaseSettings):
    """ClickHouse配置"""
    host: str = Field(default="localhost", env="CLICKHOUSE_HOST")
    port: int = Field(default=9000, env="CLICKHOUSE_PORT")
    database: str = Field(default="trading", env="CLICKHOUSE_DATABASE")
    username: str = Field(default="default", env="CLICKHOUSE_USERNAME")
    password: str = Field(default="", env="CLICKHOUSE_PASSWORD")
    
    class Config:
        env_prefix = "CLICKHOUSE_"


class TradingConfig(BaseSettings):
    """交易配置"""
    # 风险管理
    max_position_size: float = Field(default=100000.0, description="最大仓位")
    max_daily_loss: float = Field(default=10000.0, description="日最大亏损")
    max_drawdown: float = Field(default=0.05, description="最大回撤5%")
    
    # 执行参数
    slippage_tolerance: float = Field(default=0.001, description="滑点容忍度0.1%")
    order_timeout: int = Field(default=30, description="订单超时时间(秒)")
    
    # AI模型参数
    model_update_interval: int = Field(default=3600, description="模型更新间隔(秒)")
    prediction_threshold: float = Field(default=0.6, description="预测置信度阈值")
    
    # 市场参数
    supported_symbols: List[str] = Field(
        default=["BTCUSDT", "ETHUSDT", "AAPL", "GOOGL"],
        description="支持的交易标的"
    )
    
    class Config:
        env_prefix = "TRADING_"


class ExchangeConfig(BaseSettings):
    """交易所配置"""
    # 币安配置
    binance_api_key: Optional[str] = Field(default=None, env="BINANCE_API_KEY")
    binance_secret_key: Optional[str] = Field(default=None, env="BINANCE_SECRET_KEY")
    binance_testnet: bool = Field(default=True, env="BINANCE_TESTNET")
    
    # 其他交易所可以类似配置
    # okx_api_key: Optional[str] = None
    # huobi_api_key: Optional[str] = None
    
    class Config:
        env_prefix = "EXCHANGE_"


class MonitoringConfig(BaseSettings):
    """监控配置"""
    prometheus_port: int = Field(default=8000, env="PROMETHEUS_PORT")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="logs/trading.log", env="LOG_FILE")
    
    # 告警配置
    alert_webhook_url: Optional[str] = Field(default=None, env="ALERT_WEBHOOK_URL")
    max_latency_ms: int = Field(default=500, description="最大延迟阈值(毫秒)")
    
    class Config:
        env_prefix = "MONITORING_"


class SystemConfig(BaseSettings):
    """系统总配置"""
    # 环境设置
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=True, env="DEBUG")
    
    # 性能设置
    max_workers: int = Field(default=4, env="MAX_WORKERS")
    queue_size: int = Field(default=10000, env="QUEUE_SIZE")
    
    # 安全设置
    secret_key: str = Field(default="your-secret-key", env="SECRET_KEY")
    encryption_key: Optional[str] = Field(default=None, env="ENCRYPTION_KEY")
    
    # 子配置
    redis: RedisConfig = RedisConfig()
    kafka: KafkaConfig = KafkaConfig()
    clickhouse: ClickHouseConfig = ClickHouseConfig()
    trading: TradingConfig = TradingConfig()
    exchange: ExchangeConfig = ExchangeConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# 全局配置实例
config = SystemConfig()


def get_config() -> SystemConfig:
    """获取系统配置"""
    return config


def update_config(**kwargs) -> None:
    """更新配置"""
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)