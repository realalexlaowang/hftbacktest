"""
市场数据收集器 - 负责从多个数据源收集实时市场数据
"""
import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import ccxt
import websockets
from kafka import KafkaProducer
from loguru import logger
import redis
from config.config import get_config


@dataclass
class MarketData:
    """市场数据结构"""
    symbol: str
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_volume: Optional[float] = None
    ask_volume: Optional[float] = None
    exchange: str = "binance"


@dataclass
class OrderBookData:
    """订单簿数据"""
    symbol: str
    timestamp: int
    bids: List[List[float]]  # [[price, volume], ...]
    asks: List[List[float]]
    exchange: str = "binance"


class MarketDataCollector:
    """市场数据收集器"""
    
    def __init__(self):
        self.config = get_config()
        self.kafka_producer = None
        self.redis_client = None
        self.exchanges = {}
        self.websocket_connections = {}
        self.running = False
        
    async def initialize(self):
        """初始化连接"""
        # 初始化Kafka生产者
        self.kafka_producer = KafkaProducer(
            bootstrap_servers=self.config.kafka.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None,
            batch_size=16384,
            linger_ms=1,  # 低延迟设置
            compression_type='lz4'
        )
        
        # 初始化Redis客户端
        self.redis_client = redis.Redis(
            host=self.config.redis.host,
            port=self.config.redis.port,
            password=self.config.redis.password,
            db=self.config.redis.db,
            decode_responses=True
        )
        
        # 初始化交易所连接
        await self._initialize_exchanges()
        
        logger.info("市场数据收集器初始化完成")
    
    async def _initialize_exchanges(self):
        """初始化交易所连接"""
        # 币安交易所
        if self.config.exchange.binance_api_key:
            self.exchanges['binance'] = ccxt.binance({
                'apiKey': self.config.exchange.binance_api_key,
                'secret': self.config.exchange.binance_secret_key,
                'sandbox': self.config.exchange.binance_testnet,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot'  # spot, future, margin
                }
            })
        
        # 可以添加其他交易所
        # self.exchanges['okx'] = ccxt.okx(...)
    
    async def start_collection(self, symbols: List[str]):
        """开始数据收集"""
        self.running = True
        logger.info(f"开始收集数据，标的: {symbols}")
        
        # 创建任务列表
        tasks = []
        
        # REST API数据收集任务
        for symbol in symbols:
            task = asyncio.create_task(
                self._collect_rest_data(symbol)
            )
            tasks.append(task)
        
        # WebSocket数据收集任务
        for symbol in symbols:
            task = asyncio.create_task(
                self._collect_websocket_data(symbol)
            )
            tasks.append(task)
        
        # 订单簿数据收集
        for symbol in symbols:
            task = asyncio.create_task(
                self._collect_orderbook_data(symbol)
            )
            tasks.append(task)
        
        # 等待所有任务完成
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _collect_rest_data(self, symbol: str):
        """通过REST API收集OHLCV数据"""
        while self.running:
            try:
                for exchange_name, exchange in self.exchanges.items():
                    # 获取1分钟K线数据
                    ohlcv = await self._fetch_ohlcv(exchange, symbol, '1m', 1)
                    if ohlcv:
                        market_data = MarketData(
                            symbol=symbol,
                            timestamp=int(ohlcv[0][0]),
                            open=float(ohlcv[0][1]),
                            high=float(ohlcv[0][2]),
                            low=float(ohlcv[0][3]),
                            close=float(ohlcv[0][4]),
                            volume=float(ohlcv[0][5]),
                            exchange=exchange_name
                        )
                        
                        # 发送到Kafka
                        await self._send_to_kafka(
                            topic=self.config.kafka.market_data_topic,
                            key=f"{symbol}_{exchange_name}",
                            data=asdict(market_data)
                        )
                        
                        # 缓存到Redis
                        await self._cache_to_redis(f"market_data:{symbol}", market_data)
                
                # 1秒收集一次
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"REST数据收集错误: {e}")
                await asyncio.sleep(5)
    
    async def _fetch_ohlcv(self, exchange, symbol: str, timeframe: str, limit: int):
        """异步获取OHLCV数据"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            exchange.fetch_ohlcv, 
            symbol, 
            timeframe, 
            None, 
            limit
        )
    
    async def _collect_websocket_data(self, symbol: str):
        """通过WebSocket收集实时tick数据"""
        # 币安WebSocket URL
        ws_url = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@ticker"
        
        while self.running:
            try:
                async with websockets.connect(ws_url) as websocket:
                    logger.info(f"WebSocket连接建立: {symbol}")
                    
                    async for message in websocket:
                        if not self.running:
                            break
                            
                        data = json.loads(message)
                        
                        # 解析Ticker数据
                        market_data = MarketData(
                            symbol=data['s'],
                            timestamp=int(data['E']),
                            open=float(data['o']),
                            high=float(data['h']),
                            low=float(data['l']),
                            close=float(data['c']),
                            volume=float(data['v']),
                            bid=float(data['b']),
                            ask=float(data['a']),
                            bid_volume=float(data['B']),
                            ask_volume=float(data['A']),
                            exchange="binance"
                        )
                        
                        # 发送到Kafka
                        await self._send_to_kafka(
                            topic=self.config.kafka.market_data_topic,
                            key=f"{symbol}_binance_tick",
                            data=asdict(market_data)
                        )
                        
                        # 实时缓存
                        await self._cache_to_redis(f"tick_data:{symbol}", market_data)
                        
            except Exception as e:
                logger.error(f"WebSocket数据收集错误 {symbol}: {e}")
                await asyncio.sleep(5)
    
    async def _collect_orderbook_data(self, symbol: str):
        """收集订单簿数据"""
        # 币安深度数据WebSocket
        ws_url = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@depth10@100ms"
        
        while self.running:
            try:
                async with websockets.connect(ws_url) as websocket:
                    logger.info(f"订单簿WebSocket连接建立: {symbol}")
                    
                    async for message in websocket:
                        if not self.running:
                            break
                            
                        data = json.loads(message)
                        
                        orderbook_data = OrderBookData(
                            symbol=symbol,
                            timestamp=int(time.time() * 1000),
                            bids=[[float(price), float(qty)] for price, qty in data['bids']],
                            asks=[[float(price), float(qty)] for price, qty in data['asks']],
                            exchange="binance"
                        )
                        
                        # 发送到Kafka
                        await self._send_to_kafka(
                            topic="orderbook_data",
                            key=f"{symbol}_orderbook",
                            data=asdict(orderbook_data)
                        )
                        
                        # 缓存最新订单簿
                        await self._cache_to_redis(f"orderbook:{symbol}", orderbook_data)
                        
            except Exception as e:
                logger.error(f"订单簿数据收集错误 {symbol}: {e}")
                await asyncio.sleep(5)
    
    async def _send_to_kafka(self, topic: str, key: str, data: Dict[str, Any]):
        """发送数据到Kafka"""
        try:
            # 添加时间戳和延迟计算
            data['pipeline_timestamp'] = int(time.time() * 1000)
            
            # 异步发送
            future = self.kafka_producer.send(topic, key=key, value=data)
            
            # 不等待确认，以减少延迟
            # result = await asyncio.wrap_future(future)
            
        except Exception as e:
            logger.error(f"Kafka发送错误: {e}")
    
    async def _cache_to_redis(self, key: str, data: Any):
        """缓存数据到Redis"""
        try:
            if isinstance(data, (MarketData, OrderBookData)):
                data_dict = asdict(data)
            else:
                data_dict = data
                
            # 设置过期时间为60秒
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.redis_client.setex(
                    key,
                    60,
                    json.dumps(data_dict)
                )
            )
            
        except Exception as e:
            logger.error(f"Redis缓存错误: {e}")
    
    async def get_latest_data(self, symbol: str) -> Optional[MarketData]:
        """获取最新市场数据"""
        try:
            data = await asyncio.get_event_loop().run_in_executor(
                None,
                self.redis_client.get,
                f"tick_data:{symbol}"
            )
            
            if data:
                data_dict = json.loads(data)
                return MarketData(**data_dict)
            return None
            
        except Exception as e:
            logger.error(f"获取最新数据错误: {e}")
            return None
    
    async def get_orderbook(self, symbol: str) -> Optional[OrderBookData]:
        """获取最新订单簿"""
        try:
            data = await asyncio.get_event_loop().run_in_executor(
                None,
                self.redis_client.get,
                f"orderbook:{symbol}"
            )
            
            if data:
                data_dict = json.loads(data)
                return OrderBookData(**data_dict)
            return None
            
        except Exception as e:
            logger.error(f"获取订单簿错误: {e}")
            return None
    
    def stop_collection(self):
        """停止数据收集"""
        self.running = False
        logger.info("停止数据收集")
    
    async def cleanup(self):
        """清理资源"""
        self.running = False
        
        if self.kafka_producer:
            self.kafka_producer.close()
        
        if self.redis_client:
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.redis_client.close
            )
        
        logger.info("数据收集器资源清理完成")


# 使用示例
async def main():
    collector = MarketDataCollector()
    await collector.initialize()
    
    symbols = ["BTCUSDT", "ETHUSDT"]
    try:
        await collector.start_collection(symbols)
    except KeyboardInterrupt:
        logger.info("收到停止信号")
    finally:
        await collector.cleanup()


if __name__ == "__main__":
    asyncio.run(main())