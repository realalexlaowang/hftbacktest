"""
秒级AI交易系统主程序
"""
import asyncio
import signal
import sys
from typing import Optional
from loguru import logger
from contextlib import asynccontextmanager

from config.config import get_config
from data_pipeline.market_data_collector import MarketDataCollector
from ai_engine.trading_ai import TradingAI
from risk_management.risk_manager import PortfolioManager, RiskManager
from execution_engine.order_executor import OrderExecutor
from monitoring.system_monitor import SystemMonitor


class TradingSystem:
    """AI交易系统主类"""
    
    def __init__(self):
        self.config = get_config()
        self.running = False
        
        # 核心组件
        self.data_collector: Optional[MarketDataCollector] = None
        self.ai_engine: Optional[TradingAI] = None
        self.portfolio_manager: Optional[PortfolioManager] = None
        self.risk_manager: Optional[RiskManager] = None
        self.order_executor: Optional[OrderExecutor] = None
        self.system_monitor: Optional[SystemMonitor] = None
        
        # 支持的交易标的
        self.symbols = self.config.trading.supported_symbols
        
        # 设置日志
        self._setup_logging()
        
    def _setup_logging(self):
        """设置日志系统"""
        logger.remove()  # 移除默认处理器
        
        # 控制台输出
        logger.add(
            sys.stdout,
            level=self.config.monitoring.log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                   "<level>{message}</level>",
            colorize=True
        )
        
        # 文件输出
        logger.add(
            self.config.monitoring.log_file,
            level=self.config.monitoring.log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="100 MB",
            retention="30 days",
            compression="zip"
        )
        
        logger.info("日志系统初始化完成")
    
    async def initialize(self):
        """初始化所有组件"""
        try:
            logger.info("开始初始化AI交易系统...")
            
            # 1. 初始化数据收集器
            self.data_collector = MarketDataCollector()
            await self.data_collector.initialize()
            logger.info("✓ 数据收集器初始化完成")
            
            # 2. 初始化AI引擎
            self.ai_engine = TradingAI()
            logger.info("✓ AI引擎初始化完成")
            
            # 3. 初始化投资组合管理器
            self.portfolio_manager = PortfolioManager()
            await self.portfolio_manager.initialize()
            logger.info("✓ 投资组合管理器初始化完成")
            
            # 4. 初始化风险管理器
            self.risk_manager = RiskManager(self.portfolio_manager)
            logger.info("✓ 风险管理器初始化完成")
            
            # 5. 初始化订单执行引擎
            self.order_executor = OrderExecutor()
            await self.order_executor.initialize()
            logger.info("✓ 订单执行引擎初始化完成")
            
            # 6. 初始化系统监控
            self.system_monitor = SystemMonitor()
            await self.system_monitor.initialize()
            logger.info("✓ 系统监控初始化完成")
            
            logger.info("🚀 AI交易系统初始化完成！")
            
        except Exception as e:
            logger.error(f"系统初始化失败: {e}")
            raise
    
    async def start(self):
        """启动交易系统"""
        try:
            self.running = True
            logger.info("🔥 启动秒级AI交易系统...")
            
            # 创建主要任务
            tasks = [
                # 数据收集任务
                asyncio.create_task(
                    self.data_collector.start_collection(self.symbols),
                    name="data_collection"
                ),
                
                # 交易决策任务
                asyncio.create_task(
                    self._trading_loop(),
                    name="trading_loop"
                ),
                
                # 系统监控任务
                asyncio.create_task(
                    self.system_monitor.start_monitoring(),
                    name="system_monitoring"
                ),
                
                # 风险监控任务
                asyncio.create_task(
                    self._risk_monitoring_loop(),
                    name="risk_monitoring"
                )
            ]
            
            logger.info("所有任务已启动，系统正在运行...")
            
            # 等待所有任务完成或异常
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"系统运行错误: {e}")
            raise
        finally:
            await self.shutdown()
    
    async def _trading_loop(self):
        """主交易循环"""
        logger.info("启动交易决策循环")
        
        while self.running:
            try:
                # 处理每个交易标的
                for symbol in self.symbols:
                    if not self.running:
                        break
                    
                    # 获取最新市场数据
                    market_data = await self.data_collector.get_latest_data(symbol)
                    if not market_data:
                        continue
                    
                    # 获取订单簿数据
                    orderbook_data = await self.data_collector.get_orderbook(symbol)
                    
                    # 更新投资组合市场数据
                    await self.portfolio_manager.update_market_data(market_data)
                    
                    # AI预测
                    signal = await self.ai_engine.predict(
                        symbol, market_data, orderbook_data
                    )
                    
                    if signal and signal.signal_type != 'HOLD':
                        # 风险验证
                        is_valid, reason, position_size = await self.risk_manager.validate_signal(
                            signal, market_data
                        )
                        
                        if is_valid and position_size > 0:
                            # 执行订单
                            order = await self.order_executor.submit_order(
                                signal, position_size, market_data
                            )
                            
                            if order:
                                logger.info(f"订单提交成功: {symbol} {signal.signal_type} {position_size}")
                                
                                # 更新持仓
                                side = 'long' if signal.signal_type == 'BUY' else 'short'
                                await self.portfolio_manager.update_position(
                                    symbol, position_size, market_data.close, side
                                )
                                
                                # 记录交易
                                self.risk_manager.add_trade_record(
                                    symbol, side, position_size, market_data.close, 0
                                )
                                
                                # 记录监控指标
                                self.system_monitor.metrics.record_order(
                                    'submitted', symbol
                                )
                                self.system_monitor.metrics.record_prediction(
                                    signal.signal_type, signal.strategy_name
                                )
                            else:
                                logger.warning(f"订单执行失败: {symbol}")
                                self.system_monitor.metrics.record_error(
                                    'execution', 'order_failed'
                                )
                        else:
                            logger.debug(f"信号被风控拒绝: {symbol} - {reason}")
                
                # 短暂等待，控制循环频率
                await asyncio.sleep(1)  # 1秒循环一次
                
            except Exception as e:
                logger.error(f"交易循环错误: {e}")
                self.system_monitor.metrics.record_error('trading', 'loop_error')
                await asyncio.sleep(5)
    
    async def _risk_monitoring_loop(self):
        """风险监控循环"""
        logger.info("启动风险监控循环")
        
        while self.running:
            try:
                # 更新风险指标
                risk_metrics = await self.risk_manager.get_risk_metrics()
                
                # 检查紧急情况
                if risk_metrics.risk_level.value == 'critical':
                    logger.critical("触发紧急风险警报！")
                    
                    # 紧急停止交易
                    emergency_success = await self.risk_manager.emergency_stop()
                    if emergency_success:
                        logger.critical("紧急停止执行成功")
                    else:
                        logger.critical("紧急停止执行失败！")
                
                # 更新监控指标
                self.system_monitor.metrics.update_positions(
                    len(self.portfolio_manager.positions)
                )
                self.system_monitor.metrics.update_pnl(
                    self.portfolio_manager.total_value - 100000  # 减去初始资金
                )
                self.system_monitor.metrics.update_drawdown(
                    risk_metrics.max_drawdown
                )
                
                # 每分钟检查一次
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"风险监控错误: {e}")
                self.system_monitor.metrics.record_error('risk', 'monitoring_error')
                await asyncio.sleep(30)
    
    async def shutdown(self):
        """优雅关闭系统"""
        logger.info("开始关闭AI交易系统...")
        
        self.running = False
        
        # 停止数据收集
        if self.data_collector:
            self.data_collector.stop_collection()
            await self.data_collector.cleanup()
            logger.info("✓ 数据收集器已关闭")
        
        # 停止订单执行
        if self.order_executor:
            await self.order_executor.cleanup()
            logger.info("✓ 订单执行引擎已关闭")
        
        # 停止系统监控
        if self.system_monitor:
            self.system_monitor.stop_monitoring()
            await self.system_monitor.cleanup()
            logger.info("✓ 系统监控已关闭")
        
        logger.info("🛑 AI交易系统已安全关闭")
    
    def setup_signal_handlers(self):
        """设置信号处理器"""
        def signal_handler(signum, frame):
            logger.info(f"收到信号 {signum}，开始优雅关闭...")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def health_check(self):
        """系统健康检查"""
        if not self.system_monitor:
            return {"status": "not_initialized"}
        
        return await self.system_monitor.get_health_status()


async def main():
    """主函数"""
    # 创建交易系统
    trading_system = TradingSystem()
    
    try:
        # 设置信号处理
        trading_system.setup_signal_handlers()
        
        # 初始化系统
        await trading_system.initialize()
        
        # 启动系统
        await trading_system.start()
        
    except KeyboardInterrupt:
        logger.info("收到中断信号")
    except Exception as e:
        logger.error(f"系统运行异常: {e}")
        raise
    finally:
        await trading_system.shutdown()


if __name__ == "__main__":
    # 设置事件循环策略（Linux）
    if sys.platform == 'linux':
        import uvloop
        uvloop.install()
    
    # 运行主程序
    asyncio.run(main())