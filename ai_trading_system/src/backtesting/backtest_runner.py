"""
高保真回测运行器
整合所有高级模型，提供完整的回测流程和详细的分析报告
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import asdict

from .high_fidelity_backtest_engine import (
    HighFidelityBacktestEngine, BacktestOrder, MarketDataTick, 
    OrderType, OrderSide, LatencyModel, FeeModel
)
from .advanced_backtest_models import (
    LiquidityProfile, MarketImpactModel, QueuePositionModel,
    MakerTakerLogic, OrderBookStateModel, AdvancedLatencyModel,
    SlippageCalculator, create_btc_liquidity_profile
)

logger = logging.getLogger(__name__)

class BacktestConfig:
    """回测配置"""
    
    def __init__(self):
        self.start_date: datetime = datetime(2023, 1, 1)
        self.end_date: datetime = datetime(2023, 12, 31)
        self.initial_balance: float = 100000.0
        self.symbols: List[str] = ['BTCUSDT']
        
        # 高保真设置
        self.enable_latency_simulation: bool = True
        self.enable_slippage_simulation: bool = True
        self.enable_queue_simulation: bool = True
        self.enable_maker_taker_logic: bool = True
        self.enable_market_impact: bool = True
        self.enable_time_effects: bool = True
        
        # 数据设置
        self.tick_data_path: str = "./data/tick_data/"
        self.orderbook_data_path: str = "./data/orderbook_data/"
        
        # 输出设置
        self.output_path: str = "./backtest_results/"
        self.save_detailed_logs: bool = True
        self.generate_plots: bool = True

class BacktestDataLoader:
    """回测数据加载器"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        
    def load_tick_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """加载tick数据"""
        # 这里应该从实际数据源加载，这里模拟数据
        return self._generate_synthetic_tick_data(symbol, start_date, end_date)
    
    def load_orderbook_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """加载订单簿数据"""
        # 这里应该从实际数据源加载，这里模拟数据
        return self._generate_synthetic_orderbook_data(symbol, start_date, end_date)
    
    def _generate_synthetic_tick_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """生成合成tick数据用于测试"""
        # 生成模拟的高频tick数据
        time_range = pd.date_range(start_date, end_date, freq='1S')
        
        # 使用几何布朗运动生成价格
        initial_price = 45000.0 if symbol == 'BTCUSDT' else 1000.0
        drift = 0.0001  # 年化漂移率
        volatility = 0.02  # 年化波动率
        
        dt = 1.0 / (365 * 24 * 3600)  # 1秒的年化时间
        price_changes = np.random.normal(drift * dt, volatility * np.sqrt(dt), len(time_range))
        prices = initial_price * np.exp(np.cumsum(price_changes))
        
        # 生成成交量（基于泊松分布）
        base_volume = 10.0
        volumes = np.random.exponential(base_volume, len(time_range))
        
        # 生成买卖方向
        sides = np.random.choice(['buy', 'sell'], len(time_range), p=[0.5, 0.5])
        
        tick_data = pd.DataFrame({
            'timestamp': time_range,
            'symbol': symbol,
            'price': prices,
            'volume': volumes,
            'side': sides
        })
        
        return tick_data
    
    def _generate_synthetic_orderbook_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """生成合成订单簿数据"""
        # 简化的订单簿数据生成
        time_range = pd.date_range(start_date, end_date, freq='100ms')  # 100ms频率的订单簿快照
        
        orderbook_data = []
        base_price = 45000.0 if symbol == 'BTCUSDT' else 1000.0
        
        for timestamp in time_range:
            # 生成买卖价差
            spread = np.random.uniform(0.01, 0.05) * base_price / 100  # 1-5基点价差
            mid_price = base_price + np.random.normal(0, base_price * 0.001)  # 价格随机游走
            
            best_bid = mid_price - spread / 2
            best_ask = mid_price + spread / 2
            
            # 生成深度数据
            bid_depths = []
            ask_depths = []
            
            for i in range(10):  # 10档深度
                bid_price = best_bid - i * 0.01
                ask_price = best_ask + i * 0.01
                
                # 深度随档位衰减
                bid_qty = np.random.exponential(50) * np.exp(-i * 0.2)
                ask_qty = np.random.exponential(50) * np.exp(-i * 0.2)
                
                bid_depths.append({'price': bid_price, 'quantity': bid_qty})
                ask_depths.append({'price': ask_price, 'quantity': ask_qty})
            
            orderbook_data.append({
                'timestamp': timestamp,
                'symbol': symbol,
                'bids': bid_depths,
                'asks': ask_depths
            })
        
        return pd.DataFrame(orderbook_data)

class BacktestRunner:
    """高保真回测运行器"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.data_loader = BacktestDataLoader(config)
        
        # 创建流动性档案
        self.liquidity_profiles = {
            'BTCUSDT': create_btc_liquidity_profile()
        }
        
        # 初始化引擎和模型
        self.engine: Optional[HighFidelityBacktestEngine] = None
        self.advanced_models: Dict[str, Any] = {}
        
        # 策略函数
        self.strategy_func: Optional[Callable] = None
        
        # 结果存储
        self.results: Dict[str, Any] = {}
        
    def set_strategy(self, strategy_func: Callable):
        """设置交易策略函数"""
        self.strategy_func = strategy_func
    
    def initialize_engine(self):
        """初始化回测引擎和高级模型"""
        # 创建高级延迟模型
        advanced_latency = AdvancedLatencyModel()
        
        # 创建手续费模型
        fee_model = FeeModel()
        
        # 创建回测引擎
        self.engine = HighFidelityBacktestEngine(
            initial_balance=self.config.initial_balance,
            symbols=self.config.symbols,
            latency_model=LatencyModel(),
            commission_model=fee_model
        )
        
        # 为每个标的创建高级模型
        for symbol in self.config.symbols:
            profile = self.liquidity_profiles.get(symbol)
            if profile:
                self.advanced_models[symbol] = {
                    'market_impact': MarketImpactModel(profile),
                    'queue_position': QueuePositionModel(),
                    'maker_taker': MakerTakerLogic(fee_model),
                    'orderbook_state': OrderBookStateModel(symbol),
                    'slippage_calculator': SlippageCalculator(profile),
                    'advanced_latency': advanced_latency
                }
    
    async def run_backtest(self) -> Dict[str, Any]:
        """运行完整回测"""
        logger.info("开始高保真回测...")
        
        # 初始化
        self.initialize_engine()
        
        if not self.strategy_func:
            raise ValueError("必须设置交易策略函数")
        
        # 加载数据
        all_data = {}
        for symbol in self.config.symbols:
            logger.info(f"加载{symbol}数据...")
            tick_data = self.data_loader.load_tick_data(symbol, self.config.start_date, self.config.end_date)
            orderbook_data = self.data_loader.load_orderbook_data(symbol, self.config.start_date, self.config.end_date)
            
            all_data[symbol] = {
                'ticks': tick_data,
                'orderbook': orderbook_data
            }
        
        # 按时间排序所有数据
        all_ticks = []
        for symbol, data in all_data.items():
            for _, row in data['ticks'].iterrows():
                tick = MarketDataTick(
                    timestamp=row['timestamp'],
                    symbol=row['symbol'],
                    price=row['price'],
                    volume=row['volume'],
                    side=row['side']
                )
                all_ticks.append(tick)
        
        all_ticks.sort(key=lambda x: x.timestamp)
        
        logger.info(f"总共加载{len(all_ticks)}个tick数据点")
        
        # 运行回测主循环
        processed_ticks = 0
        for tick in all_ticks:
            # 更新引擎
            self.engine.add_market_data(tick)
            
            # 更新订单簿模型
            if tick.symbol in self.advanced_models:
                orderbook_model = self.advanced_models[tick.symbol]['orderbook_state']
                orderbook_model.update_from_market_data(tick.timestamp, tick.price, tick.volume, tick.side)
            
            # 运行策略
            if processed_ticks % 100 == 0:  # 每100个tick运行一次策略
                signals = await self.strategy_func(tick, self.engine, self.advanced_models.get(tick.symbol, {}))
                
                # 处理策略信号
                if signals:
                    await self._process_strategy_signals(signals, tick)
            
            processed_ticks += 1
            
            # 进度报告
            if processed_ticks % 10000 == 0:
                logger.info(f"已处理{processed_ticks}/{len(all_ticks)}个tick")
        
        # 获取结果
        performance_stats = self.engine.get_performance_stats()
        
        # 生成详细分析
        detailed_analysis = await self._generate_detailed_analysis()
        
        # 合并结果
        self.results = {
            'performance_stats': performance_stats,
            'detailed_analysis': detailed_analysis,
            'engine_state': {
                'final_balance': self.engine.balance,
                'positions': dict(self.engine.positions),
                'total_trades': len(self.engine.trades),
                'total_orders': len(self.engine.orders)
            }
        }
        
        # 保存结果
        if self.config.output_path:
            await self._save_results()
        
        # 生成图表
        if self.config.generate_plots:
            await self._generate_plots()
        
        logger.info("回测完成!")
        return self.results
    
    async def _process_strategy_signals(self, signals: List[Dict], current_tick: MarketDataTick):
        """处理策略信号"""
        for signal in signals:
            try:
                # 创建订单
                order = BacktestOrder(
                    order_id=f"order_{len(self.engine.orders)}_{int(current_tick.timestamp.timestamp())}",
                    symbol=signal['symbol'],
                    side=OrderSide.BUY if signal['side'].upper() == 'BUY' else OrderSide.SELL,
                    order_type=OrderType.LIMIT if signal.get('order_type', 'LIMIT').upper() == 'LIMIT' else OrderType.MARKET,
                    quantity=signal['quantity'],
                    price=signal.get('price'),
                    submit_time=current_tick.timestamp
                )
                
                # 应用高级模型调整
                if current_tick.symbol in self.advanced_models:
                    await self._apply_advanced_models(order, current_tick)
                
                # 提交订单
                order_id = self.engine.submit_order(order)
                
            except Exception as e:
                logger.error(f"处理策略信号失败: {e}")
    
    async def _apply_advanced_models(self, order: BacktestOrder, current_tick: MarketDataTick):
        """应用高级模型调整"""
        models = self.advanced_models[current_tick.symbol]
        
        # 计算高级延迟
        if self.config.enable_latency_simulation:
            network_condition = models['advanced_latency'].simulate_network_conditions()
            order.latency_ms = models['advanced_latency'].calculate_latency(
                market_load=network_condition,
                order_type=order.order_type.value.lower(),
                order_size_percentile=0.5  # 简化处理
            )
        
        # 估算排队位置
        if self.config.enable_queue_simulation and order.order_type == OrderType.LIMIT:
            orderbook_state = models['orderbook_state'].get_market_state()
            if orderbook_state:
                best_price = orderbook_state['best_bid'] if order.side == OrderSide.BUY else orderbook_state['best_ask']
                order.queue_position = models['queue_position'].estimate_queue_position(
                    order.price, best_price, 5  # 假设前面有5个订单
                )
        
        # 计算Maker/Taker状态
        if self.config.enable_maker_taker_logic:
            orderbook_state = models['orderbook_state'].get_market_state()
            if orderbook_state:
                maker_taker, fee_rate = models['maker_taker'].determine_maker_taker(
                    order.price or current_tick.price,
                    current_tick.price,
                    order.side.value,
                    orderbook_state
                )
                order.maker_taker = maker_taker
    
    async def _generate_detailed_analysis(self) -> Dict[str, Any]:
        """生成详细分析"""
        analysis = {}
        
        # 交易分析
        trades_df = pd.DataFrame([{
            'timestamp': trade.timestamp,
            'symbol': trade.symbol,
            'side': trade.side.value,
            'quantity': trade.quantity,
            'price': trade.price,
            'fees': trade.fees,
            'is_maker': trade.is_maker
        } for trade in self.engine.trades])
        
        if not trades_df.empty:
            analysis['trade_analysis'] = {
                'total_volume': trades_df['quantity'].sum(),
                'avg_trade_size': trades_df['quantity'].mean(),
                'maker_ratio': trades_df['is_maker'].mean(),
                'total_fees': trades_df['fees'].sum(),
                'trades_per_hour': len(trades_df) / ((self.config.end_date - self.config.start_date).total_seconds() / 3600)
            }
        
        # 订单分析
        orders_df = pd.DataFrame([{
            'order_id': order.order_id,
            'symbol': order.symbol,
            'side': order.side.value,
            'order_type': order.order_type.value,
            'quantity': order.quantity,
            'filled_quantity': order.filled_quantity,
            'status': order.status.value,
            'latency_ms': order.latency_ms,
            'slippage': order.slippage,
            'queue_position': order.queue_position,
            'maker_taker': order.maker_taker
        } for order in self.engine.orders.values()])
        
        if not orders_df.empty:
            analysis['order_analysis'] = {
                'total_orders': len(orders_df),
                'fill_rate': (orders_df['status'] == 'FILLED').mean(),
                'avg_latency_ms': orders_df['latency_ms'].mean(),
                'avg_slippage_bps': orders_df['slippage'].mean() * 10000,
                'avg_queue_position': orders_df['queue_position'].mean(),
                'order_type_distribution': orders_df['order_type'].value_counts().to_dict()
            }
        
        # 权益曲线分析
        if self.engine.equity_curve:
            equity_df = pd.DataFrame(self.engine.equity_curve, columns=['timestamp', 'equity'])
            equity_df['returns'] = equity_df['equity'].pct_change()
            
            analysis['equity_analysis'] = {
                'total_return': (equity_df['equity'].iloc[-1] - equity_df['equity'].iloc[0]) / equity_df['equity'].iloc[0],
                'volatility': equity_df['returns'].std() * np.sqrt(252 * 24),  # 年化波动率
                'sharpe_ratio': equity_df['returns'].mean() / equity_df['returns'].std() * np.sqrt(252 * 24),
                'max_drawdown': self._calculate_max_drawdown(equity_df['equity'].tolist()),
                'calmar_ratio': analysis.get('equity_analysis', {}).get('total_return', 0) / max(self._calculate_max_drawdown(equity_df['equity'].tolist()), 0.001)
            }
        
        return analysis
    
    def _calculate_max_drawdown(self, equity_values: List[float]) -> float:
        """计算最大回撤"""
        peak = equity_values[0]
        max_dd = 0.0
        
        for value in equity_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    async def _save_results(self):
        """保存回测结果"""
        output_path = Path(self.config.output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存主要结果
        with open(output_path / 'backtest_results.json', 'w') as f:
            # 序列化结果（处理datetime等不可序列化对象）
            serializable_results = self._make_serializable(self.results)
            json.dump(serializable_results, f, indent=2, default=str)
        
        # 保存详细交易记录
        if self.config.save_detailed_logs:
            trades_df = pd.DataFrame([{
                'timestamp': trade.timestamp,
                'trade_id': trade.trade_id,
                'order_id': trade.order_id,
                'symbol': trade.symbol,
                'side': trade.side.value,
                'quantity': trade.quantity,
                'price': trade.price,
                'fees': trade.fees,
                'is_maker': trade.is_maker
            } for trade in self.engine.trades])
            
            trades_df.to_csv(output_path / 'trades.csv', index=False)
            
            # 保存订单记录
            orders_df = pd.DataFrame([asdict(order) for order in self.engine.orders.values()])
            orders_df.to_csv(output_path / 'orders.csv', index=False)
        
        logger.info(f"结果已保存到: {output_path}")
    
    def _make_serializable(self, obj):
        """使对象可序列化"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj
    
    async def _generate_plots(self):
        """生成分析图表"""
        output_path = Path(self.config.output_path)
        
        # 设置图表样式
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 权益曲线
        if self.engine.equity_curve:
            equity_df = pd.DataFrame(self.engine.equity_curve, columns=['timestamp', 'equity'])
            axes[0, 0].plot(equity_df['timestamp'], equity_df['equity'])
            axes[0, 0].set_title('权益曲线')
            axes[0, 0].set_xlabel('时间')
            axes[0, 0].set_ylabel('权益')
            
            # 回撤图
            equity_df['peak'] = equity_df['equity'].expanding().max()
            equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak']
            axes[0, 1].fill_between(equity_df['timestamp'], equity_df['drawdown'], 0, alpha=0.3, color='red')
            axes[0, 1].set_title('回撤分析')
            axes[0, 1].set_xlabel('时间')
            axes[0, 1].set_ylabel('回撤')
        
        # 交易分析
        if self.engine.trades:
            trades_df = pd.DataFrame([{
                'timestamp': trade.timestamp,
                'pnl': trade.quantity * trade.price * (1 if trade.side == OrderSide.BUY else -1),
                'fees': trade.fees
            } for trade in self.engine.trades])
            
            # 累计PnL
            trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
            axes[1, 0].plot(trades_df['timestamp'], trades_df['cumulative_pnl'])
            axes[1, 0].set_title('累计损益')
            axes[1, 0].set_xlabel('时间')
            axes[1, 0].set_ylabel('累计PnL')
            
            # 手续费分析
            trades_df['cumulative_fees'] = trades_df['fees'].cumsum()
            axes[1, 1].plot(trades_df['timestamp'], trades_df['cumulative_fees'])
            axes[1, 1].set_title('累计手续费')
            axes[1, 1].set_xlabel('时间')
            axes[1, 1].set_ylabel('累计费用')
        
        plt.tight_layout()
        plt.savefig(output_path / 'backtest_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 延迟和滑点分析
        if self.engine.orders:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            orders_df = pd.DataFrame([{
                'latency_ms': order.latency_ms,
                'slippage_bps': order.slippage * 10000
            } for order in self.engine.orders.values()])
            
            # 延迟分布
            axes[0].hist(orders_df['latency_ms'], bins=30, alpha=0.7)
            axes[0].set_title('延迟分布')
            axes[0].set_xlabel('延迟 (ms)')
            axes[0].set_ylabel('频次')
            
            # 滑点分布
            axes[1].hist(orders_df['slippage_bps'], bins=30, alpha=0.7)
            axes[1].set_title('滑点分布')
            axes[1].set_xlabel('滑点 (bps)')
            axes[1].set_ylabel('频次')
            
            plt.tight_layout()
            plt.savefig(output_path / 'latency_slippage_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info("图表已生成")

# 示例策略函数
async def simple_momentum_strategy(tick: MarketDataTick, engine: HighFidelityBacktestEngine, 
                                 models: Dict[str, Any]) -> List[Dict]:
    """简单动量策略示例"""
    signals = []
    
    # 获取当前价格
    current_price = tick.price
    
    # 简单的动量信号（这里只是示例）
    if hasattr(engine, '_last_prices'):
        if len(engine._last_prices) >= 10:
            short_ma = np.mean(engine._last_prices[-5:])
            long_ma = np.mean(engine._last_prices[-10:])
            
            if short_ma > long_ma * 1.001:  # 上升趋势
                signals.append({
                    'symbol': tick.symbol,
                    'side': 'BUY',
                    'quantity': 0.1,
                    'order_type': 'LIMIT',
                    'price': current_price * 0.999  # 稍低于市价的限价单
                })
            elif short_ma < long_ma * 0.999:  # 下降趋势
                signals.append({
                    'symbol': tick.symbol,
                    'side': 'SELL',
                    'quantity': 0.1,
                    'order_type': 'LIMIT',
                    'price': current_price * 1.001  # 稍高于市价的限价单
                })
    else:
        engine._last_prices = []
    
    # 更新价格历史
    engine._last_prices.append(current_price)
    if len(engine._last_prices) > 20:
        engine._last_prices.pop(0)
    
    return signals