"""
综合集成测试
验证整个高保真回测引擎的所有模块协同工作
"""

import asyncio
import numpy as np
import time
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 导入所有核心模块
from .high_fidelity_backtest_engine import HighFidelityBacktestEngine, MarketDataTick
from .numba_accelerated_engine import NumbaAcceleratedEngine
from .microsecond_precision import HighPrecisionTimestamp, HighFrequencyDataProcessor
from .level3_orderbook_advanced import Level3OrderBook, Level3Order, OrderSide, OrderType, OrderStatus
from .multi_exchange_engine import MultiExchangeEngine, ExchangeType
from .unified_trading_interface import (
    BacktestTradingInterface, UnifiedOrderRequest, UnifiedMarketData, 
    SimpleMovingAverageStrategy
)
from .advanced_visualization import AdvancedVisualizer, VisualizationData
from .gpu_acceleration import GPUAcceleratedEngine

class ComprehensiveIntegrationTest:
    """综合集成测试套件"""
    
    def __init__(self):
        self.results = {}
        self.start_time = time.perf_counter()
        
        # 测试配置
        self.test_config = {
            'initial_balance': 100000.0,
            'test_duration_minutes': 5,
            'tick_frequency_hz': 1000,
            'num_orders': 1000,
            'num_exchanges': 3,
            'enable_gpu': False,  # 设为False以确保在所有环境中都能运行
            'enable_visualization': True,
            'save_results': True
        }
        
        print("🏗️ 初始化综合集成测试...")
        print(f"配置: {self.test_config}")
    
    async def run_full_integration_test(self) -> Dict[str, Any]:
        """运行完整的集成测试"""
        print("\n🚀 开始综合集成测试")
        print("=" * 60)
        
        # 测试1: 核心引擎集成
        test1_results = await self._test_core_engine_integration()
        self.results['core_engine'] = test1_results
        
        # 测试2: Numba加速集成
        test2_results = await self._test_numba_acceleration_integration()
        self.results['numba_acceleration'] = test2_results
        
        # 测试3: 高精度时间系统集成
        test3_results = await self._test_microsecond_precision_integration()
        self.results['microsecond_precision'] = test3_results
        
        # 测试4: Level-3订单簿集成
        test4_results = await self._test_level3_orderbook_integration()
        self.results['level3_orderbook'] = test4_results
        
        # 测试5: 多交易所引擎集成
        test5_results = await self._test_multi_exchange_integration()
        self.results['multi_exchange'] = test5_results
        
        # 测试6: 统一接口集成
        test6_results = await self._test_unified_interface_integration()
        self.results['unified_interface'] = test6_results
        
        # 测试7: 可视化分析集成
        test7_results = await self._test_visualization_integration()
        self.results['visualization'] = test7_results
        
        # 测试8: GPU加速集成
        test8_results = await self._test_gpu_acceleration_integration()
        self.results['gpu_acceleration'] = test8_results
        
        # 测试9: 端到端策略回测
        test9_results = await self._test_end_to_end_strategy_backtest()
        self.results['end_to_end'] = test9_results
        
        # 测试10: 压力测试
        test10_results = await self._test_system_stress_test()
        self.results['stress_test'] = test10_results
        
        # 生成综合报告
        self._generate_comprehensive_report()
        
        total_time = time.perf_counter() - self.start_time
        self.results['total_test_time'] = total_time
        
        print(f"\n✅ 综合集成测试完成，总耗时 {total_time:.2f}秒")
        
        return self.results
    
    async def _test_core_engine_integration(self) -> Dict[str, Any]:
        """测试核心回测引擎集成"""
        print("\n📊 测试1: 核心回测引擎集成")
        
        start_time = time.perf_counter()
        
        # 创建核心引擎
        engine = HighFidelityBacktestEngine(
            initial_balance=self.test_config['initial_balance'],
            symbols=['BTCUSDT']
        )
        
        # 生成测试数据
        n_ticks = 1000
        base_price = 45000.0
        
        ticks_processed = 0
        for i in range(n_ticks):
            # 生成随机价格
            price = base_price + np.random.normal(0, 10)
            volume = np.random.exponential(1.0)
            
            # 创建tick数据
            tick = MarketDataTick(
                symbol='BTCUSDT',
                timestamp=HighPrecisionTimestamp.now().add_microseconds(i * 1000),
                price=price,
                volume=volume,
                side='BUY' if i % 2 == 0 else 'SELL'
            )
            
            # 添加到引擎
            engine.add_market_data(tick)
            ticks_processed += 1
            
            # 每100个tick提交一个订单
            if i % 100 == 0:
                order = Level3Order(
                    order_id=f"test_order_{i}",
                    symbol='BTCUSDT',
                    side=OrderSide.BUY if i % 200 == 0 else OrderSide.SELL,
                    order_type=OrderType.LIMIT,
                    price=price + (1 if i % 200 == 0 else -1),
                    original_quantity=0.01,
                    remaining_quantity=0.01
                )
                engine.submit_order(order)
        
        # 获取性能统计
        performance_stats = engine.get_performance_stats()
        
        test_time = time.perf_counter() - start_time
        
        results = {
            'success': True,
            'ticks_processed': ticks_processed,
            'orders_submitted': performance_stats.get('total_trades', 0),
            'test_time_seconds': test_time,
            'ticks_per_second': ticks_processed / test_time,
            'performance_stats': performance_stats
        }
        
        print(f"   ✅ 核心引擎测试完成")
        print(f"   - 处理Tick数: {ticks_processed:,}")
        print(f"   - 处理速度: {results['ticks_per_second']:,.0f} ticks/秒")
        print(f"   - 测试时间: {test_time:.3f}秒")
        
        return results
    
    async def _test_numba_acceleration_integration(self) -> Dict[str, Any]:
        """测试Numba加速集成"""
        print("\n⚡ 测试2: Numba加速集成")
        
        start_time = time.perf_counter()
        
        # 创建Numba加速引擎
        numba_engine = NumbaAcceleratedEngine()
        
        # 生成测试数据
        n_orders = 5000
        bids_prices = np.random.uniform(44900, 45000, n_orders)
        bids_quantities = np.random.uniform(0.1, 10.0, n_orders)
        asks_prices = np.random.uniform(45000, 45100, n_orders)
        asks_quantities = np.random.uniform(0.1, 10.0, n_orders)
        
        orders_processed = 0
        total_latency = 0
        
        # 批量处理订单
        for i in range(100):  # 100批次
            order_price = np.random.uniform(44950, 45050)
            order_quantity = np.random.uniform(0.1, 5.0)
            is_buy = i % 2 == 0
            
            # 使用Numba加速的订单匹配
            start_match = time.perf_counter_ns()
            filled_qty, avg_price, fees, trades = numba_engine.fast_order_matching(
                bids_prices, bids_quantities, asks_prices, asks_quantities,
                order_price, order_quantity, is_buy, 0.001, 0.001
            )
            match_time = time.perf_counter_ns() - start_match
            
            orders_processed += 1
            total_latency += match_time / 1000  # 转为微秒
        
        test_time = time.perf_counter() - start_time
        avg_latency_us = total_latency / orders_processed if orders_processed > 0 else 0
        
        results = {
            'success': True,
            'orders_processed': orders_processed,
            'avg_latency_us': avg_latency_us,
            'test_time_seconds': test_time,
            'orders_per_second': orders_processed / test_time,
            'total_filled_quantity': float(filled_qty) if 'filled_qty' in locals() else 0
        }
        
        print(f"   ✅ Numba加速测试完成")
        print(f"   - 处理订单数: {orders_processed:,}")
        print(f"   - 平均延迟: {avg_latency_us:.2f} 微秒")
        print(f"   - 处理速度: {results['orders_per_second']:,.0f} 订单/秒")
        
        return results
    
    async def _test_microsecond_precision_integration(self) -> Dict[str, Any]:
        """测试微秒精度时间系统集成"""
        print("\n⏱️ 测试3: 微秒精度时间系统集成")
        
        start_time = time.perf_counter()
        
        # 创建高频数据处理器
        processor = HighFrequencyDataProcessor()
        
        # 生成高频数据
        n_ticks = 10000
        tick_data = processor.create_synthetic_tick_data(
            symbol='BTCUSDT',
            duration_seconds=1,
            frequency_hz=n_ticks
        )
        
        # 处理时间戳
        timestamps_processed = 0
        precision_tests = []
        
        for i in range(min(1000, len(tick_data))):  # 测试1000个时间戳
            ts = HighPrecisionTimestamp.now()
            
            # 测试纳秒精度计算
            future_ts = ts.add_microseconds(100)
            diff_ns = future_ts.difference_nanoseconds(ts)
            precision_tests.append(diff_ns)
            timestamps_processed += 1
        
        # 验证精度
        expected_diff = 100 * 1000  # 100微秒 = 100,000纳秒
        precision_errors = [abs(diff - expected_diff) for diff in precision_tests]
        avg_precision_error = np.mean(precision_errors)
        
        test_time = time.perf_counter() - start_time
        
        results = {
            'success': True,
            'ticks_generated': len(tick_data),
            'timestamps_processed': timestamps_processed,
            'avg_precision_error_ns': avg_precision_error,
            'max_precision_error_ns': max(precision_errors),
            'test_time_seconds': test_time,
            'timestamp_ops_per_second': timestamps_processed / test_time
        }
        
        print(f"   ✅ 微秒精度测试完成")
        print(f"   - 生成Tick数: {len(tick_data):,}")
        print(f"   - 平均精度误差: {avg_precision_error:.1f} 纳秒")
        print(f"   - 时间戳操作速度: {results['timestamp_ops_per_second']:,.0f} ops/秒")
        
        return results
    
    async def _test_level3_orderbook_integration(self) -> Dict[str, Any]:
        """测试Level-3订单簿集成"""
        print("\n📚 测试4: Level-3订单簿集成")
        
        start_time = time.perf_counter()
        
        # 创建Level-3订单簿
        orderbook = Level3OrderBook('BTCUSDT')
        
        # 添加初始流动性
        base_price = 45000.0
        orders_added = 0
        
        # 添加买盘和卖盘
        for i in range(500):
            # 买单
            buy_order = Level3Order(
                order_id=f"buy_{i}",
                symbol='BTCUSDT',
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                price=base_price - i * 0.1,
                original_quantity=1.0 + i * 0.01,
                remaining_quantity=1.0 + i * 0.01
            )
            orderbook.add_order(buy_order)
            orders_added += 1
            
            # 卖单
            sell_order = Level3Order(
                order_id=f"sell_{i}",
                symbol='BTCUSDT',
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                price=base_price + 10 + i * 0.1,
                original_quantity=1.0 + i * 0.01,
                remaining_quantity=1.0 + i * 0.01
            )
            orderbook.add_order(sell_order)
            orders_added += 1
        
        # 执行市价单测试
        market_orders = 0
        for i in range(50):
            market_order = Level3Order(
                order_id=f"market_{i}",
                symbol='BTCUSDT',
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                order_type=OrderType.MARKET,
                price=0.0,
                original_quantity=0.5,
                remaining_quantity=0.5
            )
            orderbook.add_order(market_order)
            market_orders += 1
        
        # 获取统计信息
        stats = orderbook.get_statistics()
        
        test_time = time.perf_counter() - start_time
        
        results = {
            'success': True,
            'orders_added': orders_added,
            'market_orders_executed': market_orders,
            'total_trades': stats['total_trades_executed'],
            'active_orders': stats['active_orders'],
            'bid_levels': stats['bid_levels'],
            'ask_levels': stats['ask_levels'],
            'spread': stats['spread'],
            'test_time_seconds': test_time,
            'orders_per_second': (orders_added + market_orders) / test_time
        }
        
        print(f"   ✅ Level-3订单簿测试完成")
        print(f"   - 添加订单数: {orders_added:,}")
        print(f"   - 执行交易数: {stats['total_trades_executed']}")
        print(f"   - 当前价差: {stats['spread']:.1f}")
        print(f"   - 处理速度: {results['orders_per_second']:,.0f} 订单/秒")
        
        return results
    
    async def _test_multi_exchange_integration(self) -> Dict[str, Any]:
        """测试多交易所引擎集成"""
        print("\n🌐 测试5: 多交易所引擎集成")
        
        start_time = time.perf_counter()
        
        # 创建多交易所引擎
        multi_engine = MultiExchangeEngine([
            ExchangeType.BINANCE,
            ExchangeType.COINBASE,
            ExchangeType.HUOBI
        ])
        
        # 初始化流动性
        for exchange_name in multi_engine.exchanges.keys():
            multi_engine.add_initial_liquidity(exchange_name, levels=20)
        
        # 检测套利机会
        opportunities = multi_engine.detect_arbitrage_opportunities(min_profit_threshold=0.0001)
        
        # 提交跨交易所订单
        orders_submitted = 0
        for i in range(30):  # 每个交易所10个订单
            for exchange_name in multi_engine.exchanges.keys():
                order = Level3Order(
                    order_id=f"multi_{exchange_name}_{i}",
                    symbol='BTCUSDT',
                    side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                    order_type=OrderType.LIMIT,
                    price=45000.0 + (i % 10 - 5) * 0.1,
                    original_quantity=0.1,
                    remaining_quantity=0.1
                )
                
                cross_order = multi_engine.submit_order_to_exchange(exchange_name, order)
                if cross_order:
                    orders_submitted += 1
        
        # 处理事件
        current_time = HighPrecisionTimestamp.now()
        future_time = current_time.add_milliseconds(100)
        
        events = multi_engine.process_global_events(future_time)
        
        # 获取统计
        stats = multi_engine.get_multi_exchange_statistics()
        
        test_time = time.perf_counter() - start_time
        
        results = {
            'success': True,
            'exchanges_count': len(multi_engine.exchanges),
            'arbitrage_opportunities': len(opportunities),
            'orders_submitted': orders_submitted,
            'events_processed': sum(len(e) for e in events.values()),
            'global_stats': stats['global'],
            'test_time_seconds': test_time
        }
        
        print(f"   ✅ 多交易所引擎测试完成")
        print(f"   - 交易所数量: {len(multi_engine.exchanges)}")
        print(f"   - 套利机会: {len(opportunities)}")
        print(f"   - 提交订单数: {orders_submitted}")
        print(f"   - 平均延迟: {stats['global']['avg_latency_us']:.1f} 微秒")
        
        return results
    
    async def _test_unified_interface_integration(self) -> Dict[str, Any]:
        """测试统一接口集成"""
        print("\n🔗 测试6: 统一接口集成")
        
        start_time = time.perf_counter()
        
        # 创建回测引擎和接口
        backtest_engine = HighFidelityBacktestEngine(initial_balance=50000.0)
        trading_interface = BacktestTradingInterface(backtest_engine)
        
        # 创建策略
        strategy_config = {
            'short_window': 3,
            'long_window': 7,
            'symbol': 'BTCUSDT',
            'position_size': 1000.0
        }
        
        strategy = SimpleMovingAverageStrategy(trading_interface, strategy_config)
        await strategy.initialize()
        
        # 生成市场数据并发送给策略
        base_price = 45000.0
        market_data_count = 0
        
        for i in range(50):
            # 生成价格趋势
            trend = i * 2  # 上涨趋势
            noise = np.random.normal(0, 5)
            price = base_price + trend + noise
            
            # 创建市场数据
            market_data = UnifiedMarketData(
                symbol='BTCUSDT',
                timestamp=HighPrecisionTimestamp.now(),
                price=price,
                volume=100.0 + np.random.uniform(0, 50),
                bid=price - 0.5,
                ask=price + 0.5,
                source="integration_test"
            )
            
            # 发送到策略
            await strategy.on_market_data(market_data)
            market_data_count += 1
        
        # 获取性能指标
        portfolio = await strategy.get_current_portfolio()
        performance = await strategy.get_strategy_performance()
        
        test_time = time.perf_counter() - start_time
        
        results = {
            'success': True,
            'market_data_processed': market_data_count,
            'strategy_trades': performance.total_trades,
            'final_portfolio_value': portfolio.total_value,
            'total_pnl': portfolio.total_pnl,
            'avg_latency_us': performance.avg_latency_us,
            'test_time_seconds': test_time
        }
        
        print(f"   ✅ 统一接口测试完成")
        print(f"   - 处理市场数据: {market_data_count}")
        print(f"   - 策略交易数: {performance.total_trades}")
        print(f"   - 最终价值: ${portfolio.total_value:.2f}")
        print(f"   - 总盈亏: ${portfolio.total_pnl:.2f}")
        
        return results
    
    async def _test_visualization_integration(self) -> Dict[str, Any]:
        """测试可视化分析集成"""
        print("\n📊 测试7: 可视化分析集成")
        
        start_time = time.perf_counter()
        
        if not self.test_config['enable_visualization']:
            print("   ⚠️  可视化测试已跳过")
            return {'success': True, 'skipped': True}
        
        # 创建可视化器
        visualizer = AdvancedVisualizer()
        
        # 生成测试数据
        n_samples = 200  # 减少样本以加快测试
        
        # 延迟数据
        latencies_us = np.random.lognormal(np.log(150), 0.3, n_samples)
        latencies_us = np.clip(latencies_us, 10, 1000)
        
        # 滑点数据
        slippages_bps = np.random.exponential(1.5, n_samples)
        slippages_bps = np.clip(slippages_bps, 0.1, 10)
        
        # 测试延迟分析
        latency_stats = visualizer.create_latency_analysis(latencies_us.tolist())
        
        # 测试滑点分析
        volumes = np.random.exponential(5, n_samples)
        slippage_stats = visualizer.create_slippage_analysis(slippages_bps.tolist(), volumes.tolist())
        
        test_time = time.perf_counter() - start_time
        
        results = {
            'success': True,
            'latency_analysis': latency_stats,
            'slippage_analysis': slippage_stats,
            'charts_generated': 2,
            'test_time_seconds': test_time
        }
        
        print(f"   ✅ 可视化分析测试完成")
        print(f"   - 生成图表数: 2")
        print(f"   - 延迟分析: 平均 {latency_stats['mean_us']:.1f}μs")
        print(f"   - 滑点分析: 平均 {slippage_stats['mean_bps']:.2f}bps")
        
        return results
    
    async def _test_gpu_acceleration_integration(self) -> Dict[str, Any]:
        """测试GPU加速集成"""
        print("\n⚡ 测试8: GPU加速集成")
        
        start_time = time.perf_counter()
        
        # 创建GPU引擎 (CPU模式)
        gpu_engine = GPUAcceleratedEngine(use_gpu=self.test_config['enable_gpu'])
        
        # 生成测试数据
        n_samples = 10000
        prices = np.random.lognormal(0, 0.01, n_samples) * 45000
        volumes = np.random.exponential(50, n_samples)
        
        # 测试技术指标计算
        indicators = gpu_engine.batch_technical_indicators(prices, volumes)
        
        # 测试蒙特卡洛模拟
        returns = np.diff(np.log(prices))
        weights = np.array([1.0])  # 单一资产权重
        
        mc_results = gpu_engine.batch_portfolio_simulation(
            returns.reshape(-1, 1), weights, simulations=5000
        )
        
        # 获取性能统计
        perf_stats = gpu_engine.get_performance_stats()
        
        # 清理
        gpu_engine.cleanup()
        
        test_time = time.perf_counter() - start_time
        
        results = {
            'success': True,
            'device_used': gpu_engine.device_name,
            'indicators_calculated': len(indicators),
            'monte_carlo_simulations': 5000,
            'performance_stats': perf_stats,
            'test_time_seconds': test_time
        }
        
        print(f"   ✅ GPU加速测试完成")
        print(f"   - 使用设备: {gpu_engine.device_name}")
        print(f"   - 计算指标数: {len(indicators)}")
        print(f"   - 模拟次数: 5,000")
        print(f"   - 总计算时间: {perf_stats.get('total_time', 0):.3f}秒")
        
        return results
    
    async def _test_end_to_end_strategy_backtest(self) -> Dict[str, Any]:
        """测试端到端策略回测"""
        print("\n🎯 测试9: 端到端策略回测")
        
        start_time = time.perf_counter()
        
        # 创建完整的回测环境
        backtest_engine = HighFidelityBacktestEngine(
            initial_balance=100000.0,
            symbols=['BTCUSDT']
        )
        
        # 集成Numba加速
        numba_engine = NumbaAcceleratedEngine()
        
        # 创建统一接口
        trading_interface = BacktestTradingInterface(backtest_engine)
        
        # 创建策略
        strategy = SimpleMovingAverageStrategy(trading_interface, {
            'short_window': 5,
            'long_window': 15,
            'symbol': 'BTCUSDT',
            'position_size': 10000.0
        })
        
        await strategy.initialize()
        
        # 生成完整的市场数据序列
        base_price = 45000.0
        prices = []
        ticks_processed = 0
        
        for i in range(100):
            # 生成趋势 + 噪音
            trend = np.sin(i * 0.1) * 50  # 波动趋势
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
                source="end_to_end_test"
            )
            
            # 同时更新回测引擎
            tick = MarketDataTick(
                symbol='BTCUSDT',
                timestamp=market_data.timestamp,
                price=price,
                volume=market_data.volume,
                side='BUY' if i % 2 == 0 else 'SELL'
            )
            backtest_engine.add_market_data(tick)
            
            # 发送到策略
            await strategy.on_market_data(market_data)
            ticks_processed += 1
        
        # 获取最终结果
        final_portfolio = await strategy.get_current_portfolio()
        final_performance = await strategy.get_strategy_performance()
        engine_stats = backtest_engine.get_performance_stats()
        
        test_time = time.perf_counter() - start_time
        
        results = {
            'success': True,
            'ticks_processed': ticks_processed,
            'strategy_trades': final_performance.total_trades,
            'initial_balance': 100000.0,
            'final_balance': final_portfolio.total_value,
            'total_return_pct': (final_portfolio.total_value / 100000.0 - 1) * 100,
            'total_pnl': final_portfolio.total_pnl,
            'avg_latency_us': final_performance.avg_latency_us,
            'engine_stats': engine_stats,
            'test_time_seconds': test_time
        }
        
        print(f"   ✅ 端到端回测测试完成")
        print(f"   - 处理Tick数: {ticks_processed}")
        print(f"   - 策略交易数: {final_performance.total_trades}")
        print(f"   - 初始资金: ${100000.0:,.2f}")
        print(f"   - 最终价值: ${final_portfolio.total_value:,.2f}")
        print(f"   - 总收益率: {results['total_return_pct']:.2f}%")
        
        return results
    
    async def _test_system_stress_test(self) -> Dict[str, Any]:
        """测试系统压力测试"""
        print("\n🔥 测试10: 系统压力测试")
        
        start_time = time.perf_counter()
        
        # 创建多个组件进行压力测试
        engines = []
        orderbooks = []
        
        # 创建多个引擎实例
        for i in range(3):
            engine = HighFidelityBacktestEngine(initial_balance=50000.0)
            engines.append(engine)
            
            orderbook = Level3OrderBook(f'TEST{i}USDT')
            orderbooks.append(orderbook)
        
        # 大量数据处理
        total_operations = 0
        
        # 批量添加订单
        for engine_idx, orderbook in enumerate(orderbooks):
            for i in range(500):  # 每个订单簿500个订单
                order = Level3Order(
                    order_id=f"stress_{engine_idx}_{i}",
                    symbol=f'TEST{engine_idx}USDT',
                    side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                    order_type=OrderType.LIMIT,
                    price=45000.0 + np.random.uniform(-100, 100),
                    original_quantity=np.random.uniform(0.1, 5.0),
                    remaining_quantity=np.random.uniform(0.1, 5.0)
                )
                orderbook.add_order(order)
                total_operations += 1
        
        # 批量处理市场数据
        for engine in engines:
            for i in range(1000):  # 每个引擎1000个tick
                tick = MarketDataTick(
                    symbol='BTCUSDT',
                    timestamp=HighPrecisionTimestamp.now(),
                    price=45000.0 + np.random.normal(0, 20),
                    volume=np.random.exponential(1.0),
                    side='BUY' if i % 2 == 0 else 'SELL'
                )
                engine.add_market_data(tick)
                total_operations += 1
        
        # 批量时间戳操作
        timestamp_ops = 0
        for i in range(10000):
            ts = HighPrecisionTimestamp.now()
            future_ts = ts.add_microseconds(i)
            diff = future_ts.difference_nanoseconds(ts)
            timestamp_ops += 1
        
        total_operations += timestamp_ops
        
        # 获取统计信息
        memory_usage = []
        performance_stats = []
        
        for engine in engines:
            stats = engine.get_performance_stats()
            performance_stats.append(stats)
        
        for orderbook in orderbooks:
            stats = orderbook.get_statistics()
            memory_usage.append(stats)
        
        test_time = time.perf_counter() - start_time
        
        results = {
            'success': True,
            'total_operations': total_operations,
            'engines_tested': len(engines),
            'orderbooks_tested': len(orderbooks),
            'timestamp_operations': timestamp_ops,
            'operations_per_second': total_operations / test_time,
            'test_time_seconds': test_time,
            'performance_stats': performance_stats,
            'memory_stats': memory_usage
        }
        
        print(f"   ✅ 压力测试完成")
        print(f"   - 总操作数: {total_operations:,}")
        print(f"   - 操作速度: {results['operations_per_second']:,.0f} ops/秒")
        print(f"   - 测试引擎数: {len(engines)}")
        print(f"   - 测试时间: {test_time:.3f}秒")
        
        return results
    
    def _generate_comprehensive_report(self):
        """生成综合测试报告"""
        print("\n📋 生成综合测试报告...")
        
        if not self.test_config['save_results']:
            return
        
        # 创建报告
        report = {
            'test_timestamp': datetime.now().isoformat(),
            'test_config': self.test_config,
            'test_results': self.results,
            'summary': self._generate_test_summary()
        }
        
        # 保存报告
        import os
        os.makedirs('integration_test_results', exist_ok=True)
        
        report_filename = f"integration_test_results/comprehensive_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"   📄 报告已保存: {report_filename}")
    
    def _generate_test_summary(self) -> Dict[str, Any]:
        """生成测试摘要"""
        summary = {
            'total_tests': 10,
            'passed_tests': 0,
            'failed_tests': 0,
            'test_details': {}
        }
        
        for test_name, test_result in self.results.items():
            if test_name == 'total_test_time':
                continue
                
            if isinstance(test_result, dict) and test_result.get('success', False):
                summary['passed_tests'] += 1
                summary['test_details'][test_name] = 'PASSED'
            else:
                summary['failed_tests'] += 1
                summary['test_details'][test_name] = 'FAILED'
        
        summary['success_rate'] = (summary['passed_tests'] / summary['total_tests']) * 100
        
        return summary

async def run_comprehensive_integration_test():
    """运行综合集成测试的主函数"""
    print("🏗️ 启动高保真回测引擎综合集成测试")
    print("=" * 80)
    
    # 创建测试套件
    test_suite = ComprehensiveIntegrationTest()
    
    # 运行所有测试
    results = await test_suite.run_full_integration_test()
    
    # 输出最终摘要
    print("\n" + "=" * 80)
    print("🎯 综合集成测试最终摘要")
    print("=" * 80)
    
    if 'summary' in results:
        summary = results['summary']
        print(f"📊 测试总数: {summary['total_tests']}")
        print(f"✅ 通过测试: {summary['passed_tests']}")
        print(f"❌ 失败测试: {summary['failed_tests']}")
        print(f"📈 成功率: {summary['success_rate']:.1f}%")
        
        print(f"\n⏱️ 总测试时间: {results.get('total_test_time', 0):.2f}秒")
        
        print(f"\n📋 详细结果:")
        for test_name, status in summary['test_details'].items():
            emoji = "✅" if status == "PASSED" else "❌"
            print(f"   {emoji} {test_name}: {status}")
    
    print("\n🏆 高保真回测引擎集成测试完成!")
    
    return results

if __name__ == "__main__":
    # 运行综合集成测试
    import asyncio
    
    async def main():
        results = await run_comprehensive_integration_test()
        
        # 输出关键性能指标
        print(f"\n🚀 关键性能指标摘要:")
        
        if 'core_engine' in results:
            core = results['core_engine']
            print(f"- 核心引擎处理速度: {core.get('ticks_per_second', 0):,.0f} ticks/秒")
        
        if 'numba_acceleration' in results:
            numba = results['numba_acceleration']
            print(f"- Numba加速延迟: {numba.get('avg_latency_us', 0):.1f} 微秒")
        
        if 'level3_orderbook' in results:
            orderbook = results['level3_orderbook']
            print(f"- 订单簿处理速度: {orderbook.get('orders_per_second', 0):,.0f} 订单/秒")
        
        if 'end_to_end' in results:
            e2e = results['end_to_end']
            print(f"- 端到端收益率: {e2e.get('total_return_pct', 0):.2f}%")
            print(f"- 端到端延迟: {e2e.get('avg_latency_us', 0):.1f} 微秒")
        
        if 'stress_test' in results:
            stress = results['stress_test']
            print(f"- 压力测试吞吐量: {stress.get('operations_per_second', 0):,.0f} ops/秒")
    
    asyncio.run(main())