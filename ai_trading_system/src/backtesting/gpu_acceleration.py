"""
GPU加速模块
使用CuPy进行大规模并行计算，显著提升回测性能
"""

import numpy as np
import time
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import warnings

# 尝试导入CuPy，如果不可用则使用NumPy
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("✅ GPU (CuPy) 可用")
except ImportError:
    import numpy as cp
    GPU_AVAILABLE = False
    print("⚠️ GPU不可用，使用CPU (NumPy) 模式")

from .microsecond_precision import HighPrecisionTimestamp

@dataclass
class GPUMemoryInfo:
    """GPU内存信息"""
    total_memory: int
    free_memory: int
    used_memory: int
    
    @property
    def memory_usage_percent(self) -> float:
        return (self.used_memory / self.total_memory) * 100 if self.total_memory > 0 else 0

class GPUAcceleratedEngine:
    """GPU加速计算引擎"""
    
    def __init__(self, use_gpu: bool = True, memory_pool_size: Optional[int] = None):
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.device_name = "GPU" if self.use_gpu else "CPU"
        
        if self.use_gpu:
            # 设置GPU内存池
            if memory_pool_size:
                cp.cuda.MemoryPool().set_limit(size=memory_pool_size)
            
            # 获取GPU信息
            self.device_id = cp.cuda.Device().id
            self.device_info = cp.cuda.Device().attributes
            print(f"🚀 使用GPU设备: {self.device_id}")
        else:
            print("💻 使用CPU计算")
        
        # 性能统计
        self.computation_times = []
        self.memory_usage_history = []
        
    def get_memory_info(self) -> GPUMemoryInfo:
        """获取内存信息"""
        if self.use_gpu:
            mempool = cp.get_default_memory_pool()
            total = cp.cuda.Device().mem_info[1]  # 总内存
            free = cp.cuda.Device().mem_info[0]   # 可用内存
            used = total - free
            
            return GPUMemoryInfo(
                total_memory=total,
                free_memory=free,
                used_memory=used
            )
        else:
            # CPU模式返回虚拟信息
            return GPUMemoryInfo(
                total_memory=0,
                free_memory=0,
                used_memory=0
            )
    
    def batch_technical_indicators(self, prices: np.ndarray, volumes: np.ndarray, 
                                 batch_size: int = 10000) -> Dict[str, np.ndarray]:
        """批量计算技术指标 - GPU加速"""
        start_time = time.perf_counter()
        
        # 转换为GPU数组
        if self.use_gpu:
            prices_gpu = cp.asarray(prices)
            volumes_gpu = cp.asarray(volumes)
        else:
            prices_gpu = prices
            volumes_gpu = volumes
        
        n = len(prices_gpu)
        results = {}
        
        # 分批处理以管理内存
        for i in range(0, n, batch_size):
            end_idx = min(i + batch_size, n)
            batch_prices = prices_gpu[i:end_idx]
            batch_volumes = volumes_gpu[i:end_idx]
            
            # 计算各种技术指标
            batch_results = self._compute_batch_indicators(batch_prices, batch_volumes)
            
            # 合并结果
            for key, value in batch_results.items():
                if key not in results:
                    results[key] = []
                results[key].append(value)
        
        # 连接所有批次结果
        for key in results:
            if self.use_gpu:
                results[key] = cp.concatenate(results[key])
                # 转换回CPU
                results[key] = cp.asnumpy(results[key])
            else:
                results[key] = np.concatenate(results[key])
        
        computation_time = time.perf_counter() - start_time
        self.computation_times.append(computation_time)
        
        print(f"🔥 {self.device_name}批量技术指标计算完成: {n:,}个数据点, 耗时{computation_time:.3f}秒")
        
        return results
    
    def _compute_batch_indicators(self, prices: Any, volumes: Any) -> Dict[str, Any]:
        """计算单批次技术指标"""
        batch_size = len(prices)
        results = {}
        
        # 1. 移动平均线
        for window in [5, 10, 20, 50]:
            if batch_size >= window:
                if self.use_gpu:
                    # GPU版本的移动平均
                    sma = self._gpu_moving_average(prices, window)
                else:
                    # CPU版本
                    sma = self._cpu_moving_average(prices, window)
                results[f'SMA_{window}'] = sma
        
        # 2. 指数移动平均
        for alpha in [0.1, 0.2, 0.3]:
            ema = self._compute_ema(prices, alpha)
            results[f'EMA_{int(alpha*100)}'] = ema
        
        # 3. 相对强弱指标 (RSI)
        if batch_size >= 14:
            rsi = self._compute_rsi(prices, period=14)
            results['RSI_14'] = rsi
        
        # 4. 布林带
        if batch_size >= 20:
            bb_upper, bb_middle, bb_lower = self._compute_bollinger_bands(prices, period=20, std_dev=2)
            results['BB_Upper'] = bb_upper
            results['BB_Middle'] = bb_middle
            results['BB_Lower'] = bb_lower
        
        # 5. MACD
        if batch_size >= 26:
            macd, signal, histogram = self._compute_macd(prices)
            results['MACD'] = macd
            results['MACD_Signal'] = signal
            results['MACD_Histogram'] = histogram
        
        # 6. 成交量相关指标
        if len(volumes) == len(prices):
            # 成交量移动平均
            vol_sma = self._gpu_moving_average(volumes, 20) if self.use_gpu else self._cpu_moving_average(volumes, 20)
            results['Volume_SMA_20'] = vol_sma
            
            # 成交量加权平均价格 (VWAP)
            vwap = self._compute_vwap(prices, volumes)
            results['VWAP'] = vwap
        
        return results
    
    def _gpu_moving_average(self, data: Any, window: int) -> Any:
        """GPU版本移动平均"""
        if not self.use_gpu:
            return self._cpu_moving_average(data, window)
        
        # 使用CuPy的卷积计算移动平均
        kernel = cp.ones(window) / window
        # 处理边界
        padded_data = cp.pad(data, (window-1, 0), mode='edge')
        ma = cp.convolve(padded_data, kernel, mode='valid')
        return ma
    
    def _cpu_moving_average(self, data: np.ndarray, window: int) -> np.ndarray:
        """CPU版本移动平均"""
        kernel = np.ones(window) / window
        padded_data = np.pad(data, (window-1, 0), mode='edge')
        ma = np.convolve(padded_data, kernel, mode='valid')
        return ma
    
    def _compute_ema(self, prices: Any, alpha: float) -> Any:
        """计算指数移动平均"""
        if self.use_gpu:
            ema = cp.zeros_like(prices)
            ema[0] = prices[0]
            
            # GPU并行计算EMA
            for i in range(1, len(prices)):
                ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        else:
            ema = np.zeros_like(prices)
            ema[0] = prices[0]
            
            for i in range(1, len(prices)):
                ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    def _compute_rsi(self, prices: Any, period: int = 14) -> Any:
        """计算相对强弱指标"""
        if len(prices) < period + 1:
            return cp.full_like(prices, 50) if self.use_gpu else np.full_like(prices, 50)
        
        # 计算价格变化
        price_changes = cp.diff(prices) if self.use_gpu else np.diff(prices)
        
        # 分离涨跌
        gains = cp.where(price_changes > 0, price_changes, 0) if self.use_gpu else np.where(price_changes > 0, price_changes, 0)
        losses = cp.where(price_changes < 0, -price_changes, 0) if self.use_gpu else np.where(price_changes < 0, -price_changes, 0)
        
        # 计算平均涨跌
        avg_gains = self._gpu_moving_average(gains, period) if self.use_gpu else self._cpu_moving_average(gains, period)
        avg_losses = self._gpu_moving_average(losses, period) if self.use_gpu else self._cpu_moving_average(losses, period)
        
        # 计算RSI
        rs = avg_gains / (avg_losses + 1e-10)  # 避免除零
        rsi = 100 - (100 / (1 + rs))
        
        # 添加第一个值
        if self.use_gpu:
            rsi = cp.concatenate([cp.array([50]), rsi])
        else:
            rsi = np.concatenate([np.array([50]), rsi])
        
        return rsi
    
    def _compute_bollinger_bands(self, prices: Any, period: int = 20, std_dev: float = 2) -> Tuple[Any, Any, Any]:
        """计算布林带"""
        # 中轨：移动平均
        middle = self._gpu_moving_average(prices, period) if self.use_gpu else self._cpu_moving_average(prices, period)
        
        # 计算标准差
        if self.use_gpu:
            std = cp.zeros_like(prices)
            for i in range(period-1, len(prices)):
                window_data = prices[i-period+1:i+1]
                std[i] = cp.std(window_data)
        else:
            std = np.zeros_like(prices)
            for i in range(period-1, len(prices)):
                window_data = prices[i-period+1:i+1]
                std[i] = np.std(window_data)
        
        # 上下轨
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        
        return upper, middle, lower
    
    def _compute_macd(self, prices: Any, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[Any, Any, Any]:
        """计算MACD"""
        # 快速和慢速EMA
        ema_fast = self._compute_ema(prices, 2/(fast+1))
        ema_slow = self._compute_ema(prices, 2/(slow+1))
        
        # MACD线
        macd_line = ema_fast - ema_slow
        
        # 信号线
        signal_line = self._compute_ema(macd_line, 2/(signal+1))
        
        # 柱状图
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def _compute_vwap(self, prices: Any, volumes: Any) -> Any:
        """计算成交量加权平均价格"""
        if self.use_gpu:
            cumulative_pv = cp.cumsum(prices * volumes)
            cumulative_v = cp.cumsum(volumes)
            vwap = cumulative_pv / (cumulative_v + 1e-10)
        else:
            cumulative_pv = np.cumsum(prices * volumes)
            cumulative_v = np.cumsum(volumes)
            vwap = cumulative_pv / (cumulative_v + 1e-10)
        
        return vwap
    
    def batch_portfolio_simulation(self, returns: np.ndarray, weights: np.ndarray, 
                                 simulations: int = 10000) -> Dict[str, np.ndarray]:
        """批量投资组合蒙特卡洛模拟 - GPU加速"""
        start_time = time.perf_counter()
        
        if self.use_gpu:
            returns_gpu = cp.asarray(returns)
            weights_gpu = cp.asarray(weights)
            
            # 生成随机数 - GPU并行
            random_returns = cp.random.multivariate_normal(
                cp.mean(returns_gpu, axis=0),
                cp.cov(returns_gpu.T),
                size=simulations
            )
            
            # 计算投资组合收益 - GPU并行矩阵运算
            portfolio_returns = cp.dot(random_returns, weights_gpu)
            
            # 计算风险指标
            portfolio_mean = cp.mean(portfolio_returns)
            portfolio_std = cp.std(portfolio_returns)
            portfolio_var_95 = cp.percentile(portfolio_returns, 5)
            portfolio_var_99 = cp.percentile(portfolio_returns, 1)
            
            # 转换回CPU
            results = {
                'simulated_returns': cp.asnumpy(portfolio_returns),
                'mean_return': float(cp.asnumpy(portfolio_mean)),
                'volatility': float(cp.asnumpy(portfolio_std)),
                'var_95': float(cp.asnumpy(portfolio_var_95)),
                'var_99': float(cp.asnumpy(portfolio_var_99))
            }
        else:
            # CPU版本
            random_returns = np.random.multivariate_normal(
                np.mean(returns, axis=0),
                np.cov(returns.T),
                size=simulations
            )
            
            portfolio_returns = np.dot(random_returns, weights)
            
            results = {
                'simulated_returns': portfolio_returns,
                'mean_return': float(np.mean(portfolio_returns)),
                'volatility': float(np.std(portfolio_returns)),
                'var_95': float(np.percentile(portfolio_returns, 5)),
                'var_99': float(np.percentile(portfolio_returns, 1))
            }
        
        computation_time = time.perf_counter() - start_time
        self.computation_times.append(computation_time)
        
        print(f"🎲 {self.device_name}蒙特卡洛模拟完成: {simulations:,}次模拟, 耗时{computation_time:.3f}秒")
        
        return results
    
    def batch_option_pricing(self, spot_prices: np.ndarray, strike_prices: np.ndarray,
                           time_to_expiry: np.ndarray, risk_free_rate: float = 0.05,
                           volatility: float = 0.2) -> Dict[str, np.ndarray]:
        """批量期权定价 - 黑-舒尔斯模型GPU加速"""
        start_time = time.perf_counter()
        
        if self.use_gpu:
            S = cp.asarray(spot_prices)
            K = cp.asarray(strike_prices)
            T = cp.asarray(time_to_expiry)
            r = risk_free_rate
            sigma = volatility
            
            # 黑-舒尔斯公式 - GPU并行计算
            d1 = (cp.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*cp.sqrt(T))
            d2 = d1 - sigma*cp.sqrt(T)
            
            # 标准正态分布累积分布函数 (近似)
            def norm_cdf_gpu(x):
                return 0.5 * (1 + cp.erf(x / cp.sqrt(2)))
            
            # 看涨期权价格
            call_prices = S * norm_cdf_gpu(d1) - K * cp.exp(-r*T) * norm_cdf_gpu(d2)
            
            # 看跌期权价格 (看跌-看涨平价)
            put_prices = K * cp.exp(-r*T) * norm_cdf_gpu(-d2) - S * norm_cdf_gpu(-d1)
            
            # 希腊字母
            delta_call = norm_cdf_gpu(d1)
            delta_put = delta_call - 1
            
            gamma = cp.exp(-0.5*d1**2) / (S*sigma*cp.sqrt(2*cp.pi*T))
            
            # 转换回CPU
            results = {
                'call_prices': cp.asnumpy(call_prices),
                'put_prices': cp.asnumpy(put_prices),
                'delta_call': cp.asnumpy(delta_call),
                'delta_put': cp.asnumpy(delta_put),
                'gamma': cp.asnumpy(gamma)
            }
        else:
            # CPU版本
            from scipy.stats import norm
            
            S = spot_prices
            K = strike_prices
            T = time_to_expiry
            r = risk_free_rate
            sigma = volatility
            
            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            
            call_prices = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
            put_prices = K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            
            delta_call = norm.cdf(d1)
            delta_put = delta_call - 1
            gamma = norm.pdf(d1) / (S*sigma*np.sqrt(T))
            
            results = {
                'call_prices': call_prices,
                'put_prices': put_prices,
                'delta_call': delta_call,
                'delta_put': delta_put,
                'gamma': gamma
            }
        
        computation_time = time.perf_counter() - start_time
        self.computation_times.append(computation_time)
        
        print(f"📈 {self.device_name}期权定价完成: {len(spot_prices):,}个期权, 耗时{computation_time:.3f}秒")
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        if not self.computation_times:
            return {}
        
        stats = {
            'device': self.device_name,
            'gpu_available': GPU_AVAILABLE,
            'total_computations': len(self.computation_times),
            'total_time': sum(self.computation_times),
            'avg_time': np.mean(self.computation_times),
            'min_time': min(self.computation_times),
            'max_time': max(self.computation_times),
            'std_time': np.std(self.computation_times)
        }
        
        if self.use_gpu:
            memory_info = self.get_memory_info()
            stats.update({
                'gpu_memory_usage_percent': memory_info.memory_usage_percent,
                'gpu_total_memory_gb': memory_info.total_memory / (1024**3),
                'gpu_used_memory_gb': memory_info.used_memory / (1024**3)
            })
        
        return stats
    
    def cleanup(self):
        """清理GPU内存"""
        if self.use_gpu:
            cp.get_default_memory_pool().free_all_blocks()
            print("🧹 GPU内存已清理")

class MemoryPool:
    """内存池管理器"""
    
    def __init__(self, initial_size: int = 1024*1024*1024):  # 1GB
        self.initial_size = initial_size
        self.use_gpu = GPU_AVAILABLE
        
        if self.use_gpu:
            self.mempool = cp.get_default_memory_pool()
            self.mempool.set_limit(size=initial_size)
            print(f"🏊 GPU内存池初始化: {initial_size / (1024**3):.1f}GB")
        else:
            print("💾 使用系统内存管理")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """获取内存使用情况"""
        if self.use_gpu:
            used_bytes = self.mempool.used_bytes()
            total_bytes = self.mempool.total_bytes()
            
            return {
                'used_gb': used_bytes / (1024**3),
                'total_gb': total_bytes / (1024**3),
                'usage_percent': (used_bytes / total_bytes * 100) if total_bytes > 0 else 0,
                'n_free_blocks': self.mempool.n_free_blocks()
            }
        else:
            return {'message': 'CPU模式，无GPU内存信息'}
    
    def optimize_memory(self):
        """优化内存使用"""
        if self.use_gpu:
            self.mempool.free_all_blocks()
            print("🔧 内存池已优化")

def benchmark_gpu_acceleration():
    """GPU加速性能基准测试"""
    print("⚡ GPU加速性能基准测试")
    print("=" * 50)
    
    # 创建GPU引擎
    gpu_engine = GPUAcceleratedEngine(use_gpu=True)
    cpu_engine = GPUAcceleratedEngine(use_gpu=False)
    
    # 生成测试数据
    np.random.seed(42)
    n_samples = 100000
    n_assets = 10
    
    print(f"生成测试数据: {n_samples:,} 个价格点, {n_assets} 个资产")
    
    prices = np.random.lognormal(0, 0.02, (n_samples, n_assets)) * 45000
    volumes = np.random.exponential(100, (n_samples, n_assets))
    returns = np.diff(np.log(prices), axis=0)
    weights = np.random.random(n_assets)
    weights = weights / np.sum(weights)  # 标准化权重
    
    # 测试1: 技术指标计算
    print("\n📊 测试1: 批量技术指标计算")
    
    # GPU测试
    start_time = time.perf_counter()
    gpu_indicators = gpu_engine.batch_technical_indicators(prices[:, 0], volumes[:, 0])
    gpu_time_indicators = time.perf_counter() - start_time
    
    # CPU测试
    start_time = time.perf_counter()
    cpu_indicators = cpu_engine.batch_technical_indicators(prices[:, 0], volumes[:, 0])
    cpu_time_indicators = time.perf_counter() - start_time
    
    print(f"{gpu_engine.device_name}时间: {gpu_time_indicators:.3f}秒")
    print(f"{cpu_engine.device_name}时间: {cpu_time_indicators:.3f}秒")
    if gpu_time_indicators > 0:
        speedup_indicators = cpu_time_indicators / gpu_time_indicators
        print(f"加速比: {speedup_indicators:.2f}x")
    
    # 测试2: 蒙特卡洛模拟
    print("\n📊 测试2: 投资组合蒙特卡洛模拟")
    
    simulations = 50000
    
    # GPU测试
    gpu_mc_results = gpu_engine.batch_portfolio_simulation(returns, weights, simulations)
    
    # CPU测试
    cpu_mc_results = cpu_engine.batch_portfolio_simulation(returns, weights, simulations)
    
    # 测试3: 期权定价
    print("\n📊 测试3: 批量期权定价")
    
    n_options = 10000
    spot_prices = np.random.uniform(40000, 50000, n_options)
    strike_prices = np.random.uniform(35000, 55000, n_options)
    time_to_expiry = np.random.uniform(0.1, 2.0, n_options)
    
    # GPU测试
    gpu_options = gpu_engine.batch_option_pricing(spot_prices, strike_prices, time_to_expiry)
    
    # CPU测试
    cpu_options = cpu_engine.batch_option_pricing(spot_prices, strike_prices, time_to_expiry)
    
    # 内存使用情况
    print("\n📊 内存使用情况")
    gpu_stats = gpu_engine.get_performance_stats()
    cpu_stats = cpu_engine.get_performance_stats()
    
    print(f"GPU统计: {gpu_stats}")
    print(f"CPU统计: {cpu_stats}")
    
    # 清理
    gpu_engine.cleanup()
    
    print("\n✅ GPU加速基准测试完成!")
    
    return {
        'gpu_available': GPU_AVAILABLE,
        'indicators_speedup': cpu_time_indicators / gpu_time_indicators if gpu_time_indicators > 0 else 1,
        'gpu_stats': gpu_stats,
        'cpu_stats': cpu_stats,
        'technical_indicators_count': len(gpu_indicators),
        'monte_carlo_simulations': simulations,
        'options_priced': n_options
    }

if __name__ == "__main__":
    # 运行GPU加速基准测试
    results = benchmark_gpu_acceleration()
    
    print(f"\n🎯 GPU加速性能总结:")
    print(f"- GPU可用: {'是' if results['gpu_available'] else '否'}")
    print(f"- 技术指标加速比: {results['indicators_speedup']:.2f}x")
    print(f"- 计算的技术指标数: {results['technical_indicators_count']}")
    print(f"- 蒙特卡洛模拟次数: {results['monte_carlo_simulations']:,}")
    print(f"- 期权定价数量: {results['options_priced']:,}")
    print(f"- GPU总计算时间: {results['gpu_stats'].get('total_time', 0):.3f}秒")
    print(f"- CPU总计算时间: {results['cpu_stats'].get('total_time', 0):.3f}秒")