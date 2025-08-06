"""
高级可视化分析模块
提供延迟分布、滑点分析、交易性能等专业图表
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from plotly.offline import plot
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import time
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from .microsecond_precision import HighPrecisionTimestamp
from .unified_trading_interface import UnifiedOrderResponse, UnifiedPortfolioInfo, UnifiedPerformanceMetrics

# 设置绘图样式
plt.style.use('dark_background')
sns.set_palette("husl")

@dataclass
class VisualizationData:
    """可视化数据容器"""
    timestamps: List[HighPrecisionTimestamp]
    prices: List[float]
    volumes: List[float]
    orders: List[UnifiedOrderResponse]
    portfolio_values: List[float]
    latencies_us: List[float]
    slippages_bps: List[float]
    spreads_bps: List[float]
    
    # 交易统计
    trade_returns: List[float] = None
    drawdowns: List[float] = None
    rolling_sharpe: List[float] = None
    
    def __post_init__(self):
        if self.trade_returns is None:
            self.trade_returns = []
        if self.drawdowns is None:
            self.drawdowns = []
        if self.rolling_sharpe is None:
            self.rolling_sharpe = []

class AdvancedVisualizer:
    """高级可视化分析器"""
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        self.figsize = figsize
        self.color_palette = {
            'primary': '#00D4FF',
            'secondary': '#FF6B6B', 
            'success': '#4ECDC4',
            'warning': '#FFE66D',
            'danger': '#FF6B9D',
            'info': '#95E1D3',
            'dark': '#2C3E50',
            'light': '#ECF0F1'
        }
        
        # 配置seaborn
        sns.set_style("darkgrid")
        plt.rcParams.update({
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.titlesize': 14
        })
    
    def create_latency_analysis(self, latencies_us: List[float], save_path: str = None) -> Dict[str, Any]:
        """创建延迟分析图表"""
        print("📊 生成延迟分析图表...")
        
        if not latencies_us:
            print("警告: 没有延迟数据")
            return {}
        
        latencies_array = np.array(latencies_us)
        
        # 创建子图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('🚀 延迟性能分析 (Latency Performance Analysis)', fontsize=16, color='white')
        
        # 1. 延迟分布直方图
        ax1 = axes[0, 0]
        n, bins, patches = ax1.hist(latencies_array, bins=50, alpha=0.7, color=self.color_palette['primary'], edgecolor='white')
        ax1.axvline(np.mean(latencies_array), color=self.color_palette['warning'], linestyle='--', linewidth=2, label=f'Mean: {np.mean(latencies_array):.1f}μs')
        ax1.axvline(np.percentile(latencies_array, 95), color=self.color_palette['danger'], linestyle='--', linewidth=2, label=f'P95: {np.percentile(latencies_array, 95):.1f}μs')
        ax1.set_xlabel('延迟 (微秒)')
        ax1.set_ylabel('频次')
        ax1.set_title('延迟分布直方图')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 延迟箱型图
        ax2 = axes[0, 1]
        box_plot = ax2.boxplot(latencies_array, patch_artist=True, labels=['延迟'])
        box_plot['boxes'][0].set_facecolor(self.color_palette['success'])
        box_plot['boxes'][0].set_alpha(0.7)
        ax2.set_ylabel('延迟 (微秒)')
        ax2.set_title('延迟箱型图')
        ax2.grid(True, alpha=0.3)
        
        # 添加统计信息
        stats_text = f'''统计信息:
Min: {np.min(latencies_array):.1f}μs
Q1: {np.percentile(latencies_array, 25):.1f}μs
Median: {np.median(latencies_array):.1f}μs
Q3: {np.percentile(latencies_array, 75):.1f}μs
Max: {np.max(latencies_array):.1f}μs
Std: {np.std(latencies_array):.1f}μs'''
        ax2.text(1.1, 0.5, stats_text, transform=ax2.transAxes, fontsize=9, 
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
        
        # 3. 延迟时间序列
        ax3 = axes[0, 2]
        time_points = range(len(latencies_array))
        ax3.plot(time_points, latencies_array, color=self.color_palette['primary'], alpha=0.6, linewidth=1)
        # 添加滚动平均
        window = min(100, len(latencies_array) // 10)
        if window > 1:
            rolling_mean = pd.Series(latencies_array).rolling(window=window).mean()
            ax3.plot(time_points, rolling_mean, color=self.color_palette['warning'], linewidth=2, label=f'滚动平均({window})')
            ax3.legend()
        ax3.set_xlabel('时间序列')
        ax3.set_ylabel('延迟 (微秒)')
        ax3.set_title('延迟时间序列')
        ax3.grid(True, alpha=0.3)
        
        # 4. 延迟CDF (累积分布函数)
        ax4 = axes[1, 0]
        sorted_latencies = np.sort(latencies_array)
        p = np.arange(1, len(sorted_latencies) + 1) / len(sorted_latencies)
        ax4.plot(sorted_latencies, p * 100, color=self.color_palette['success'], linewidth=2)
        ax4.axvline(np.percentile(latencies_array, 95), color=self.color_palette['danger'], linestyle='--', label='P95')
        ax4.axvline(np.percentile(latencies_array, 99), color=self.color_palette['warning'], linestyle='--', label='P99')
        ax4.set_xlabel('延迟 (微秒)')
        ax4.set_ylabel('累积概率 (%)')
        ax4.set_title('延迟累积分布函数')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. 延迟热力图 (按小时分组)
        ax5 = axes[1, 1]
        if len(latencies_array) >= 24:
            # 模拟24小时数据
            hours = np.array([i % 24 for i in range(len(latencies_array))])
            df_hourly = pd.DataFrame({'hour': hours, 'latency': latencies_array})
            hourly_stats = df_hourly.groupby('hour')['latency'].agg(['mean', 'std', 'count']).reset_index()
            
            # 创建热力图数据
            heatmap_data = hourly_stats.pivot_table(values='mean', index='hour', aggfunc='first').fillna(0)
            if len(heatmap_data) > 1:
                sns.heatmap(heatmap_data.T, annot=True, fmt='.1f', cmap='viridis', ax=ax5)
                ax5.set_title('延迟热力图 (按小时)')
                ax5.set_xlabel('小时')
                ax5.set_ylabel('')
        
        # 6. 延迟性能指标雷达图
        ax6 = axes[1, 2]
        
        # 计算性能指标 (标准化到0-100)
        metrics = {
            '平均延迟': 100 - min(100, np.mean(latencies_array) / 1000 * 100),  # 越低越好
            'P95延迟': 100 - min(100, np.percentile(latencies_array, 95) / 1000 * 100),
            'P99延迟': 100 - min(100, np.percentile(latencies_array, 99) / 1000 * 100),
            '稳定性': 100 - min(100, np.std(latencies_array) / np.mean(latencies_array) * 100),
            '一致性': 100 - min(100, (np.percentile(latencies_array, 75) - np.percentile(latencies_array, 25)) / np.median(latencies_array) * 100)
        }
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        values = list(metrics.values())
        
        # 闭合雷达图
        angles += angles[:1]
        values += values[:1]
        
        ax6.plot(angles, values, 'o-', linewidth=2, color=self.color_palette['primary'])
        ax6.fill(angles, values, color=self.color_palette['primary'], alpha=0.25)
        ax6.set_xticks(angles[:-1])
        ax6.set_xticklabels(metrics.keys(), fontsize=8)
        ax6.set_ylim(0, 100)
        ax6.set_title('延迟性能雷达图')
        ax6.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black')
            print(f"延迟分析图已保存: {save_path}")
        
        plt.show()
        
        # 计算延迟统计
        latency_stats = {
            'mean_us': float(np.mean(latencies_array)),
            'median_us': float(np.median(latencies_array)),
            'std_us': float(np.std(latencies_array)),
            'min_us': float(np.min(latencies_array)),
            'max_us': float(np.max(latencies_array)),
            'p95_us': float(np.percentile(latencies_array, 95)),
            'p99_us': float(np.percentile(latencies_array, 99)),
            'count': len(latencies_array)
        }
        
        return latency_stats
    
    def create_slippage_analysis(self, slippages_bps: List[float], volumes: List[float] = None, save_path: str = None) -> Dict[str, Any]:
        """创建滑点分析图表"""
        print("📊 生成滑点分析图表...")
        
        if not slippages_bps:
            print("警告: 没有滑点数据")
            return {}
        
        slippages_array = np.array(slippages_bps)
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('💹 滑点分析 (Slippage Analysis)', fontsize=16, color='white')
        
        # 1. 滑点分布
        ax1 = axes[0, 0]
        n, bins, patches = ax1.hist(slippages_array, bins=30, alpha=0.7, color=self.color_palette['secondary'], edgecolor='white')
        ax1.axvline(np.mean(slippages_array), color=self.color_palette['warning'], linestyle='--', linewidth=2, label=f'Mean: {np.mean(slippages_array):.2f}bps')
        ax1.set_xlabel('滑点 (基点)')
        ax1.set_ylabel('频次')
        ax1.set_title('滑点分布')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 滑点vs成交量散点图
        ax2 = axes[0, 1]
        if volumes and len(volumes) == len(slippages_array):
            volumes_array = np.array(volumes)
            scatter = ax2.scatter(volumes_array, slippages_array, alpha=0.6, c=self.color_palette['info'], s=30)
            ax2.set_xlabel('成交量')
            ax2.set_ylabel('滑点 (基点)')
            ax2.set_title('滑点 vs 成交量')
            
            # 添加趋势线
            if len(volumes_array) > 1:
                z = np.polyfit(volumes_array, slippages_array, 1)
                p = np.poly1d(z)
                ax2.plot(volumes_array, p(volumes_array), color=self.color_palette['danger'], linestyle='--', linewidth=2)
        else:
            # 滑点时间序列
            time_points = range(len(slippages_array))
            ax2.plot(time_points, slippages_array, color=self.color_palette['secondary'], alpha=0.6, linewidth=1)
            ax2.set_xlabel('时间序列')
            ax2.set_ylabel('滑点 (基点)')
            ax2.set_title('滑点时间序列')
        ax2.grid(True, alpha=0.3)
        
        # 3. 滑点累积分布
        ax3 = axes[1, 0]
        sorted_slippages = np.sort(slippages_array)
        p = np.arange(1, len(sorted_slippages) + 1) / len(sorted_slippages)
        ax3.plot(sorted_slippages, p * 100, color=self.color_palette['success'], linewidth=2)
        ax3.axvline(np.percentile(slippages_array, 90), color=self.color_palette['danger'], linestyle='--', label='P90')
        ax3.set_xlabel('滑点 (基点)')
        ax3.set_ylabel('累积概率 (%)')
        ax3.set_title('滑点累积分布')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 滑点统计总结
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        slippage_stats = {
            'Mean': f'{np.mean(slippages_array):.2f} bps',
            'Median': f'{np.median(slippages_array):.2f} bps',
            'Std': f'{np.std(slippages_array):.2f} bps',
            'Min': f'{np.min(slippages_array):.2f} bps',
            'Max': f'{np.max(slippages_array):.2f} bps',
            'P90': f'{np.percentile(slippages_array, 90):.2f} bps',
            'P95': f'{np.percentile(slippages_array, 95):.2f} bps',
            'Count': f'{len(slippages_array):,}'
        }
        
        # 创建统计表格
        table_data = []
        for key, value in slippage_stats.items():
            table_data.append([key, value])
        
        table = ax4.table(cellText=table_data, colLabels=['指标', '值'], 
                         cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # 设置表格样式
        for i, key in enumerate(['指标', '值']):
            table[(0, i)].set_facecolor(self.color_palette['dark'])
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax4.set_title('滑点统计摘要', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black')
            print(f"滑点分析图已保存: {save_path}")
        
        plt.show()
        
        return {
            'mean_bps': float(np.mean(slippages_array)),
            'median_bps': float(np.median(slippages_array)),
            'std_bps': float(np.std(slippages_array)),
            'p90_bps': float(np.percentile(slippages_array, 90)),
            'p95_bps': float(np.percentile(slippages_array, 95)),
            'count': len(slippages_array)
        }
    
    def create_trading_performance_dashboard(self, data: VisualizationData, save_path: str = None) -> Dict[str, Any]:
        """创建交易性能仪表盘"""
        print("📊 生成交易性能仪表盘...")
        
        # 创建子图
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[
                '价格与投资组合价值', '订单延迟分布', '滑点分布',
                '收益率分布', '回撤分析', '风险指标',
                '交易量分析', '胜率统计', '夏普比率'
            ],
            specs=[
                [{"secondary_y": True}, {}, {}],
                [{}, {}, {}],
                [{}, {}, {}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.08
        )
        
        # 1. 价格与投资组合价值
        if data.timestamps and data.prices and data.portfolio_values:
            timestamps_dt = [ts.to_datetime() for ts in data.timestamps]
            
            fig.add_trace(
                go.Scatter(x=timestamps_dt, y=data.prices, name='价格', line=dict(color='#00D4FF')),
                row=1, col=1, secondary_y=False
            )
            
            fig.add_trace(
                go.Scatter(x=timestamps_dt, y=data.portfolio_values, name='投资组合价值', line=dict(color='#FF6B6B')),
                row=1, col=1, secondary_y=True
            )
        
        # 2. 订单延迟分布
        if data.latencies_us:
            fig.add_trace(
                go.Histogram(x=data.latencies_us, name='延迟分布', marker_color='#4ECDC4', nbinsx=30),
                row=1, col=2
            )
        
        # 3. 滑点分布
        if data.slippages_bps:
            fig.add_trace(
                go.Histogram(x=data.slippages_bps, name='滑点分布', marker_color='#FFE66D', nbinsx=30),
                row=1, col=3
            )
        
        # 4. 收益率分布
        if data.trade_returns:
            fig.add_trace(
                go.Histogram(x=data.trade_returns, name='收益率分布', marker_color='#FF6B9D', nbinsx=30),
                row=2, col=1
            )
        
        # 5. 回撤分析
        if data.drawdowns:
            timestamps_dt = [ts.to_datetime() for ts in data.timestamps[:len(data.drawdowns)]]
            fig.add_trace(
                go.Scatter(x=timestamps_dt, y=data.drawdowns, name='回撤', 
                          line=dict(color='#95E1D3'), fill='tonexty'),
                row=2, col=2
            )
        
        # 6. 风险指标雷达图 (使用散点图模拟)
        if data.latencies_us and data.slippages_bps:
            risk_metrics = {
                '延迟风险': min(100, np.mean(data.latencies_us) / 10),
                '滑点风险': min(100, np.mean(data.slippages_bps) * 10),
                '波动风险': min(100, np.std(data.trade_returns) * 100) if data.trade_returns else 50,
            }
            
            categories = list(risk_metrics.keys())
            values = list(risk_metrics.values())
            
            fig.add_trace(
                go.Scatterpolar(r=values, theta=categories, fill='toself', name='风险指标'),
                row=2, col=3
            )
        
        # 7. 交易量分析
        if data.volumes:
            fig.add_trace(
                go.Bar(x=list(range(len(data.volumes))), y=data.volumes, name='交易量', marker_color='#2C3E50'),
                row=3, col=1
            )
        
        # 8. 胜率统计
        if data.trade_returns:
            winning_trades = [r for r in data.trade_returns if r > 0]
            losing_trades = [r for r in data.trade_returns if r <= 0]
            
            fig.add_trace(
                go.Bar(x=['盈利交易', '亏损交易'], y=[len(winning_trades), len(losing_trades)], 
                      marker_color=['#4ECDC4', '#FF6B6B'], name='交易统计'),
                row=3, col=2
            )
        
        # 9. 滚动夏普比率
        if data.rolling_sharpe:
            fig.add_trace(
                go.Scatter(x=list(range(len(data.rolling_sharpe))), y=data.rolling_sharpe, 
                          name='滚动夏普比率', line=dict(color='#ECF0F1')),
                row=3, col=3
            )
        
        # 更新布局
        fig.update_layout(
            title='🚀 高频交易性能仪表盘',
            height=1200,
            showlegend=False,
            plot_bgcolor='black',
            paper_bgcolor='black',
            font=dict(color='white')
        )
        
        # 保存或显示
        if save_path:
            fig.write_html(save_path)
            print(f"交易性能仪表盘已保存: {save_path}")
        else:
            fig.show()
        
        return {
            'dashboard_created': True,
            'components': 9,
            'data_points': len(data.timestamps) if data.timestamps else 0
        }
    
    def create_order_book_heatmap(self, orderbook_snapshots: List[Dict], save_path: str = None):
        """创建订单簿热力图"""
        print("📊 生成订单簿热力图...")
        
        if not orderbook_snapshots:
            print("警告: 没有订单簿数据")
            return
        
        # 提取价格层级数据
        all_prices = []
        all_quantities = []
        timestamps = []
        
        for i, snapshot in enumerate(orderbook_snapshots):
            timestamp = i  # 简化时间戳
            
            # 买盘数据
            for price, qty, _ in snapshot.get('bids', []):
                all_prices.append(price)
                all_quantities.append(qty)
                timestamps.append(timestamp)
            
            # 卖盘数据
            for price, qty, _ in snapshot.get('asks', []):
                all_prices.append(price)
                all_quantities.append(-qty)  # 卖盘用负数表示
                timestamps.append(timestamp)
        
        if not all_prices:
            print("警告: 订单簿数据为空")
            return
        
        # 创建DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'price': all_prices,
            'quantity': all_quantities
        })
        
        # 创建价格区间
        price_min, price_max = df['price'].min(), df['price'].max()
        price_bins = np.linspace(price_min, price_max, 50)
        df['price_bin'] = pd.cut(df['price'], bins=price_bins)
        
        # 聚合数据
        heatmap_data = df.groupby(['timestamp', 'price_bin'])['quantity'].sum().unstack(fill_value=0)
        
        # 创建热力图
        plt.figure(figsize=(16, 10))
        
        # 使用自定义颜色映射 (红色=卖盘, 绿色=买盘)
        colors = ['#FF4444', '#000000', '#44FF44']
        n_bins = 256
        cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list('orderbook', colors, N=n_bins)
        
        sns.heatmap(heatmap_data.T, cmap=cmap, center=0, cbar_kws={'label': '数量'})
        
        plt.title('📚 订单簿深度热力图', fontsize=16, color='white', pad=20)
        plt.xlabel('时间快照')
        plt.ylabel('价格区间')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black')
            print(f"订单簿热力图已保存: {save_path}")
        
        plt.show()
    
    def create_execution_timeline(self, orders: List[UnifiedOrderResponse], save_path: str = None):
        """创建执行时间线图"""
        print("📊 生成执行时间线图...")
        
        if not orders:
            print("警告: 没有订单数据")
            return
        
        # 提取数据
        timestamps = [order.timestamp.to_datetime() for order in orders]
        latencies = [order.latency_us for order in orders]
        order_types = [order.order_type.value for order in orders]
        sides = [order.side.value for order in orders]
        
        # 创建图表
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12))
        fig.suptitle('⏱️ 订单执行时间线', fontsize=16, color='white')
        
        # 1. 延迟时间线
        colors = ['#4ECDC4' if side == 'BUY' else '#FF6B6B' for side in sides]
        scatter = ax1.scatter(timestamps, latencies, c=colors, alpha=0.6, s=30)
        ax1.set_ylabel('延迟 (微秒)')
        ax1.set_title('订单执行延迟')
        ax1.grid(True, alpha=0.3)
        
        # 添加延迟趋势线
        if len(latencies) > 1:
            time_numeric = [(ts - timestamps[0]).total_seconds() for ts in timestamps]
            z = np.polyfit(time_numeric, latencies, 1)
            p = np.poly1d(z)
            trend_line = [p(t) for t in time_numeric]
            ax1.plot(timestamps, trend_line, color='yellow', linestyle='--', linewidth=2, alpha=0.8)
        
        # 2. 订单类型分布
        order_type_counts = pd.Series(order_types).value_counts()
        bars = ax2.bar(range(len(order_type_counts)), order_type_counts.values, 
                      color=[self.color_palette['primary'], self.color_palette['secondary']])
        ax2.set_xticks(range(len(order_type_counts)))
        ax2.set_xticklabels(order_type_counts.index)
        ax2.set_ylabel('数量')
        ax2.set_title('订单类型分布')
        ax2.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, count in zip(bars, order_type_counts.values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(order_type_counts.values)*0.01, 
                    str(count), ha='center', va='bottom', color='white')
        
        # 3. 买卖订单时间分布
        buy_times = [ts for ts, side in zip(timestamps, sides) if side == 'BUY']
        sell_times = [ts for ts, side in zip(timestamps, sides) if side == 'SELL']
        
        ax3.hist([buy_times, sell_times], bins=20, alpha=0.7, 
                label=['买单', '卖单'], color=['#4ECDC4', '#FF6B6B'])
        ax3.set_xlabel('时间')
        ax3.set_ylabel('订单数量')
        ax3.set_title('买卖订单时间分布')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black')
            print(f"执行时间线图已保存: {save_path}")
        
        plt.show()
    
    def generate_performance_report(self, data: VisualizationData, output_dir: str = "analysis_output"):
        """生成完整的性能分析报告"""
        print("📈 生成完整性能分析报告...")
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'analysis_components': []
        }
        
        # 1. 延迟分析
        if data.latencies_us:
            latency_stats = self.create_latency_analysis(
                data.latencies_us, 
                save_path=f"{output_dir}/latency_analysis.png"
            )
            report['latency_analysis'] = latency_stats
            report['analysis_components'].append('latency_analysis')
        
        # 2. 滑点分析
        if data.slippages_bps:
            slippage_stats = self.create_slippage_analysis(
                data.slippages_bps, 
                data.volumes,
                save_path=f"{output_dir}/slippage_analysis.png"
            )
            report['slippage_analysis'] = slippage_stats
            report['analysis_components'].append('slippage_analysis')
        
        # 3. 交易性能仪表盘
        dashboard_result = self.create_trading_performance_dashboard(
            data,
            save_path=f"{output_dir}/performance_dashboard.html"
        )
        report['dashboard'] = dashboard_result
        report['analysis_components'].append('performance_dashboard')
        
        # 4. 订单执行时间线
        if data.orders:
            self.create_execution_timeline(
                data.orders,
                save_path=f"{output_dir}/execution_timeline.png"
            )
            report['analysis_components'].append('execution_timeline')
        
        # 5. 保存报告摘要
        import json
        with open(f"{output_dir}/analysis_report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"✅ 完整分析报告已生成到目录: {output_dir}")
        print(f"   包含组件: {', '.join(report['analysis_components'])}")
        
        return report

def demo_advanced_visualization():
    """演示高级可视化功能"""
    print("🎨 高级可视化功能演示")
    print("=" * 50)
    
    # 创建可视化器
    visualizer = AdvancedVisualizer()
    
    # 生成示例数据
    np.random.seed(42)
    n_samples = 1000
    
    # 生成延迟数据 (模拟真实分布)
    base_latency = 200  # 200微秒基础延迟
    latencies_us = np.random.lognormal(np.log(base_latency), 0.5, n_samples)
    latencies_us = np.clip(latencies_us, 10, 2000)  # 10-2000微秒范围
    
    print(f"生成延迟数据: {len(latencies_us)} 个样本")
    print(f"延迟范围: {np.min(latencies_us):.1f} - {np.max(latencies_us):.1f} 微秒")
    
    # 生成滑点数据
    base_slippage = 2.0  # 2基点基础滑点
    slippages_bps = np.random.exponential(base_slippage, n_samples)
    slippages_bps = np.clip(slippages_bps, 0.1, 20)  # 0.1-20基点范围
    
    print(f"生成滑点数据: {len(slippages_bps)} 个样本")
    print(f"滑点范围: {np.min(slippages_bps):.2f} - {np.max(slippages_bps):.2f} 基点")
    
    # 生成交易数据
    timestamps = [HighPrecisionTimestamp.now().add_microseconds(i * 1000) for i in range(n_samples)]
    prices = 45000 + np.cumsum(np.random.normal(0, 1, n_samples))  # 随机游走价格
    volumes = np.random.exponential(10, n_samples)
    portfolio_values = 100000 + np.cumsum(np.random.normal(10, 50, n_samples))  # 投资组合增长
    
    # 生成交易收益
    trade_returns = np.random.normal(0.001, 0.02, n_samples // 10)  # 0.1%平均收益，2%波动
    
    # 计算回撤
    cumulative_returns = np.cumprod(1 + trade_returns)
    rolling_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = (cumulative_returns - rolling_max) / rolling_max
    
    # 创建可视化数据
    viz_data = VisualizationData(
        timestamps=timestamps,
        prices=prices.tolist(),
        volumes=volumes.tolist(),
        orders=[],  # 简化示例，不包含详细订单
        portfolio_values=portfolio_values.tolist(),
        latencies_us=latencies_us.tolist(),
        slippages_bps=slippages_bps.tolist(),
        spreads_bps=[],
        trade_returns=trade_returns.tolist(),
        drawdowns=drawdowns.tolist(),
        rolling_sharpe=[]
    )
    
    # 测试各个可视化功能
    print("\n1. 📊 延迟分析...")
    latency_stats = visualizer.create_latency_analysis(latencies_us.tolist())
    
    print("\n2. 💹 滑点分析...")
    slippage_stats = visualizer.create_slippage_analysis(slippages_bps.tolist(), volumes.tolist())
    
    print("\n3. 📈 性能仪表盘...")
    dashboard_stats = visualizer.create_trading_performance_dashboard(viz_data)
    
    # 生成完整报告
    print("\n4. 📋 生成完整报告...")
    report = visualizer.generate_performance_report(viz_data)
    
    print("\n✅ 高级可视化演示完成!")
    
    return {
        'latency_stats': latency_stats,
        'slippage_stats': slippage_stats,
        'dashboard_stats': dashboard_stats,
        'report_components': len(report['analysis_components'])
    }

if __name__ == "__main__":
    # 运行可视化演示
    results = demo_advanced_visualization()
    
    print(f"\n🎯 可视化功能总结:")
    print(f"- 延迟分析: 平均 {results['latency_stats']['mean_us']:.1f}μs, P95 {results['latency_stats']['p95_us']:.1f}μs")
    print(f"- 滑点分析: 平均 {results['slippage_stats']['mean_bps']:.2f}bps, P95 {results['slippage_stats']['p95_bps']:.2f}bps")
    print(f"- 仪表盘组件: {results['dashboard_stats']['components']} 个")
    print(f"- 报告组件: {results['report_components']} 个")
    print(f"- 输出目录: analysis_output/")