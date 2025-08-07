#!/usr/bin/env python
"""
AI-HFT Backtester 快速开始脚本
这个脚本帮助新手快速体验系统功能
"""

import sys
import os
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_banner():
    """打印欢迎横幅"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║      🚀 AI-HFT Backtester - 高频交易回测系统                  ║
    ║      版本: 0.1.0  |  作者: AI HFT Team                        ║
    ╔═══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_environment():
    """检查环境配置"""
    print("\n📋 检查环境配置...")
    
    # 检查Python版本
    python_version = sys.version_info
    if python_version.major >= 3 and python_version.minor >= 8:
        print(f"✅ Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        print(f"❌ Python版本过低: {python_version.major}.{python_version.minor}")
        print("   需要Python 3.8或更高版本")
        return False
    
    # 检查必要的包
    required_packages = ['numpy', 'pandas', 'torch', 'numba']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} 已安装")
        except ImportError:
            print(f"❌ {package} 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n请安装缺失的包: pip install {' '.join(missing_packages)}")
        return False
    
    return True

def generate_sample_data(n_hours=1):
    """生成示例数据"""
    print(f"\n📊 生成{n_hours}小时的示例数据...")
    
    # 时间范围
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = []
    
    # 生成订单簿数据
    orderbook_data = []
    trades_data = []
    
    # 初始价格
    base_price = 50000
    current_price = base_price
    
    # 每秒生成一次数据
    for i in range(n_hours * 3600):
        timestamp = start_time + timedelta(seconds=i)
        timestamp_ms = int(timestamp.timestamp() * 1000)
        timestamps.append(timestamp_ms)
        
        # 价格随机游走
        price_change = np.random.normal(0, 0.0001)
        current_price = current_price * (1 + price_change)
        
        # 生成买卖盘
        spread = current_price * 0.0001  # 0.01%的价差
        best_bid = current_price - spread/2
        best_ask = current_price + spread/2
        
        # 订单簿深度
        bids = []
        asks = []
        for j in range(10):
            bid_price = best_bid - j * 0.1
            ask_price = best_ask + j * 0.1
            bid_volume = np.random.exponential(1) * (10 - j) / 10
            ask_volume = np.random.exponential(1) * (10 - j) / 10
            bids.append([bid_price, bid_volume])
            asks.append([ask_price, ask_volume])
        
        orderbook_data.append({
            'timestamp': timestamp_ms,
            'bids': bids,
            'asks': asks,
            'mid_price': current_price
        })
        
        # 随机生成成交
        if np.random.random() < 0.3:  # 30%概率有成交
            trade_price = current_price + np.random.normal(0, spread/4)
            trade_volume = np.random.exponential(0.1)
            trade_side = 'buy' if np.random.random() < 0.5 else 'sell'
            
            trades_data.append({
                'timestamp': timestamp_ms + np.random.randint(0, 999),
                'price': trade_price,
                'quantity': trade_volume,
                'side': trade_side
            })
    
    print(f"✅ 生成了 {len(orderbook_data)} 个订单簿快照")
    print(f"✅ 生成了 {len(trades_data)} 笔成交记录")
    
    return pd.DataFrame(orderbook_data), pd.DataFrame(trades_data)

def run_simple_backtest():
    """运行简单回测示例"""
    print("\n🚀 运行简单回测示例...")
    
    try:
        from ai_hft_backtester import Backtester
        from ai_hft_backtester.strategies import SimpleMarketMaker
    except ImportError:
        print("❌ 无法导入回测模块，请确保已正确安装")
        return
    
    # 生成示例数据
    orderbook_df, trades_df = generate_sample_data(n_hours=1)
    
    # 模拟策略运行
    print("\n📈 模拟策略运行...")
    
    # 策略参数
    initial_capital = 10000
    position = 0
    pnl = 0
    trades_count = 0
    
    # 简单的做市策略逻辑
    for i in range(100):  # 处理前100个时间点
        mid_price = orderbook_df.iloc[i]['mid_price']
        
        # 模拟下单
        if i % 10 == 0:  # 每10秒交易一次
            if position == 0:
                # 开仓
                position = 0.01
                entry_price = mid_price
                trades_count += 1
                print(f"  [交易 {trades_count}] 买入 0.01 BTC @ ${mid_price:.2f}")
            else:
                # 平仓
                exit_price = mid_price
                trade_pnl = position * (exit_price - entry_price)
                pnl += trade_pnl
                position = 0
                trades_count += 1
                print(f"  [交易 {trades_count}] 卖出 0.01 BTC @ ${mid_price:.2f}, 盈亏: ${trade_pnl:.2f}")
    
    # 显示结果
    print("\n📊 回测结果汇总:")
    print("=" * 50)
    print(f"初始资金: ${initial_capital:.2f}")
    print(f"最终资金: ${initial_capital + pnl:.2f}")
    print(f"总盈亏: ${pnl:.2f}")
    print(f"收益率: {pnl/initial_capital*100:.2f}%")
    print(f"交易次数: {trades_count}")
    print("=" * 50)

def show_next_steps():
    """显示下一步操作"""
    print("\n🎯 下一步:")
    print("1. 阅读详细文档:")
    print("   - 菜鸟指南: BEGINNER_GUIDE.md")
    print("   - 用户手册: USER_GUIDE.md")
    print("   - 发展蓝图: ROADMAP.md")
    print("\n2. 尝试更多示例:")
    print("   - examples/simple_backtest.py")
    print("   - examples/ai_strategy.py")
    print("   - examples/train_model.py")
    print("\n3. 开发自己的策略:")
    print("   - 继承 BaseStrategy 类")
    print("   - 实现交易逻辑")
    print("   - 集成AI模型")
    print("\n4. 加入社区:")
    print("   - GitHub: https://github.com/ai-hft-backtester")
    print("   - Discord: https://discord.gg/ai-hft")
    
def interactive_menu():
    """交互式菜单"""
    while True:
        print("\n📋 请选择操作:")
        print("1. 检查环境配置")
        print("2. 生成示例数据")
        print("3. 运行简单回测")
        print("4. 查看下一步")
        print("5. 退出")
        
        choice = input("\n请输入选项 (1-5): ")
        
        if choice == '1':
            check_environment()
        elif choice == '2':
            generate_sample_data()
        elif choice == '3':
            run_simple_backtest()
        elif choice == '4':
            show_next_steps()
        elif choice == '5':
            print("\n👋 感谢使用AI-HFT Backtester！")
            break
        else:
            print("❌ 无效选项，请重试")

def main():
    """主函数"""
    print_banner()
    
    # 检查是否有命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] == '--check':
            check_environment()
        elif sys.argv[1] == '--demo':
            if check_environment():
                run_simple_backtest()
        elif sys.argv[1] == '--help':
            print("使用方法:")
            print("  python quickstart.py          # 交互式菜单")
            print("  python quickstart.py --check  # 检查环境")
            print("  python quickstart.py --demo   # 运行演示")
            print("  python quickstart.py --help   # 显示帮助")
        else:
            print(f"未知参数: {sys.argv[1]}")
            print("使用 --help 查看帮助")
    else:
        # 交互式模式
        interactive_menu()

if __name__ == "__main__":
    main()