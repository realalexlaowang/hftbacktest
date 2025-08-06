import numpy as np
from numba import njit
from hftbacktest import BacktestAsset, HashMapMarketDepthBacktest

@njit
def simple_observer(hbt):
    """
    简单的市场观察者 - 我们的第一个策略
    只观察市场数据，不进行任何交易
    """
    asset_no = 0  # 我们只有一个资产
    iteration = 0
    
    print("开始观察市场数据...")
    
    # 主循环：每10秒观察一次市场
    while hbt.elapse(10_000_000_000) == 0:  # 10秒 = 10 * 10^9 纳秒
        iteration += 1
        
        # 获取市场深度信息
        depth = hbt.depth(asset_no)
        
        # 获取当前时间戳
        current_time = hbt.current_timestamp
        
        # 打印市场信息
        print(f"第{iteration}次观察:")
        print(f"  时间戳: {current_time}")
        print(f"  最佳买价: {depth.best_bid:.1f}")
        print(f"  最佳卖价: {depth.best_ask:.1f}")
        print(f"  价差: {(depth.best_ask - depth.best_bid):.1f}")
        print(f"  中间价: {((depth.best_bid + depth.best_ask) / 2):.1f}")
        print("-" * 40)
        
        # 限制观察次数，避免输出过多
        if iteration >= 10:
            print("观察完成！")
            break
    
    return True

def run_first_backtest():
    """运行我们的第一个回测"""
    print("=== HftBacktest 第一个示例 ===\n")
    
    # 配置资产
    asset = (
        BacktestAsset()
        # 指定数据文件
        .data(['data/sample_btcusdt.npz'])
        # 配置为线性资产（如BTCUSDT）
        .linear_asset(1.0)
        # 设置常数延迟：进单延迟10ms，响应延迟10ms
        .constant_latency(10_000_000, 10_000_000)  
        # 使用风险规避队列模型
        .risk_adverse_queue_model()
        # 不允许部分成交
        .no_partial_fill_exchange()
        # 设置手续费：做市商-0.005%，吃单者0.07%
        .trading_value_fee_model(-0.00005, 0.0007)
        # 设置最小价格变动
        .tick_size(0.1)
        # 设置最小交易单位
        .lot_size(0.001)
    )
    
    print("配置完成，资产参数:")
    print(f"- 最小价格变动(tick_size): 0.1 USDT")
    print(f"- 最小交易单位(lot_size): 0.001 BTC")
    print(f"- 做市商手续费: -0.005% (返佣)")
    print(f"- 吃单者手续费: 0.07%")
    print(f"- 订单延迟: 10ms")
    print()
    
    # 创建回测引擎
    print("创建回测引擎...")
    hbt = HashMapMarketDepthBacktest([asset])
    
    print("开始回测...")
    try:
        # 运行我们的观察策略
        result = simple_observer(hbt)
        if result:
            print("\n✅ 回测成功完成！")
        else:
            print("\n❌ 回测过程中出现错误")
    except Exception as e:
        print(f"\n❌ 回测失败: {e}")
    
    # 显示基本统计信息
    print("\n=== 回测统计 ===")
    print(f"当前时间戳: {hbt.current_timestamp}")
    print(f"当前持仓: {hbt.position(0):.6f} BTC")
    print(f"账户余额: {hbt.balance:.2f} USDT")

if __name__ == "__main__":
    run_first_backtest()