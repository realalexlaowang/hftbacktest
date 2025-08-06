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
    
    # 主循环：每5秒观察一次市场
    while hbt.elapse(5_000_000_000) == 0:  # 5秒 = 5 * 10^9 纳秒
        iteration += 1
        
        # 获取市场深度信息
        depth = hbt.depth(asset_no)
        
        # 获取当前时间戳
        current_time = hbt.current_timestamp
        
        # 获取当前持仓
        position = hbt.position(asset_no)
        
        # 使用简单的print语句（Numba兼容）
        print("=== 第", iteration, "次观察 ===")
        print("时间戳:", current_time)
        print("最佳买价:", round(depth.best_bid, 1))
        print("最佳卖价:", round(depth.best_ask, 1))
        print("价差:", round(depth.best_ask - depth.best_bid, 1))
        print("中间价:", round((depth.best_bid + depth.best_ask) / 2, 1))
        print("当前持仓:", round(position, 6))
        print()
        
        # 限制观察次数，避免输出过多
        if iteration >= 10:
            print("观察完成！")
            break
    
    return True

def run_working_backtest():
    """运行可工作的回测示例"""
    print("=== HftBacktest 可工作示例 ===\n")
    
    # 配置资产
    asset = (
        BacktestAsset()
        # 指定数据文件
        .data(['data/btcusdt_sample.npz'])
        # 配置为线性资产（如BTCUSDT）
        .linear_asset(1.0)
        # 使用新的API设置订单延迟
        .constant_order_latency(10_000_000, 10_000_000)  
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
    print("- 资产: BTCUSDT (模拟)")
    print("- 最小价格变动(tick_size): 0.1 USDT")
    print("- 最小交易单位(lot_size): 0.001 BTC")
    print("- 做市商手续费: -0.005% (返佣)")
    print("- 吃单者手续费: 0.07%")
    print("- 订单延迟: 10ms")
    print()
    
    # 创建回测引擎
    print("创建回测引擎...")
    hbt = HashMapMarketDepthBacktest([asset])
    
    print("开始回测...\n")
    try:
        # 运行我们的观察策略
        result = simple_observer(hbt)
        if result:
            print("✅ 回测成功完成！")
        else:
            print("❌ 回测过程中出现错误")
    except Exception as e:
        print(f"❌ 回测失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 显示基本统计信息
    print("\n=== 最终统计 ===")
    print(f"最终时间戳: {hbt.current_timestamp}")
    print(f"最终持仓: {hbt.position(0):.6f} BTC")
    
    # 尝试获取其他可用的统计信息
    try:
        depth = hbt.depth(0)
        print(f"最后的买价: {depth.best_bid:.1f} USDT")
        print(f"最后的卖价: {depth.best_ask:.1f} USDT")
        print(f"最后的中间价: {(depth.best_bid + depth.best_ask) / 2:.1f} USDT")
    except:
        print("无法获取最终市场深度信息")

if __name__ == "__main__":
    run_working_backtest()