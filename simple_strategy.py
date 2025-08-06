import numpy as np
from numba import njit
from hftbacktest import BacktestAsset, HashMapMarketDepthBacktest, BUY, SELL, GTX, LIMIT

@njit
def simple_market_making_strategy(hbt):
    """
    🎯 第一个简单的做市策略 (Numba兼容版本)
    
    策略逻辑：
    1. 在当前最佳价格附近放置买卖订单
    2. 赚取买卖价差
    3. 控制持仓风险
    """
    asset_no = 0
    tick_size = hbt.depth(asset_no).tick_size
    lot_size = hbt.depth(asset_no).lot_size
    
    # 策略参数
    spread_ticks = 2      # 我们的报价距离最佳价格的tick数
    order_qty = 0.01      # 每单数量 (0.01 BTC)
    max_position = 0.05   # 最大持仓 (0.05 BTC)
    run_time_seconds = 60  # 运行1分钟（缩短时间方便测试）
    
    print("=== 开始简单做市策略 ===")
    print("策略参数:")
    print("- 价差:", spread_ticks, "ticks")
    print("- 订单量:", order_qty, "BTC")
    print("- 最大持仓:", max_position, "BTC")
    print("- 运行时间:", run_time_seconds, "秒")
    print()
    
    start_time = hbt.current_timestamp
    target_end_time = start_time + run_time_seconds * 1_000_000_000  # 转换为纳秒
    iteration = 0
    successful_orders = 0
    
    # 策略主循环
    while hbt.current_timestamp < target_end_time:
        # 每2秒检查一次
        if hbt.elapse(2_000_000_000) != 0:  # 2秒
            break
            
        iteration += 1
        
        # 清理无效订单（已成交、已取消等）
        hbt.clear_inactive_orders(asset_no)
        
        # 获取当前市场状态
        depth = hbt.depth(asset_no)
        position = hbt.position(asset_no)
        
        print("=== 第", iteration, "次检查 ===")
        print("当前最佳买价:", round(depth.best_bid, 1))
        print("当前最佳卖价:", round(depth.best_ask, 1))
        print("当前持仓:", round(position, 4), "BTC")
        
        # 检查是否有有效的买卖价
        if depth.best_bid > 0 and depth.best_ask > 0:
            # 计算我们的报价
            our_bid_price = depth.best_bid + spread_ticks * tick_size
            our_ask_price = depth.best_ask - spread_ticks * tick_size
            
            print("我们的买价:", round(our_bid_price, 1))
            print("我们的卖价:", round(our_ask_price, 1))
            
            # 风险控制：检查持仓限制
            can_buy = position < max_position
            can_sell = position > -max_position
            
            # 计算订单ID（简单方法：使用iteration + 偏移）
            buy_order_id = iteration * 2
            sell_order_id = iteration * 2 + 1
            
            # 提交买单（如果允许且价格合理）
            if can_buy and our_bid_price > 0 and our_bid_price < depth.best_ask:
                hbt.submit_buy_order(
                    asset_no,           # 资产编号
                    buy_order_id,       # 订单ID
                    our_bid_price,      # 价格
                    order_qty,          # 数量
                    GTX,                # 订单类型
                    LIMIT,              # 限价单
                    False               # 不是后处理订单
                )
                print("✅ 提交买单: ID", buy_order_id, "价格", round(our_bid_price, 1))
                successful_orders += 1
                
            # 提交卖单（如果允许且价格合理）
            if can_sell and our_ask_price > 0 and our_ask_price > depth.best_bid:
                hbt.submit_sell_order(
                    asset_no,
                    sell_order_id,
                    our_ask_price,
                    order_qty,
                    GTX,
                    LIMIT,
                    False
                )
                print("✅ 提交卖单: ID", sell_order_id, "价格", round(our_ask_price, 1))
                successful_orders += 1
        else:
            print("❌ 市场数据无效，跳过此次检查")
            
        print("持仓:", round(position, 4), "成功订单数:", successful_orders)
        print("---")
    
    print("=== 策略运行完成 ===")
    
    # 清理所有剩余订单
    hbt.clear_inactive_orders(asset_no)
    
    final_position = hbt.position(asset_no)
    final_timestamp = hbt.current_timestamp
    total_runtime = (final_timestamp - start_time) / 1_000_000_000
    
    print("📈 最终统计:")
    print("运行时间:", round(total_runtime, 1), "秒")
    print("最终持仓:", round(final_position, 6), "BTC")
    print("总检查次数:", iteration)
    print("成功提交订单:", successful_orders)
    
    return True

def run_simple_strategy():
    """运行简单策略"""
    print("=== 我的第一个HftBacktest交易策略 ===\n")
    
    # 首先创建测试数据
    create_simple_test_data()
    
    # 配置资产
    asset = (
        BacktestAsset()
        .data(['data/simple_btcusdt.npz'])
        .linear_asset(1.0)
        .constant_order_latency(10_000_000, 10_000_000)  # 10ms延迟
        .risk_adverse_queue_model()
        .no_partial_fill_exchange()
        .trading_value_fee_model(-0.00005, 0.0007)  # 币安手续费
        .tick_size(0.1)
        .lot_size(0.001)
    )
    
    print("✅ 资产配置完成")
    
    # 创建回测引擎
    hbt = HashMapMarketDepthBacktest([asset])
    print("✅ 回测引擎已创建")
    
    print("\n🚀 开始运行交易策略...\n")
    
    result = simple_market_making_strategy(hbt)
    
    if result:
        print("\n🎉 交易策略执行成功！")
        print("🎓 恭喜！您已经成功运行了第一个HftBacktest策略")
    else:
        print("\n❌ 交易策略执行失败")

def create_simple_test_data():
    """创建简单的测试数据"""
    import os
    
    # 定义事件常量
    BUY_EVENT = 1
    SELL_EVENT = 2  
    DEPTH_EVENT = 1 << 31
    EXCH_EVENT = 1 << 29
    LOCAL_EVENT = 1 << 28
    
    print("📊 生成简单测试数据...")
    
    events = []
    start_time = 1000000000000000000
    current_time = start_time
    
    # 创建稳定的市场数据，60秒，每2秒一个数据点
    for i in range(35):  # 35个时间点，约70秒数据
        current_time += 2_000_000_000  # 固定2秒间隔
        
        # 价格围绕50000小幅波动
        base_price = 50000.0 + np.sin(i * 0.1) * 2.0  # 正弦波动
        bid_price = base_price - 0.5  # 买价
        ask_price = base_price + 0.5  # 卖价
        
        exch_ts = current_time - 5_000_000  # 5ms前
        local_ts = current_time
        
        # 添加买方深度
        events.append([
            BUY_EVENT | DEPTH_EVENT | EXCH_EVENT | LOCAL_EVENT,
            exch_ts, local_ts, bid_price, 2.0,  # 固定数量2.0
            0, 0, 0.0
        ])
        
        # 添加卖方深度
        events.append([
            SELL_EVENT | DEPTH_EVENT | EXCH_EVENT | LOCAL_EVENT,
            exch_ts, local_ts, ask_price, 2.0,  # 固定数量2.0
            0, 0, 0.0
        ])
    
    # 转换并保存
    events_array = np.array(events)
    
    structured_events = np.zeros(len(events_array), dtype=[
        ('ev', 'u8'), ('exch_ts', 'i8'), ('local_ts', 'i8'),
        ('px', 'f8'), ('qty', 'f8'), ('order_id', 'u8'),
        ('ival', 'i8'), ('fval', 'f8')
    ])
    
    for i, field in enumerate(['ev', 'exch_ts', 'local_ts', 'px', 'qty', 'order_id', 'ival', 'fval']):
        structured_events[field] = events_array[:, i]
    
    os.makedirs("data", exist_ok=True)
    np.savez_compressed("data/simple_btcusdt.npz", data=structured_events)
    
    print(f"✅ 生成了 {len(structured_events)} 个事件的简单数据")
    print(f"价格范围: {structured_events['px'].min():.1f} - {structured_events['px'].max():.1f}")

if __name__ == "__main__":
    run_simple_strategy()