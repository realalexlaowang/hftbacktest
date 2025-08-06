import numpy as np
from numba import njit
from hftbacktest import BacktestAsset, HashMapMarketDepthBacktest, BUY, SELL, GTX, LIMIT

@njit
def simple_market_making_strategy(hbt):
    """
    🎯 第一个简单的做市策略
    
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
    run_time_seconds = 300  # 运行5分钟
    
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
    total_trades = 0
    
    # 策略主循环
    while hbt.current_timestamp < target_end_time:
        # 每1秒检查一次
        if hbt.elapse(1_000_000_000) != 0:  # 1秒
            break
            
        iteration += 1
        
        # 清理无效订单（已成交、已取消等）
        hbt.clear_inactive_orders(asset_no)
        
        # 获取当前市场状态
        depth = hbt.depth(asset_no)
        position = hbt.position(asset_no)
        
        # 检查是否有有效的买卖价
        if depth.best_bid <= 0 or depth.best_ask <= 0:
            print("第", iteration, "次: 无效市场数据，跳过")
            continue
            
        # 计算目标价格
        our_bid_price = depth.best_bid + spread_ticks * tick_size
        our_ask_price = depth.best_ask - spread_ticks * tick_size
        
        # 风险控制：检查持仓限制
        can_buy = position < max_position
        can_sell = position > -max_position
        
        # 检查当前是否有订单
        orders = hbt.orders(asset_no)
        has_buy_order = False
        has_sell_order = False
        
        # 计算订单ID（使用价格tick作为简单的ID）
        buy_order_id = int(our_bid_price / tick_size)
        sell_order_id = int(our_ask_price / tick_size) + 1000000  # 加偏移避免冲突
        
        print("第", iteration, "次检查:")
        print("  当前最佳买价:", round(depth.best_bid, 1))
        print("  当前最佳卖价:", round(depth.best_ask, 1))
        print("  我们的买价:", round(our_bid_price, 1))
        print("  我们的卖价:", round(our_ask_price, 1))
        print("  当前持仓:", round(position, 4))
        
        try:
            # 提交买单（如果允许）
            if can_buy and our_bid_price > 0:
                hbt.submit_buy_order(
                    asset_no,           # 资产编号
                    buy_order_id,       # 订单ID
                    our_bid_price,      # 价格
                    order_qty,          # 数量
                    GTX,                # 订单类型：Good Till Crossing
                    LIMIT,              # 限价单
                    False               # 不是后处理订单
                )
                print("  ✅ 提交买单: ID", buy_order_id, "价格", round(our_bid_price, 1))
                
            # 提交卖单（如果允许）
            if can_sell and our_ask_price > 0:
                hbt.submit_sell_order(
                    asset_no,
                    sell_order_id,
                    our_ask_price,
                    order_qty,
                    GTX,
                    LIMIT,
                    False
                )
                print("  ✅ 提交卖单: ID", sell_order_id, "价格", round(our_ask_price, 1))
                
        except Exception as e:
            print("  ❌ 提交订单失败")
            
        print("  ---")
        
        # 每10次检查显示一次统计
        if iteration % 10 == 0:
            elapsed_time = (hbt.current_timestamp - start_time) / 1_000_000_000
            print("📊 第", iteration, "次检查 (", round(elapsed_time, 1), "秒):")
            print("   持仓:", round(position, 4), "BTC")
            print("   活跃订单数:", "待实现")  # 实际项目中可以统计
            print()
    
    print("=== 策略运行完成 ===")
    
    # 清理所有剩余订单
    hbt.clear_inactive_orders(asset_no)
    
    final_position = hbt.position(asset_no)
    final_timestamp = hbt.current_timestamp
    total_runtime = (final_timestamp - start_time) / 1_000_000_000
    
    print("📈 最终统计:")
    print("   运行时间:", round(total_runtime, 1), "秒")
    print("   最终持仓:", round(final_position, 6), "BTC")
    print("   总检查次数:", iteration)
    
    return True

def run_first_trading_strategy():
    """运行第一个交易策略"""
    print("=== 我的第一个HftBacktest交易策略 ===\n")
    
    # 首先，让我们创建更好的测试数据
    # 因为之前的数据卖价有问题
    create_better_test_data()
    
    # 配置资产
    asset = (
        BacktestAsset()
        .data(['data/better_btcusdt.npz'])
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
    
    try:
        result = simple_market_making_strategy(hbt)
        if result:
            print("\n🎉 交易策略执行成功！")
        else:
            print("\n❌ 交易策略执行失败")
    except Exception as e:
        print(f"\n💥 策略运行出错: {e}")
        import traceback
        traceback.print_exc()

def create_better_test_data():
    """创建更好的测试数据，确保买卖价都有效"""
    import os
    
    # 定义事件常量
    BUY_EVENT = 1
    SELL_EVENT = 2  
    DEPTH_EVENT = 1 << 31
    EXCH_EVENT = 1 << 29
    LOCAL_EVENT = 1 << 28
    
    print("📊 生成更好的测试数据...")
    
    events = []
    start_time = 1000000000000000000
    current_time = start_time
    
    # 创建更稳定的市场数据
    for i in range(500):  # 500个时间点，足够5分钟使用
        current_time += np.random.randint(500_000_000, 1_000_000_000)  # 0.5-1秒间隔
        
        # 稳定的价格围绕50000波动
        base_price = 50000.0 + np.random.normal(0, 1.0)  # 小幅波动
        bid_price = base_price - 0.5  # 买价低0.5
        ask_price = base_price + 0.5  # 卖价高0.5
        
        exch_ts = current_time - np.random.randint(1_000_000, 5_000_000)
        local_ts = current_time
        
        # 添加买方深度
        events.append([
            BUY_EVENT | DEPTH_EVENT | EXCH_EVENT | LOCAL_EVENT,
            exch_ts, local_ts, bid_price, np.random.uniform(1.0, 5.0),
            0, 0, 0.0
        ])
        
        # 添加卖方深度
        events.append([
            SELL_EVENT | DEPTH_EVENT | EXCH_EVENT | LOCAL_EVENT,
            exch_ts, local_ts, ask_price, np.random.uniform(1.0, 5.0),
            0, 0, 0.0
        ])
    
    # 转换并保存
    events_array = np.array(events)
    sort_indices = np.argsort(events_array[:, 2])
    events_array = events_array[sort_indices]
    
    structured_events = np.zeros(len(events_array), dtype=[
        ('ev', 'u8'), ('exch_ts', 'i8'), ('local_ts', 'i8'),
        ('px', 'f8'), ('qty', 'f8'), ('order_id', 'u8'),
        ('ival', 'i8'), ('fval', 'f8')
    ])
    
    for i, field in enumerate(['ev', 'exch_ts', 'local_ts', 'px', 'qty', 'order_id', 'ival', 'fval']):
        structured_events[field] = events_array[:, i]
    
    os.makedirs("data", exist_ok=True)
    np.savez_compressed("data/better_btcusdt.npz", data=structured_events)
    
    print(f"✅ 生成了 {len(structured_events)} 个事件的更好数据")

if __name__ == "__main__":
    run_first_trading_strategy()