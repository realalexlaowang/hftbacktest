import numpy as np
import os

# 定义事件常量（参考hftbacktest文档）
BUY_EVENT = 1
SELL_EVENT = 2  
DEPTH_EVENT = 1 << 31
TRADE_EVENT = 1 << 30
EXCH_EVENT = 1 << 29
LOCAL_EVENT = 1 << 28

def create_correct_data():
    """
    创建符合hftbacktest格式的8字段数据
    格式: [ev, exch_ts, local_ts, px, qty, order_id, ival, fval]
    """
    
    # 设置参数
    start_time = 1000000000000000000  # 纳秒时间戳
    duration = 60 * 60 * 1000000000  # 1小时的纳秒数
    tick_size = 0.1
    initial_price = 50000.0  # BTC价格约50000 USDT
    
    events = []
    current_time = start_time
    current_bid = initial_price - 5 * tick_size  # 买一价
    current_ask = initial_price + 5 * tick_size  # 卖一价
    
    print("正在生成符合hftbacktest格式的模拟数据...")
    
    # 生成模拟的订单簿数据
    for i in range(5000):  # 生成5000个事件
        current_time += np.random.randint(10000000, 100000000)  # 随机时间间隔 10-100ms
        
        if current_time > start_time + duration:
            break
            
        # 随机价格波动
        price_change = np.random.normal(0, 0.5) * tick_size
        current_bid += price_change
        current_ask += price_change
        
        # 确保买卖价差合理
        if current_ask - current_bid < tick_size:
            current_ask = current_bid + tick_size
        
        # 模拟延迟：exchange时间戳稍早于local时间戳
        exch_ts = current_time - np.random.randint(1000000, 10000000)  # 1-10ms前
        local_ts = current_time
        
        # 添加买方深度更新事件
        events.append([
            BUY_EVENT | DEPTH_EVENT | EXCH_EVENT | LOCAL_EVENT,  # ev
            exch_ts,        # exch_ts
            local_ts,       # local_ts  
            current_bid,    # px
            np.random.uniform(0.1, 2.0),  # qty
            0,              # order_id (对L2数据为0)
            0,              # ival
            0.0             # fval
        ])
        
        # 添加卖方深度更新事件
        events.append([
            SELL_EVENT | DEPTH_EVENT | EXCH_EVENT | LOCAL_EVENT,  # ev
            exch_ts,        # exch_ts
            local_ts,       # local_ts
            current_ask,    # px
            np.random.uniform(0.1, 2.0),  # qty
            0,              # order_id
            0,              # ival
            0.0             # fval
        ])
        
        # 偶尔添加成交事件
        if np.random.random() < 0.1:  # 10%概率
            trade_price = current_bid if np.random.random() < 0.5 else current_ask
            trade_side = BUY_EVENT if trade_price == current_ask else SELL_EVENT
            
            events.append([
                trade_side | TRADE_EVENT | EXCH_EVENT | LOCAL_EVENT,
                exch_ts,
                local_ts,
                trade_price,
                np.random.uniform(0.01, 0.5),
                0,
                0,
                0.0
            ])
    
    # 转换为numpy数组并排序
    events = np.array(events, dtype=[
        ('ev', 'u8'),
        ('exch_ts', 'i8'), 
        ('local_ts', 'i8'),
        ('px', 'f8'),
        ('qty', 'f8'),
        ('order_id', 'u8'),
        ('ival', 'i8'),
        ('fval', 'f8')
    ])
    
    # 按时间排序
    events = np.sort(events, order=['local_ts', 'exch_ts'])
    
    return events

def save_correct_data(events, filename):
    """保存数据为npz格式"""
    np.savez_compressed(filename, data=events)
    print(f"数据已保存到: {filename}")
    print(f"数据形状: {events.shape}")
    print(f"数据类型: {events.dtype}")
    print(f"时间范围: {events['exch_ts'][0]} - {events['exch_ts'][-1]}")
    print(f"价格范围: {events['px'].min():.1f} - {events['px'].max():.1f}")

if __name__ == "__main__":
    print("=== 创建正确格式的hftbacktest数据 ===")
    events = create_correct_data()
    
    # 创建数据目录
    os.makedirs("data", exist_ok=True)
    
    # 保存数据
    save_correct_data(events, "data/btcusdt_sample.npz")
    
    print("\n数据样本（前5行）:")
    print("事件类型, 交易所时间戳, 本地时间戳, 价格, 数量, 订单ID, ival, fval")
    for i in range(min(5, len(events))):
        e = events[i]
        print(f"{e['ev']}, {e['exch_ts']}, {e['local_ts']}, {e['px']:.1f}, {e['qty']:.3f}, {e['order_id']}, {e['ival']}, {e['fval']}")
        
    print(f"\n✅ 成功生成 {len(events)} 个事件")