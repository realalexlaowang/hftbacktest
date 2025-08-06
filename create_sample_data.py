import numpy as np
import os

def create_sample_data():
    """
    创建简单的模拟市场数据用于学习hftbacktest
    数据格式: [timestamp, event_type, side, price, qty]
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
    
    # 生成模拟的订单簿数据
    for i in range(10000):  # 生成10000个事件
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
            
        # 添加买一价更新事件 (event_type=1表示深度更新)
        events.append([
            current_time,  # timestamp
            1,            # event_type: 1=深度更新
            1,            # side: 1=买方
            current_bid,  # price
            np.random.uniform(0.1, 2.0)  # quantity
        ])
        
        # 添加卖一价更新事件
        events.append([
            current_time,  # timestamp
            1,            # event_type: 1=深度更新
            -1,           # side: -1=卖方
            current_ask,  # price
            np.random.uniform(0.1, 2.0)  # quantity
        ])
        
        # 偶尔添加成交事件 (event_type=2)
        if np.random.random() < 0.1:  # 10%概率
            trade_price = current_bid if np.random.random() < 0.5 else current_ask
            events.append([
                current_time,
                2,  # event_type: 2=成交
                1 if trade_price == current_bid else -1,  # 成交方向
                trade_price,
                np.random.uniform(0.01, 0.5)
            ])
    
    # 转换为numpy数组并排序
    events = np.array(events)
    events = events[events[:, 0].argsort()]  # 按时间排序
    
    return events

def save_data(events, filename):
    """保存数据为npz格式"""
    np.savez_compressed(filename, data=events)
    print(f"数据已保存到: {filename}")
    print(f"数据形状: {events.shape}")
    print(f"时间范围: {events[0, 0]:.0f} - {events[-1, 0]:.0f}")
    print(f"价格范围: {events[:, 3].min():.1f} - {events[:, 3].max():.1f}")

if __name__ == "__main__":
    print("正在生成模拟市场数据...")
    events = create_sample_data()
    
    # 创建数据目录
    os.makedirs("data", exist_ok=True)
    
    # 保存数据
    save_data(events, "data/sample_btcusdt.npz")
    
    print("\n数据样本（前10行）:")
    print("时间戳, 事件类型, 方向, 价格, 数量")
    for i in range(min(10, len(events))):
        print(f"{events[i, 0]:.0f}, {events[i, 1]:.0f}, {events[i, 2]:.0f}, {events[i, 3]:.1f}, {events[i, 4]:.3f}")