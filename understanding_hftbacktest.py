"""
HftBacktest 学习指南 - 理解关键概念和参数
"""

def explain_asset_configuration():
    """解释BacktestAsset的配置参数"""
    
    print("=== HftBacktest 核心概念解释 ===\n")
    
    print("1. 🏗️ 资产配置 (BacktestAsset)")
    print("   .data([文件路径])          - 指定市场数据文件")
    print("   .linear_asset(1.0)        - 线性资产，1 BTC = 1.0 × 价格 USDT")
    print("   .tick_size(0.1)           - 最小价格变动：0.1 USDT")
    print("   .lot_size(0.001)          - 最小交易单位：0.001 BTC")
    print()
    
    print("2. ⏱️ 延迟模型")
    print("   .constant_order_latency(进单延迟, 响应延迟)")
    print("   - 进单延迟：从决策到订单到达交易所")
    print("   - 响应延迟：从交易所确认到接收响应")
    print("   - 单位：纳秒 (10,000,000 = 10毫秒)")
    print()
    
    print("3. 📊 队列位置模型")
    print("   .risk_adverse_queue_model()  - 风险规避模型")
    print("   .risk_neutral_queue_model()  - 风险中性模型")
    print("   - 决定订单在队列中的位置和成交概率")
    print()
    
    print("4. 💰 手续费模型")
    print("   .trading_value_fee_model(做市商费率, 吃单者费率)")
    print("   - 做市商费率：-0.00005 = -0.005% (返佣)")
    print("   - 吃单者费率：0.0007 = 0.07%")
    print()
    
    print("5. 📈 交易所模型")
    print("   .no_partial_fill_exchange()   - 不允许部分成交")
    print("   .partial_fill_exchange()      - 允许部分成交")
    print()

def explain_data_format():
    """解释数据格式"""
    
    print("=== 数据格式详解 ===\n")
    
    print("HftBacktest使用8字段的结构化数组:")
    print("📋 字段说明:")
    print("   ev       (u8) - 事件类型标志")
    print("   exch_ts  (i8) - 交易所时间戳 (纳秒)")
    print("   local_ts (i8) - 本地接收时间戳 (纳秒)")
    print("   px       (f8) - 价格")
    print("   qty      (f8) - 数量")
    print("   order_id (u8) - 订单ID (L3数据用)")
    print("   ival     (i8) - 保留字段")
    print("   fval     (f8) - 保留字段")
    print()
    
    print("🏷️ 事件类型标志 (ev字段):")
    print("   BUY_EVENT = 1                   - 买方事件")
    print("   SELL_EVENT = 2                  - 卖方事件") 
    print("   DEPTH_EVENT = 1 << 31           - 深度更新事件")
    print("   TRADE_EVENT = 1 << 30           - 成交事件")
    print("   EXCH_EVENT = 1 << 29            - 交易所事件")
    print("   LOCAL_EVENT = 1 << 28           - 本地事件")
    print()
    
    print("📊 组合示例:")
    print("   买方深度更新 = BUY_EVENT | DEPTH_EVENT | EXCH_EVENT | LOCAL_EVENT")
    print("   卖方成交     = SELL_EVENT | TRADE_EVENT | EXCH_EVENT | LOCAL_EVENT")
    print()

def explain_strategy_basics():
    """解释策略基础"""
    
    print("=== 策略编写基础 ===\n")
    
    print("🎯 核心API:")
    print("   hbt.elapse(纳秒)           - 时间前进")
    print("   hbt.depth(资产ID)          - 获取市场深度")
    print("   hbt.position(资产ID)       - 获取持仓")
    print("   hbt.current_timestamp      - 当前时间戳")
    print()
    
    print("📝 订单操作:")
    print("   hbt.submit_buy_order()")
    print("   hbt.submit_sell_order()")
    print("   hbt.cancel()")
    print("   hbt.clear_inactive_orders()")
    print()
    
    print("⚠️ 重要限制:")
    print("   - 必须使用@njit装饰器")
    print("   - 不能使用f-string格式化")
    print("   - 尽量使用简单的print语句")
    print()

def explain_market_data():
    """解释市场数据"""
    
    print("=== 市场数据理解 ===\n")
    
    print("📊 深度对象 (depth):")
    print("   depth.best_bid      - 最佳买价")
    print("   depth.best_ask      - 最佳卖价")
    print("   depth.best_bid_qty  - 最佳买价数量")
    print("   depth.best_ask_qty  - 最佳卖价数量")
    print("   depth.tick_size     - 最小价格变动")
    print("   depth.lot_size      - 最小交易单位")
    print()
    
    print("🧮 常用计算:")
    print("   中间价 = (best_bid + best_ask) / 2")
    print("   价差   = best_ask - best_bid")
    print("   相对价差 = 价差 / 中间价")
    print()

def main():
    """主函数"""
    explain_asset_configuration()
    print("\n" + "="*50 + "\n")
    
    explain_data_format()
    print("\n" + "="*50 + "\n")
    
    explain_strategy_basics()
    print("\n" + "="*50 + "\n")
    
    explain_market_data()
    
    print("\n🎓 下一步: 您现在可以开始编写简单的交易策略了！")
    print("💡 建议从简单的做市策略开始，逐步增加复杂性。")

if __name__ == "__main__":
    main()