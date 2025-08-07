# AI-HFT Backtester 用户指南

## 目录

1. [系统概述](#系统概述)
2. [快速开始](#快速开始)
3. [详细功能介绍](#详细功能介绍)
4. [数据准备](#数据准备)
5. [模型训练](#模型训练)
6. [策略开发](#策略开发)
7. [回测执行](#回测执行)
8. [性能优化](#性能优化)
9. [常见问题](#常见问题)
10. [API参考](#api参考)

## 系统概述

AI-HFT Backtester是一个专为高频交易策略设计的回测框架，特别针对Binance BTCUSDT交易对进行了优化。系统集成了深度学习模型训练、实时特征工程、延迟模拟和订单簿重建等核心功能。

### 核心特性

- **高性能计算**：使用Numba JIT编译优化关键计算路径
- **AI模型集成**：支持LSTM、Transformer、强化学习等多种模型
- **真实市场模拟**：包含延迟建模、订单队列位置、滑点等
- **在线学习**：支持模型的实时更新和自适应
- **完整的训练管道**：从数据处理到模型评估的端到端解决方案

### 系统架构

```
ai_hft_backtester/
├── core/           # 核心回测引擎
├── ai/             # AI模型和特征工程
├── training/       # 训练框架
├── strategies/     # 策略实现
├── data/          # 数据加载和管理
├── exchange/      # 交易所特定实现
├── tests/         # 单元测试
└── examples/      # 示例代码
```

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/your-repo/ai_hft_backtester.git
cd ai_hft_backtester

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行第一个回测

```python
from ai_hft_backtester import Backtester
from ai_hft_backtester.strategies import SimpleMarketMaker

# 初始化回测器
backtester = Backtester(
    symbol="BTCUSDT",
    exchange="binance",
    data_path="./historical_data"
)

# 创建策略
strategy = SimpleMarketMaker(
    spread=0.0002,  # 2个基点的价差
    order_size=0.1   # 每次下单0.1 BTC
)

# 初始化策略参数
strategy.initialize(
    initial_capital=10000,
    max_position=1.0,
    inventory_target=0.0
)

# 运行回测
results = backtester.run(
    strategy=strategy,
    start_date="2024-01-01",
    end_date="2024-01-31",
    initial_capital=10000
)

# 查看结果
results.print_statistics()
results.plot_performance()
```

## 详细功能介绍

### 1. 订单簿管理

系统支持完整的Level-2和Level-3订单簿重建：

```python
from ai_hft_backtester.core.orderbook import OrderBook

# 创建订单簿
orderbook = OrderBook("BTCUSDT")

# 更新订单簿
updates = [
    {'side': 'bid', 'price': 50000.0, 'quantity': 1.0},
    {'side': 'ask', 'price': 50001.0, 'quantity': 1.0}
]
orderbook.update(updates, timestamp=1234567890)

# 获取最优买卖价
best_bid, best_ask = orderbook.get_best_bid_ask()

# 提取特征用于AI模型
features = orderbook.get_features(depth=10)
```

### 2. 延迟模拟

真实的网络和处理延迟模拟：

```python
from ai_hft_backtester.core.latency import BinanceLatencyModel

# 创建延迟模型（基于服务器位置）
latency_model = BinanceLatencyModel(location="tokyo")

# 获取数据馈送延迟
feed_latency = latency_model.get_feed_latency()

# 获取订单延迟
send_latency, processing_latency = latency_model.get_order_latency(
    order_type=1,  # 限价单
    order_size=0.1,
    market_volatility=0.5,
    system_load=0.3
)
```

### 3. 特征工程

完整的特征提取框架：

```python
from ai_hft_backtester.ai.features import FeatureEngineer

# 初始化特征工程器
feature_engineer = FeatureEngineer(config={
    'window_sizes': [10, 30, 60, 300, 600]  # 不同时间窗口
})

# 提取特征
features = feature_engineer.extract_features(
    price_data=price_df,
    orderbook_data=orderbook_snapshots,
    trade_data=trades_df
)

# 创建训练序列
X, y = feature_engineer.create_training_data(
    features=features,
    labels=labels,
    sequence_length=100  # LSTM序列长度
)
```

## 数据准备

### 1. 数据格式

系统需要以下类型的数据：

#### 订单簿快照数据
```python
# DataFrame格式
orderbook_data = pd.DataFrame({
    'timestamp': [...],  # 时间戳（毫秒）
    'bids': [...],      # [[price, quantity], ...]
    'asks': [...],      # [[price, quantity], ...]
    'mid_price': [...]  # 中间价
})
```

#### 逐笔成交数据
```python
trade_data = pd.DataFrame({
    'timestamp': [...],  # 时间戳
    'price': [...],      # 成交价
    'quantity': [...],   # 成交量
    'side': [...]       # 'buy' 或 'sell'
})
```

### 2. 数据获取

#### 从Binance获取历史数据

```python
import ccxt
import pandas as pd

# 初始化交易所
exchange = ccxt.binance({
    'apiKey': 'your_api_key',
    'secret': 'your_secret'
})

# 获取K线数据
ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1m', limit=1000)
df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

# 获取订单簿快照
orderbook = exchange.fetch_order_book('BTC/USDT', limit=20)
```

### 3. 数据预处理

```python
from ai_hft_backtester.training.data_generator import TrainingDataGenerator

# 创建数据生成器
data_generator = TrainingDataGenerator(
    feature_engineer=feature_engineer,
    label_config={
        'horizons': [10, 50, 300, 600],  # 预测时间范围
        'threshold': 0.0002  # 价格变动阈值
    }
)

# 处理数据并生成标签
for snapshot in orderbook_snapshots:
    data_generator.process_orderbook_snapshot(
        orderbook=snapshot['orderbook'],
        timestamp=snapshot['timestamp'],
        mid_price=snapshot['mid_price']
    )

# 创建训练数据集
dataset_paths = data_generator.create_training_dataset(
    output_path="./datasets/btcusdt",
    sequence_length=100,
    train_ratio=0.7,
    val_ratio=0.15
)
```

## 模型训练

### 1. 训练LSTM价格预测模型

```python
from ai_hft_backtester.training.trainer import ModelTrainer

# 创建训练器
trainer = ModelTrainer(
    model_type="lstm",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# 配置模型
model_config = {
    'hidden_size': 128,
    'num_layers': 2,
    'dropout': 0.2
}

# 配置训练
training_config = {
    'n_epochs': 100,
    'batch_size': 32,
    'learning_rate': 0.001,
    'weight_decay': 1e-5
}

# 训练模型
model = trainer.train(
    train_path=dataset_paths['train'],
    val_path=dataset_paths['val'],
    model_config=model_config,
    training_config=training_config
)
```

### 2. 训练强化学习策略

```python
from ai_hft_backtester.training.trainer import ReinforcementLearningTrainer
from ai_hft_backtester.environments import TradingEnvironment

# 创建交易环境
env = TradingEnvironment(
    orderbook_data=orderbook_data,
    initial_balance=10000,
    commission=0.0002
)

# 创建RL训练器
rl_trainer = ReinforcementLearningTrainer(environment=env)

# 训练DQN
rl_trainer.train_dqn(
    n_episodes=1000,
    batch_size=32,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.01
)
```

### 3. 模型评估

```python
from ai_hft_backtester.training.evaluation import ModelEvaluator

# 创建评估器
evaluator = ModelEvaluator(
    models={'lstm': lstm_model, 'transformer': transformer_model},
    test_data_path=dataset_paths['test'],
    backtester=backtester
)

# 评估分类性能
classification_results = evaluator.evaluate_classification_performance()

# 评估回测性能
backtest_results = evaluator.evaluate_backtesting_performance(
    start_date="2024-02-01",
    end_date="2024-02-28"
)

# 生成评估报告
evaluator.generate_evaluation_report("evaluation_report.html")

# 选择最佳模型
best_model_name, best_model = evaluator.select_best_model()
```

## 策略开发

### 1. 创建自定义策略

```python
from ai_hft_backtester.strategies.base import BaseStrategy

class MyCustomStrategy(BaseStrategy):
    def __init__(self, model, risk_params):
        super().__init__()
        self.model = model
        self.risk_params = risk_params
        self.position = 0
        
    def on_orderbook_update(self, orderbook, timestamp):
        """处理订单簿更新"""
        # 获取特征
        features = self.extract_features(orderbook)
        
        # 模型预测
        prediction = self.model.predict(features)
        
        # 生成交易信号
        if prediction > 0.7:  # 强烈看涨
            return [{
                'action': 'place_order',
                'side': 'buy',
                'price': orderbook.best_bid * 0.9999,
                'quantity': 0.1,
                'order_type': 'limit'
            }]
        
        return []
    
    def on_trade(self, trade, timestamp):
        """处理成交"""
        if trade['side'] == 'buy':
            self.position += trade['quantity']
        else:
            self.position -= trade['quantity']
```

### 2. 使用AI驱动的做市策略

```python
from ai_hft_backtester.strategies import AIMarketMaker
from ai_hft_backtester.ai.features import RealtimeFeatureEngine

# 创建实时特征引擎
feature_engine = RealtimeFeatureEngine({
    'n_features': 50,
    'means': feature_means,  # 从训练数据计算
    'stds': feature_stds,
    'window_sizes': [10, 30, 60]
})

# 创建AI做市策略
strategy = AIMarketMaker(
    price_predictor=lstm_model,
    policy_network=rl_policy,
    feature_engine=feature_engine,
    risk_params={
        'max_position': 1.0,      # 最大持仓
        'max_order_size': 0.1,    # 最大单笔订单
        'stop_loss': 0.002,       # 止损线
        'daily_loss_limit': 0.05  # 每日损失限制
    }
)
```

## 回测执行

### 1. 基础回测

```python
# 运行回测
results = backtester.run(
    strategy=strategy,
    start_date="2024-01-01",
    end_date="2024-01-31",
    initial_capital=10000,
    commission_rate=0.0002,  # Binance Maker费率
    progress_bar=True
)

# 分析结果
print(f"总收益率: {results.metrics['total_return']*100:.2f}%")
print(f"夏普比率: {results.metrics['sharpe_ratio']:.2f}")
print(f"最大回撤: {results.metrics['max_drawdown']*100:.2f}%")
print(f"胜率: {results.metrics['win_rate']*100:.2f}%")
```

### 2. Walk-Forward分析

```python
from ai_hft_backtester.training.evaluation import WalkForwardAnalysis

# 创建Walk-Forward分析器
wf_analyzer = WalkForwardAnalysis(
    trainer=trainer,
    data_generator=data_generator,
    backtester=backtester,
    window_months=6,  # 训练窗口
    step_months=1     # 步进
)

# 运行分析
wf_results = wf_analyzer.run_analysis(
    start_date="2023-01-01",
    end_date="2023-12-31",
    model_config=model_config,
    training_config=training_config
)

# 查看结果
print(f"平均样本外夏普比率: {wf_results['sharpe_ratio'].mean():.2f}")
```

### 3. 在线学习回测

```python
from ai_hft_backtester.training.online_learning import ContinuousTrainingPipeline

# 创建持续训练管道
pipeline = ContinuousTrainingPipeline(
    base_model=best_model,
    feature_engineer=feature_engine,
    training_interval=3600,  # 每小时更新
    min_samples=1000
)

# 运行带在线学习的回测
async def run_online_backtest():
    for tick in market_data_stream:
        # 处理数据并获取预测
        prediction = await pipeline.process_tick(
            orderbook=tick['orderbook'],
            mid_price=tick['mid_price'],
            timestamp=tick['timestamp']
        )
        
        # 执行交易逻辑
        if prediction is not None:
            execute_trading_logic(prediction)
```

## 性能优化

### 1. Numba优化

关键计算函数已使用Numba JIT编译优化：

```python
from numba import jit

@jit(nopython=True)
def fast_calculation(data):
    """使用Numba加速的计算函数"""
    # 这里的代码会被编译成机器码
    result = 0.0
    for i in range(len(data)):
        result += data[i] ** 2
    return result
```

### 2. 并行处理

```python
# 并行运行多个策略回测
from concurrent.futures import ProcessPoolExecutor

def run_strategy_backtest(strategy_config):
    strategy = create_strategy(strategy_config)
    return backtester.run(strategy, **backtest_params)

# 并行执行
with ProcessPoolExecutor(max_workers=4) as executor:
    strategy_configs = [config1, config2, config3, config4]
    results = list(executor.map(run_strategy_backtest, strategy_configs))
```

### 3. 内存优化

```python
# 使用生成器处理大数据集
def process_large_dataset(file_path):
    with h5py.File(file_path, 'r') as f:
        for i in range(0, len(f['data']), 1000):
            batch = f['data'][i:i+1000]
            yield process_batch(batch)

# 分批处理避免内存溢出
for processed_batch in process_large_dataset('large_data.h5'):
    update_model(processed_batch)
```

## 常见问题

### Q1: 如何处理数据不足的问题？

A: 可以使用以下方法：
1. 数据增强：添加噪声、时间偏移等
2. 迁移学习：使用其他交易对的预训练模型
3. 合成数据：使用GAN生成合成订单簿数据

```python
# 数据增强示例
def augment_orderbook_data(orderbook, noise_level=0.0001):
    augmented = orderbook.copy()
    augmented.bids[:, 0] *= (1 + np.random.randn(len(augmented.bids)) * noise_level)
    augmented.asks[:, 0] *= (1 + np.random.randn(len(augmented.asks)) * noise_level)
    return augmented
```

### Q2: 如何避免过拟合？

A: 推荐以下措施：
1. 使用Walk-Forward分析验证策略
2. 添加正则化（L1/L2、Dropout）
3. 使用集成学习
4. 监控样本内外性能差异

```python
# 监控过拟合
in_sample_sharpe = calculate_sharpe(in_sample_returns)
out_sample_sharpe = calculate_sharpe(out_sample_returns)

if in_sample_sharpe - out_sample_sharpe > 0.5:
    print("警告：可能存在过拟合！")
```

### Q3: 如何处理延迟spike？

A: 系统已内置延迟模拟，但可以自定义：

```python
class CustomLatencyModel(LatencyModel):
    def get_feed_latency(self):
        # 添加随机延迟spike
        if np.random.random() < 0.01:  # 1%概率
            return self.feed_base_latency * 10  # 10倍延迟
        return super().get_feed_latency()
```

## API参考

### 核心类

#### Backtester
```python
class Backtester:
    def __init__(self, symbol, exchange, data_path, latency_config=None)
    def run(self, strategy, start_date, end_date, initial_capital, **kwargs)
```

#### OrderBook
```python
class OrderBook:
    def __init__(self, symbol)
    def update(self, updates, timestamp)
    def get_best_bid_ask(self)
    def get_mid_price(self)
    def get_features(self, depth=10)
```

#### AIMarketMaker
```python
class AIMarketMaker:
    def __init__(self, price_predictor, policy_network, feature_engine, risk_params)
    def on_orderbook_update(self, orderbook, timestamp)
    def on_trade(self, trade, timestamp)
    def get_performance_metrics(self)
```

### 工具函数

#### 特征计算
```python
calculate_price_features(prices, volumes, window_sizes)
calculate_microstructure_features(bid_prices, ask_prices, bid_volumes, ask_volumes, trades)
```

#### 数据处理
```python
create_training_dataset(features, labels, sequence_length)
normalize_features(features, method='standard')
```

## 下一步

1. **扩展数据源**：集成更多交易所的数据
2. **优化模型**：尝试更先进的模型架构（如GPT风格的Transformer）
3. **实盘集成**：将回测策略部署到实盘交易
4. **风险管理**：开发更复杂的风险管理模块
5. **可视化**：创建实时监控仪表板

## 联系和支持

- GitHub Issues: [项目Issues页面]
- 文档: [在线文档]
- 社区: [Discord/Telegram群组]

祝您交易愉快！🚀