# AI-HFT Backtester 🚀

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> 专业级AI驱动的高频交易回测系统，专注于加密货币市场

<p align="center">
  <img src="https://img.shields.io/badge/Binance-BTCUSDT-yellow" alt="Binance BTCUSDT">
  <img src="https://img.shields.io/badge/Latency-Microsecond-green" alt="Microsecond Latency">
  <img src="https://img.shields.io/badge/AI-PyTorch-red" alt="PyTorch">
  <img src="https://img.shields.io/badge/Performance-Numba-blue" alt="Numba JIT">
</p>

## 🌟 特性

- **高保真市场模拟**：逐笔重建订单簿，考虑延迟、滑点和队列位置
- **AI模型集成**：支持深度学习（LSTM/Transformer）和强化学习（DQN/PPO）
- **性能优化**：使用Numba JIT编译，实现微秒级延迟模拟
- **完整的训练框架**：从数据生成到模型训练到策略回测的端到端解决方案
- **专业的风险管理**：仓位控制、止损、资金管理等
- **丰富的文档**：包含菜鸟指南、用户手册和API文档

## 🚀 快速开始

### 1. 安装

```bash
# 克隆仓库
git clone https://github.com/ai-hft-backtester/ai-hft-backtester.git
cd ai-hft-backtester

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行快速开始向导

```bash
python quickstart.py
```

### 3. 第一个回测

```python
from ai_hft_backtester import Backtester
from ai_hft_backtester.strategies import SimpleMarketMaker

# 创建回测器
backtester = Backtester(
    symbol="BTCUSDT",
    data_path="./sample_data"
)

# 创建策略
strategy = SimpleMarketMaker(spread=0.0002, order_size=0.01)
strategy.initialize(initial_capital=10000)

# 运行回测
results = backtester.run(
    strategy=strategy,
    start_date="2024-01-01",
    end_date="2024-01-02"
)

# 查看结果
results.print_statistics()
```

## 📚 文档

- 🆕 [菜鸟指南](BEGINNER_GUIDE.md) - 适合新手的详细教程
- 📖 [用户手册](USER_GUIDE.md) - 完整的功能说明
- 🗺️ [发展蓝图](ROADMAP.md) - 项目未来规划
- 🔧 [API文档](docs/api.md) - 详细的API参考

## 🏗️ 系统架构

```
ai_hft_backtester/
├── core/               # 核心回测引擎
│   ├── orderbook.py   # 订单簿管理
│   ├── engine.py      # 仿真引擎
│   ├── latency.py     # 延迟模型
│   └── backtester.py  # 回测主框架
├── ai/                # AI模型
│   ├── features.py    # 特征工程
│   └── models.py      # 深度学习模型
├── strategies/        # 交易策略
│   ├── base.py       # 基础策略类
│   └── ai_market_maker.py  # AI做市商
├── training/          # 训练框架
│   ├── trainer.py     # 模型训练
│   └── evaluation.py  # 模型评估
└── environments/      # RL环境
    └── trading_env.py # 交易环境
```

## 🎯 主要功能

### 1. 市场模拟
- Level-2/Level-3 订单簿重建
- 真实的延迟和滑点模拟
- 订单队列位置追踪
- 多交易所支持（开发中）

### 2. AI功能
- **特征工程**：价格、成交量、订单流等50+特征
- **预测模型**：LSTM、Transformer、CNN
- **强化学习**：DQN、PPO、多智能体
- **在线学习**：实时模型更新

### 3. 策略开发
- 灵活的策略框架
- 内置策略模板
- 实时风险管理
- 性能分析工具

## 📊 性能指标

| 指标 | 数值 |
|------|------|
| 事件处理速度 | 100K+/秒 |
| 延迟精度 | 微秒级 |
| 内存占用 | <8GB/天数据 |
| 回测准确度 | 99%+ |

## 🔬 研究案例

### 使用LSTM预测价格
```python
from ai_hft_backtester.ai.models import LSTMPredictor
from ai_hft_backtester.training import ModelTrainer

# 训练LSTM模型
trainer = ModelTrainer(model_type="lstm")
trainer.train(
    train_path="data/train.h5",
    val_path="data/val.h5",
    model_config={"hidden_size": 128, "num_layers": 2}
)
```

### 强化学习交易
```python
from ai_hft_backtester.environments import TradingEnvironment
from ai_hft_backtester.training import ReinforcementLearningTrainer

# 创建交易环境
env = TradingEnvironment(data_loader, initial_balance=10000)

# 训练DQN智能体
trainer = ReinforcementLearningTrainer(env)
trainer.train_dqn(n_episodes=1000)
```

## 🤝 贡献

我们欢迎各种形式的贡献！

1. 🐛 报告Bug
2. 💡 提出新功能
3. 📝 改进文档
4. 🔧 提交代码

详见 [贡献指南](CONTRIBUTING.md)

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- [hftbacktest](https://github.com/nkaz001/hftbacktest) - 灵感来源
- [PyTorch](https://pytorch.org/) - 深度学习框架
- [Numba](https://numba.pydata.org/) - JIT编译器
- 所有贡献者和支持者

## 📞 联系我们

- 📧 Email: team@ai-hft-backtester.com
- 💬 Discord: [加入我们的社区](https://discord.gg/ai-hft)
- 🐦 Twitter: [@AIHFTBacktester](https://twitter.com/AIHFTBacktester)

---

<p align="center">
  <b>⭐ 如果这个项目对你有帮助，请给我们一个Star！⭐</b>
</p>

<p align="center">
  Made with ❤️ by AI HFT Team
</p>