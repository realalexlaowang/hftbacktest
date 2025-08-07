# AI-Powered High-Frequency Trading Backtester for Binance BTCUSDT

## Overview

This is an advanced backtesting framework specifically designed for high-frequency trading strategies on Binance BTCUSDT. It combines state-of-the-art AI models with accurate market simulation to provide realistic backtesting results.

## Key Features

- **AI-Powered Trading Signals**: Integration of deep learning and reinforcement learning models
- **Numba JIT Optimization**: High-performance tick-by-tick simulation
- **Full Order Book Reconstruction**: Level-2 and Level-3 market data support
- **Realistic Latency Modeling**: Feed and order execution latency simulation
- **Queue Position Modeling**: Accurate order fill probability estimation
- **Binance-Specific Features**: Optimized for BTCUSDT trading pair

## Architecture

```
ai_hft_backtester/
├── core/                  # Numba-optimized core simulation engine
├── data/                  # Data pipeline and management
├── ai/                    # AI models and feature engineering
├── strategies/            # Trading strategy implementations
├── exchange/              # Binance-specific implementations
├── analytics/             # Performance analysis and reporting
└── tests/                 # Unit and integration tests
```

## Requirements

- Python 3.8+
- Numba
- NumPy
- Pandas
- PyTorch/TensorFlow
- ccxt (for Binance API)

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from ai_hft_backtester import Backtester
from ai_hft_backtester.strategies import AIMarketMaker

# Initialize backtester
backtester = Backtester(
    symbol="BTCUSDT",
    exchange="binance",
    data_path="./historical_data/"
)

# Load AI strategy
strategy = AIMarketMaker(
    model_path="./models/lstm_market_maker.pth",
    risk_params={"max_position": 1.0, "stop_loss": 0.002}
)

# Run backtest
results = backtester.run(
    strategy=strategy,
    start_date="2024-01-01",
    end_date="2024-01-31",
    initial_capital=10000
)

# Analyze results
results.plot_performance()
results.print_statistics()
```