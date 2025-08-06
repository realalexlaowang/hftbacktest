"""
AI交易决策引擎 - 集成多种机器学习算法进行交易决策
"""
import asyncio
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import joblib
from loguru import logger
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import talib
import pandas_ta as ta

from config.config import get_config
from data_pipeline.market_data_collector import MarketData


@dataclass
class TradingSignal:
    """交易信号结构"""
    symbol: str
    timestamp: int
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # 0-1之间的置信度
    predicted_price: float
    strategy_name: str
    features: Dict[str, float]
    risk_score: float


@dataclass
class MarketFeatures:
    """市场特征"""
    # 技术指标
    rsi: float
    macd: float
    macd_signal: float
    bollinger_upper: float
    bollinger_lower: float
    sma_20: float
    sma_50: float
    ema_12: float
    ema_26: float
    atr: float
    volume_ratio: float
    
    # 订单簿特征
    bid_ask_spread: float
    order_book_imbalance: float
    
    # 价格动量特征
    price_change_1m: float
    price_change_5m: float
    price_change_15m: float
    volatility: float


class LSTMPredictor(nn.Module):
    """LSTM价格预测模型"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        self.fc1 = nn.Linear(hidden_size, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # 预测价格
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)
        
        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]
        
        # 全连接层
        out = self.relu(self.fc1(last_output))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        
        return out


class FeatureExtractor:
    """特征提取器"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.price_history = {}  # 存储价格历史
        
    def extract_features(self, symbol: str, market_data: MarketData, 
                        orderbook_data: Optional[Any] = None) -> Optional[MarketFeatures]:
        """提取市场特征"""
        try:
            # 更新价格历史
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            
            self.price_history[symbol].append({
                'timestamp': market_data.timestamp,
                'open': market_data.open,
                'high': market_data.high,
                'low': market_data.low,
                'close': market_data.close,
                'volume': market_data.volume
            })
            
            # 保持最近200个数据点
            if len(self.price_history[symbol]) > 200:
                self.price_history[symbol] = self.price_history[symbol][-200:]
            
            # 需要至少50个数据点来计算技术指标
            if len(self.price_history[symbol]) < 50:
                return None
            
            # 转换为DataFrame
            df = pd.DataFrame(self.price_history[symbol])
            
            # 计算技术指标
            features = self._calculate_technical_indicators(df)
            
            # 计算订单簿特征
            if orderbook_data:
                orderbook_features = self._calculate_orderbook_features(orderbook_data)
                features.update(orderbook_features)
            else:
                features.update({
                    'bid_ask_spread': 0.0,
                    'order_book_imbalance': 0.0
                })
            
            # 计算价格动量特征
            momentum_features = self._calculate_momentum_features(df)
            features.update(momentum_features)
            
            return MarketFeatures(**features)
            
        except Exception as e:
            logger.error(f"特征提取错误: {e}")
            return None
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """计算技术指标"""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        # RSI
        rsi = talib.RSI(close, timeperiod=14)[-1]
        
        # MACD
        macd, macd_signal, _ = talib.MACD(close)
        
        # 布林带
        upper, middle, lower = talib.BBANDS(close, timeperiod=20)
        
        # 移动平均线
        sma_20 = talib.SMA(close, timeperiod=20)[-1]
        sma_50 = talib.SMA(close, timeperiod=50)[-1] if len(close) >= 50 else sma_20
        
        # 指数移动平均线
        ema_12 = talib.EMA(close, timeperiod=12)[-1]
        ema_26 = talib.EMA(close, timeperiod=26)[-1]
        
        # ATR (平均真实范围)
        atr = talib.ATR(high, low, close, timeperiod=14)[-1]
        
        # 成交量比率
        volume_ratio = volume[-1] / np.mean(volume[-20:]) if len(volume) >= 20 else 1.0
        
        return {
            'rsi': float(rsi) if not np.isnan(rsi) else 50.0,
            'macd': float(macd[-1]) if not np.isnan(macd[-1]) else 0.0,
            'macd_signal': float(macd_signal[-1]) if not np.isnan(macd_signal[-1]) else 0.0,
            'bollinger_upper': float(upper[-1]) if not np.isnan(upper[-1]) else close[-1],
            'bollinger_lower': float(lower[-1]) if not np.isnan(lower[-1]) else close[-1],
            'sma_20': float(sma_20) if not np.isnan(sma_20) else close[-1],
            'sma_50': float(sma_50) if not np.isnan(sma_50) else close[-1],
            'ema_12': float(ema_12) if not np.isnan(ema_12) else close[-1],
            'ema_26': float(ema_26) if not np.isnan(ema_26) else close[-1],
            'atr': float(atr) if not np.isnan(atr) else 0.0,
            'volume_ratio': float(volume_ratio)
        }
    
    def _calculate_orderbook_features(self, orderbook_data) -> Dict[str, float]:
        """计算订单簿特征"""
        try:
            # 买卖价差
            best_bid = orderbook_data.bids[0][0] if orderbook_data.bids else 0
            best_ask = orderbook_data.asks[0][0] if orderbook_data.asks else 0
            bid_ask_spread = (best_ask - best_bid) / best_ask if best_ask > 0 else 0
            
            # 订单簿不平衡度
            total_bid_volume = sum([bid[1] for bid in orderbook_data.bids[:5]])
            total_ask_volume = sum([ask[1] for ask in orderbook_data.asks[:5]])
            total_volume = total_bid_volume + total_ask_volume
            
            order_book_imbalance = (total_bid_volume - total_ask_volume) / total_volume if total_volume > 0 else 0
            
            return {
                'bid_ask_spread': float(bid_ask_spread),
                'order_book_imbalance': float(order_book_imbalance)
            }
            
        except Exception as e:
            logger.error(f"订单簿特征计算错误: {e}")
            return {'bid_ask_spread': 0.0, 'order_book_imbalance': 0.0}
    
    def _calculate_momentum_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """计算价格动量特征"""
        close = df['close'].values
        
        # 不同时间周期的价格变化
        price_change_1m = (close[-1] - close[-2]) / close[-2] if len(close) >= 2 else 0
        price_change_5m = (close[-1] - close[-6]) / close[-6] if len(close) >= 6 else 0
        price_change_15m = (close[-1] - close[-16]) / close[-16] if len(close) >= 16 else 0
        
        # 波动率 (20期标准差)
        volatility = np.std(close[-20:]) / np.mean(close[-20:]) if len(close) >= 20 else 0
        
        return {
            'price_change_1m': float(price_change_1m),
            'price_change_5m': float(price_change_5m),
            'price_change_15m': float(price_change_15m),
            'volatility': float(volatility)
        }


class TradingAI:
    """AI交易决策引擎"""
    
    def __init__(self):
        self.config = get_config()
        self.feature_extractor = FeatureExtractor()
        self.models = {}
        self.scalers = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化模型
        self._initialize_models()
        
    def _initialize_models(self):
        """初始化AI模型"""
        # LSTM价格预测模型
        self.models['lstm'] = LSTMPredictor(
            input_size=15,  # 特征数量
            hidden_size=128,
            num_layers=2
        ).to(self.device)
        
        # 随机森林分类器 (用于方向预测)
        self.models['rf_classifier'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # 标准化器
        self.scalers['lstm'] = StandardScaler()
        self.scalers['features'] = StandardScaler()
        
        logger.info("AI模型初始化完成")
    
    async def predict(self, symbol: str, market_data: MarketData, 
                     orderbook_data: Optional[Any] = None) -> Optional[TradingSignal]:
        """生成交易信号"""
        try:
            # 提取特征
            features = self.feature_extractor.extract_features(
                symbol, market_data, orderbook_data
            )
            
            if not features:
                return None
            
            # 转换特征为数组
            feature_array = self._features_to_array(features)
            
            # 多模型预测
            predictions = await self._multi_model_predict(feature_array, symbol)
            
            # 融合预测结果
            final_signal = self._ensemble_predictions(predictions, features, market_data)
            
            return final_signal
            
        except Exception as e:
            logger.error(f"AI预测错误: {e}")
            return None
    
    def _features_to_array(self, features: MarketFeatures) -> np.ndarray:
        """将特征转换为数组"""
        return np.array([
            features.rsi,
            features.macd,
            features.macd_signal,
            features.bollinger_upper,
            features.bollinger_lower,
            features.sma_20,
            features.sma_50,
            features.ema_12,
            features.ema_26,
            features.atr,
            features.volume_ratio,
            features.bid_ask_spread,
            features.order_book_imbalance,
            features.price_change_1m,
            features.price_change_5m,
            features.price_change_15m,
            features.volatility
        ])
    
    async def _multi_model_predict(self, features: np.ndarray, symbol: str) -> Dict[str, Any]:
        """多模型预测"""
        predictions = {}
        
        # LSTM价格预测
        try:
            lstm_input = torch.FloatTensor(features[-15:]).unsqueeze(0).unsqueeze(0).to(self.device)
            with torch.no_grad():
                lstm_pred = self.models['lstm'](lstm_input)
                predictions['lstm_price'] = float(lstm_pred.cpu().numpy()[0][0])
        except Exception as e:
            logger.warning(f"LSTM预测失败: {e}")
            predictions['lstm_price'] = 0.0
        
        # 技术指标策略
        predictions['technical'] = self._technical_analysis_strategy(features)
        
        # 动量策略
        predictions['momentum'] = self._momentum_strategy(features)
        
        # 均值回归策略
        predictions['mean_reversion'] = self._mean_reversion_strategy(features)
        
        return predictions
    
    def _technical_analysis_strategy(self, features: np.ndarray) -> Dict[str, Any]:
        """技术分析策略"""
        rsi = features[0]
        macd = features[1]
        macd_signal = features[2]
        
        signal = 0  # -1: SELL, 0: HOLD, 1: BUY
        confidence = 0.0
        
        # RSI策略
        if rsi < 30:  # 超卖
            signal += 1
            confidence += 0.3
        elif rsi > 70:  # 超买
            signal -= 1
            confidence += 0.3
        
        # MACD策略
        if macd > macd_signal:  # 黄金交叉
            signal += 1
            confidence += 0.4
        elif macd < macd_signal:  # 死亡交叉
            signal -= 1
            confidence += 0.4
        
        # 标准化信号
        if signal > 0:
            signal_type = 'BUY'
        elif signal < 0:
            signal_type = 'SELL'
        else:
            signal_type = 'HOLD'
        
        return {
            'signal': signal_type,
            'confidence': min(confidence, 1.0),
            'strategy': 'technical'
        }
    
    def _momentum_strategy(self, features: np.ndarray) -> Dict[str, Any]:
        """动量策略"""
        price_change_1m = features[13]
        price_change_5m = features[14]
        volume_ratio = features[10]
        
        signal = 0
        confidence = 0.0
        
        # 短期动量
        if price_change_1m > 0.002 and volume_ratio > 1.5:  # 强势突破
            signal += 2
            confidence += 0.6
        elif price_change_1m < -0.002 and volume_ratio > 1.5:  # 强势下跌
            signal -= 2
            confidence += 0.6
        
        # 中期动量
        if price_change_5m > 0.01:
            signal += 1
            confidence += 0.3
        elif price_change_5m < -0.01:
            signal -= 1
            confidence += 0.3
        
        if signal > 0:
            signal_type = 'BUY'
        elif signal < 0:
            signal_type = 'SELL'
        else:
            signal_type = 'HOLD'
        
        return {
            'signal': signal_type,
            'confidence': min(confidence, 1.0),
            'strategy': 'momentum'
        }
    
    def _mean_reversion_strategy(self, features: np.ndarray) -> Dict[str, Any]:
        """均值回归策略"""
        bollinger_upper = features[3]
        bollinger_lower = features[4]
        current_price = features[5]  # 使用SMA20作为当前价格代理
        
        signal_type = 'HOLD'
        confidence = 0.0
        
        # 布林带策略
        band_width = (bollinger_upper - bollinger_lower) / current_price
        
        if current_price <= bollinger_lower and band_width > 0.02:  # 触及下轨且带宽足够
            signal_type = 'BUY'
            confidence = 0.7
        elif current_price >= bollinger_upper and band_width > 0.02:  # 触及上轨且带宽足够
            signal_type = 'SELL'
            confidence = 0.7
        
        return {
            'signal': signal_type,
            'confidence': confidence,
            'strategy': 'mean_reversion'
        }
    
    def _ensemble_predictions(self, predictions: Dict[str, Any], 
                            features: MarketFeatures, 
                            market_data: MarketData) -> TradingSignal:
        """融合多个预测结果"""
        # 权重分配
        weights = {
            'technical': 0.3,
            'momentum': 0.4,
            'mean_reversion': 0.3
        }
        
        # 计算加权信号
        signal_scores = {'BUY': 0.0, 'SELL': 0.0, 'HOLD': 0.0}
        total_confidence = 0.0
        
        for strategy, weight in weights.items():
            if strategy in predictions:
                pred = predictions[strategy]
                signal_scores[pred['signal']] += weight * pred['confidence']
                total_confidence += weight * pred['confidence']
        
        # 确定最终信号
        final_signal = max(signal_scores, key=signal_scores.get)
        final_confidence = signal_scores[final_signal]
        
        # 置信度过滤
        if final_confidence < self.config.trading.prediction_threshold:
            final_signal = 'HOLD'
        
        # 预测价格 (简单移动平均)
        predicted_price = market_data.close
        if 'lstm_price' in predictions and predictions['lstm_price'] > 0:
            predicted_price = predictions['lstm_price']
        
        # 计算风险评分
        risk_score = self._calculate_risk_score(features)
        
        return TradingSignal(
            symbol=market_data.symbol,
            timestamp=market_data.timestamp,
            signal_type=final_signal,
            confidence=final_confidence,
            predicted_price=predicted_price,
            strategy_name='ensemble',
            features=features.__dict__,
            risk_score=risk_score
        )
    
    def _calculate_risk_score(self, features: MarketFeatures) -> float:
        """计算风险评分 (0-1, 越高风险越大)"""
        risk_factors = []
        
        # 波动率风险
        volatility_risk = min(features.volatility * 10, 1.0)
        risk_factors.append(volatility_risk * 0.4)
        
        # 流动性风险 (买卖价差)
        liquidity_risk = min(features.bid_ask_spread * 1000, 1.0)
        risk_factors.append(liquidity_risk * 0.3)
        
        # 市场极端情况风险
        extreme_rsi_risk = 0.0
        if features.rsi < 20 or features.rsi > 80:
            extreme_rsi_risk = 0.8
        risk_factors.append(extreme_rsi_risk * 0.3)
        
        return sum(risk_factors)
    
    async def update_models(self, symbol: str):
        """更新模型 (在生产环境中，这里应该实现在线学习)"""
        # 这里可以实现模型的增量更新逻辑
        # 例如收集最近的交易数据，重新训练模型
        pass
    
    def save_model(self, symbol: str, model_path: str):
        """保存模型"""
        try:
            torch.save(self.models['lstm'].state_dict(), f"{model_path}/lstm_{symbol}.pth")
            joblib.dump(self.models['rf_classifier'], f"{model_path}/rf_{symbol}.pkl")
            logger.info(f"模型已保存: {symbol}")
        except Exception as e:
            logger.error(f"模型保存失败: {e}")
    
    def load_model(self, symbol: str, model_path: str):
        """加载模型"""
        try:
            self.models['lstm'].load_state_dict(
                torch.load(f"{model_path}/lstm_{symbol}.pth", map_location=self.device)
            )
            self.models['rf_classifier'] = joblib.load(f"{model_path}/rf_{symbol}.pkl")
            logger.info(f"模型已加载: {symbol}")
        except Exception as e:
            logger.warning(f"模型加载失败: {e}")


# 使用示例
async def main():
    ai = TradingAI()
    
    # 模拟市场数据
    market_data = MarketData(
        symbol="BTCUSDT",
        timestamp=int(datetime.now().timestamp() * 1000),
        open=50000.0,
        high=50100.0,
        low=49900.0,
        close=50050.0,
        volume=1000.0
    )
    
    signal = await ai.predict("BTCUSDT", market_data)
    if signal:
        logger.info(f"交易信号: {signal}")


if __name__ == "__main__":
    asyncio.run(main())