"""
BTC专用AI交易策略
专门针对比特币特性优化的交易算法
"""
import asyncio
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import talib
from loguru import logger

from config.btc_only_config import get_btc_optimized_config
from ai_engine.trading_ai import TradingSignal, MarketFeatures, FeatureExtractor
from data_pipeline.market_data_collector import MarketData


@dataclass
class BTCMarketState:
    """BTC市场状态"""
    trend: str  # 'bullish', 'bearish', 'sideways'
    volatility_regime: str  # 'low', 'medium', 'high'
    volume_profile: str  # 'low', 'normal', 'high'
    market_session: str  # 'asia', 'europe', 'america', 'off_hours'
    fear_greed_index: float  # 0-100, 恐惧贪婪指数
    momentum_strength: float  # 0-1, 动量强度


class BTCTrendAnalyzer:
    """BTC趋势分析器"""
    
    def __init__(self):
        self.config = get_btc_optimized_config()['trading']
        
    def analyze_trend(self, df: pd.DataFrame) -> str:
        """分析BTC趋势"""
        if len(df) < 50:
            return 'sideways'
            
        close = df['close'].values
        
        # 短期和长期移动平均线
        sma_20 = talib.SMA(close, timeperiod=20)[-1]
        sma_50 = talib.SMA(close, timeperiod=50)[-1]
        current_price = close[-1]
        
        # 价格相对位置
        price_vs_sma20 = (current_price - sma_20) / sma_20
        sma_trend = (sma_20 - sma_50) / sma_50
        
        # 趋势判断
        if price_vs_sma20 > 0.02 and sma_trend > 0.01:
            return 'bullish'
        elif price_vs_sma20 < -0.02 and sma_trend < -0.01:
            return 'bearish'
        else:
            return 'sideways'
    
    def analyze_volatility_regime(self, df: pd.DataFrame) -> str:
        """分析波动率状态"""
        if len(df) < 20:
            return 'medium'
            
        close = df['close'].values
        returns = np.diff(close) / close[:-1]
        volatility = np.std(returns[-20:])  # 20期波动率
        
        if volatility < self.config.volatility_threshold_low:
            return 'low'
        elif volatility > self.config.volatility_threshold_high:
            return 'high'
        else:
            return 'medium'
    
    def analyze_volume_profile(self, df: pd.DataFrame) -> str:
        """分析成交量状态"""
        if len(df) < 20:
            return 'normal'
            
        volume = df['volume'].values
        avg_volume = np.mean(volume[-20:])
        current_volume = volume[-1]
        
        volume_ratio = current_volume / avg_volume
        
        if volume_ratio > self.config.volume_threshold:
            return 'high'
        elif volume_ratio < 0.7:
            return 'low'
        else:
            return 'normal'
    
    def get_market_session(self) -> str:
        """获取当前市场时段"""
        utc_hour = datetime.utcnow().hour
        
        if 0 <= utc_hour < 8:
            return 'asia'
        elif 8 <= utc_hour < 16:
            return 'europe'
        elif 16 <= utc_hour < 24:
            return 'america'
        else:
            return 'off_hours'


class BTCSpecialistAI:
    """BTC专业AI交易系统"""
    
    def __init__(self):
        self.config = get_btc_optimized_config()['trading']
        self.trend_analyzer = BTCTrendAnalyzer()
        self.feature_extractor = FeatureExtractor()
        
        # BTC特定策略权重
        self.strategy_weights = {
            'trend_following': 0.4,
            'mean_reversion': 0.25,
            'momentum': 0.2,
            'support_resistance': 0.15
        }
        
        # 历史价格数据缓存
        self.price_history = []
        
    async def predict(self, market_data: MarketData, 
                     orderbook_data: Optional[any] = None) -> Optional[TradingSignal]:
        """BTC专用预测"""
        try:
            # 更新价格历史
            self._update_price_history(market_data)
            
            if len(self.price_history) < 50:
                return None
            
            # 创建DataFrame
            df = pd.DataFrame(self.price_history)
            
            # 分析市场状态
            market_state = self._analyze_market_state(df)
            
            # 提取特征
            features = self.feature_extractor.extract_features("BTCUSDT", market_data, orderbook_data)
            if not features:
                return None
            
            # 多策略预测
            signals = await self._multi_strategy_predict(df, features, market_state)
            
            # 融合信号
            final_signal = self._ensemble_signals(signals, market_state, market_data)
            
            return final_signal
            
        except Exception as e:
            logger.error(f"BTC AI预测错误: {e}")
            return None
    
    def _update_price_history(self, market_data: MarketData):
        """更新价格历史"""
        self.price_history.append({
            'timestamp': market_data.timestamp,
            'open': market_data.open,
            'high': market_data.high,
            'low': market_data.low,
            'close': market_data.close,
            'volume': market_data.volume
        })
        
        # 保留最近168个数据点（7天的小时数据）
        if len(self.price_history) > self.config.lookback_periods:
            self.price_history = self.price_history[-self.config.lookback_periods:]
    
    def _analyze_market_state(self, df: pd.DataFrame) -> BTCMarketState:
        """分析BTC市场状态"""
        return BTCMarketState(
            trend=self.trend_analyzer.analyze_trend(df),
            volatility_regime=self.trend_analyzer.analyze_volatility_regime(df),
            volume_profile=self.trend_analyzer.analyze_volume_profile(df),
            market_session=self.trend_analyzer.get_market_session(),
            fear_greed_index=50.0,  # 可以接入外部API获取
            momentum_strength=self._calculate_momentum_strength(df)
        )
    
    def _calculate_momentum_strength(self, df: pd.DataFrame) -> float:
        """计算动量强度"""
        if len(df) < 20:
            return 0.5
            
        close = df['close'].values
        
        # RSI动量
        rsi = talib.RSI(close, timeperiod=14)[-1]
        rsi_momentum = abs(rsi - 50) / 50
        
        # 价格动量
        price_change_5 = (close[-1] - close[-6]) / close[-6] if len(close) >= 6 else 0
        price_momentum = min(abs(price_change_5) * 10, 1.0)
        
        # 成交量动量
        volume = df['volume'].values
        volume_ratio = volume[-1] / np.mean(volume[-20:]) if len(volume) >= 20 else 1
        volume_momentum = min((volume_ratio - 1) * 0.5, 1.0)
        
        return (rsi_momentum * 0.4 + price_momentum * 0.4 + volume_momentum * 0.2)
    
    async def _multi_strategy_predict(self, df: pd.DataFrame, features: MarketFeatures, 
                                    market_state: BTCMarketState) -> Dict[str, Dict]:
        """多策略预测"""
        signals = {}
        
        # 1. 趋势跟踪策略
        signals['trend_following'] = self._trend_following_strategy(df, features, market_state)
        
        # 2. 均值回归策略
        signals['mean_reversion'] = self._mean_reversion_strategy(df, features, market_state)
        
        # 3. 动量策略
        signals['momentum'] = self._momentum_strategy(df, features, market_state)
        
        # 4. 支撑阻力策略
        signals['support_resistance'] = self._support_resistance_strategy(df, features, market_state)
        
        return signals
    
    def _trend_following_strategy(self, df: pd.DataFrame, features: MarketFeatures, 
                                market_state: BTCMarketState) -> Dict:
        """趋势跟踪策略"""
        close = df['close'].values
        current_price = close[-1]
        
        signal_type = 'HOLD'
        confidence = 0.0
        
        # 基于移动平均线的趋势
        if features.sma_20 > features.sma_50 and current_price > features.sma_20:
            if market_state.trend == 'bullish' and market_state.momentum_strength > 0.6:
                signal_type = 'BUY'
                confidence = 0.8
        elif features.sma_20 < features.sma_50 and current_price < features.sma_20:
            if market_state.trend == 'bearish' and market_state.momentum_strength > 0.6:
                signal_type = 'SELL'
                confidence = 0.8
        
        # 调整基于市场状态
        if market_state.volatility_regime == 'high':
            confidence *= 0.8  # 高波动时降低confidence
        
        return {
            'signal': signal_type,
            'confidence': confidence,
            'strategy': 'trend_following'
        }
    
    def _mean_reversion_strategy(self, df: pd.DataFrame, features: MarketFeatures, 
                               market_state: BTCMarketState) -> Dict:
        """均值回归策略"""
        current_price = df['close'].values[-1]
        
        signal_type = 'HOLD'
        confidence = 0.0
        
        # 布林带均值回归
        bb_position = (current_price - features.bollinger_lower) / (features.bollinger_upper - features.bollinger_lower)
        
        if market_state.trend == 'sideways':  # 横盘市场更适合均值回归
            if bb_position < 0.2 and features.rsi < self.config.rsi_oversold:
                signal_type = 'BUY'
                confidence = 0.7
            elif bb_position > 0.8 and features.rsi > self.config.rsi_overbought:
                signal_type = 'SELL'
                confidence = 0.7
        
        # 支撑位反弹
        for level_name, level_price in self.config.price_levels.items():
            if 'support' in level_name:
                distance = abs(current_price - level_price) / level_price
                if distance < 0.02 and current_price > level_price:  # 接近支撑位且在上方
                    signal_type = 'BUY'
                    confidence = max(confidence, 0.6)
        
        return {
            'signal': signal_type,
            'confidence': confidence,
            'strategy': 'mean_reversion'
        }
    
    def _momentum_strategy(self, df: pd.DataFrame, features: MarketFeatures, 
                         market_state: BTCMarketState) -> Dict:
        """动量策略"""
        signal_type = 'HOLD'
        confidence = 0.0
        
        # 强动量突破
        if market_state.momentum_strength > 0.7:
            if features.macd > features.macd_signal and features.price_change_5m > 0.01:
                if market_state.volume_profile == 'high':
                    signal_type = 'BUY'
                    confidence = 0.75
            elif features.macd < features.macd_signal and features.price_change_5m < -0.01:
                if market_state.volume_profile == 'high':
                    signal_type = 'SELL'
                    confidence = 0.75
        
        # 基于交易时段调整
        if market_state.market_session in ['europe', 'america']:
            confidence *= 1.1  # 活跃时段提高confidence
        elif market_state.market_session == 'off_hours':
            confidence *= 0.7   # 非活跃时段降低confidence
        
        return {
            'signal': signal_type,
            'confidence': min(confidence, 1.0),
            'strategy': 'momentum'
        }
    
    def _support_resistance_strategy(self, df: pd.DataFrame, features: MarketFeatures, 
                                   market_state: BTCMarketState) -> Dict:
        """支撑阻力策略"""
        current_price = df['close'].values[-1]
        
        signal_type = 'HOLD'
        confidence = 0.0
        
        # 检查关键价格位
        for level_name, level_price in self.config.price_levels.items():
            distance_pct = abs(current_price - level_price) / level_price
            
            if distance_pct < 0.015:  # 1.5%范围内认为接近关键位
                if 'support' in level_name and current_price > level_price:
                    # 在支撑位上方，考虑买入
                    if features.rsi < 40 and market_state.volume_profile != 'low':
                        signal_type = 'BUY'
                        confidence = 0.6
                        
                elif 'resistance' in level_name and current_price < level_price:
                    # 在阻力位下方，考虑卖出
                    if features.rsi > 60 and market_state.volume_profile != 'low':
                        signal_type = 'SELL'
                        confidence = 0.6
        
        return {
            'signal': signal_type,
            'confidence': confidence,
            'strategy': 'support_resistance'
        }
    
    def _ensemble_signals(self, signals: Dict[str, Dict], market_state: BTCMarketState, 
                         market_data: MarketData) -> TradingSignal:
        """融合多个策略信号"""
        # 根据市场状态调整策略权重
        weights = self.strategy_weights.copy()
        
        if market_state.trend == 'bullish' or market_state.trend == 'bearish':
            weights['trend_following'] *= 1.3
            weights['mean_reversion'] *= 0.8
        elif market_state.trend == 'sideways':
            weights['mean_reversion'] *= 1.3
            weights['trend_following'] *= 0.8
            
        if market_state.volatility_regime == 'high':
            weights['momentum'] *= 1.2
            
        # 计算加权信号
        signal_scores = {'BUY': 0.0, 'SELL': 0.0, 'HOLD': 0.0}
        total_confidence = 0.0
        
        for strategy, weight in weights.items():
            if strategy in signals:
                signal_data = signals[strategy]
                signal_scores[signal_data['signal']] += weight * signal_data['confidence']
                total_confidence += weight * signal_data['confidence']
        
        # 确定最终信号
        final_signal = max(signal_scores, key=signal_scores.get)
        final_confidence = signal_scores[final_signal]
        
        # 应用置信度过滤
        if final_confidence < self.config.prediction_threshold:
            final_signal = 'HOLD'
        
        # 计算风险评分
        risk_score = self._calculate_btc_risk_score(market_state, market_data)
        
        # 预测价格 (简化版本)
        predicted_price = market_data.close
        if final_signal == 'BUY':
            predicted_price *= 1.02  # 预期2%上涨
        elif final_signal == 'SELL':
            predicted_price *= 0.98  # 预期2%下跌
        
        return TradingSignal(
            symbol="BTCUSDT",
            timestamp=market_data.timestamp,
            signal_type=final_signal,
            confidence=final_confidence,
            predicted_price=predicted_price,
            strategy_name='btc_specialist',
            features={
                'market_state': market_state.__dict__,
                'strategy_signals': signals
            },
            risk_score=risk_score
        )
    
    def _calculate_btc_risk_score(self, market_state: BTCMarketState, 
                                 market_data: MarketData) -> float:
        """计算BTC特定风险评分"""
        risk_factors = []
        
        # 波动率风险
        if market_state.volatility_regime == 'high':
            risk_factors.append(0.3)
        elif market_state.volatility_regime == 'low':
            risk_factors.append(0.1)
        else:
            risk_factors.append(0.2)
        
        # 流动性风险
        if market_state.volume_profile == 'low':
            risk_factors.append(0.25)
        else:
            risk_factors.append(0.1)
        
        # 时段风险
        if market_state.market_session == 'off_hours':
            risk_factors.append(0.2)
        else:
            risk_factors.append(0.05)
        
        # 趋势风险
        if market_state.trend == 'sideways':
            risk_factors.append(0.15)  # 横盘市场风险较高
        else:
            risk_factors.append(0.1)
        
        return min(sum(risk_factors), 1.0)


# 使用示例
async def main():
    btc_ai = BTCSpecialistAI()
    
    # 模拟BTC市场数据
    market_data = MarketData(
        symbol="BTCUSDT",
        timestamp=int(datetime.now().timestamp() * 1000),
        open=45000.0,
        high=45200.0,
        low=44800.0,
        close=45100.0,
        volume=1000.0
    )
    
    signal = await btc_ai.predict(market_data)
    if signal:
        logger.info(f"BTC交易信号: {signal}")


if __name__ == "__main__":
    asyncio.run(main())