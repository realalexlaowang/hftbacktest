"""Model evaluation and selection framework"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Callable
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging

from ..core.backtester import Backtester, BacktestResults
from ..strategies.ai_market_maker import AIMarketMaker
from ..ai.features import RealtimeFeatureEngine


logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation framework"""
    
    def __init__(
        self,
        models: Dict[str, nn.Module],
        test_data_path: str,
        backtester: Backtester,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize evaluator
        
        Args:
            models: Dictionary of model_name -> model
            test_data_path: Path to test dataset
            backtester: Backtester instance
            device: Computing device
        """
        self.models = {name: model.to(device) for name, model in models.items()}
        self.test_data_path = test_data_path
        self.backtester = backtester
        self.device = device
        
        # Results storage
        self.classification_results = {}
        self.backtest_results = {}
        self.statistical_results = {}
    
    def evaluate_classification_performance(self) -> Dict[str, Dict]:
        """
        Evaluate classification metrics for all models
        
        Returns:
            Dictionary of model_name -> classification metrics
        """
        import h5py
        
        # Load test data
        with h5py.File(self.test_data_path, 'r') as f:
            X_test = f['features'][:]
            y_test = f['labels'][:]
        
        results = {}
        
        for name, model in self.models.items():
            logger.info(f"Evaluating classification performance for {name}")
            
            # Get predictions
            model.eval()
            predictions = []
            
            with torch.no_grad():
                for i in range(0, len(X_test), 32):  # Batch processing
                    batch = torch.FloatTensor(X_test[i:i+32]).to(self.device)
                    outputs = model(batch)
                    
                    if isinstance(outputs, dict):
                        outputs = outputs['1s']  # Use shortest horizon
                    
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    predictions.extend(preds)
            
            predictions = np.array(predictions)
            
            # Use first horizon labels
            if len(y_test.shape) == 3:
                y_true = np.argmax(y_test[:, 0, :], axis=1)
            else:
                y_true = y_test
            
            # Calculate metrics
            report = classification_report(
                y_true[:len(predictions)],
                predictions,
                output_dict=True,
                target_names=['Down', 'Neutral', 'Up']
            )
            
            cm = confusion_matrix(y_true[:len(predictions)], predictions)
            
            results[name] = {
                'classification_report': report,
                'confusion_matrix': cm,
                'accuracy': report['accuracy'],
                'macro_f1': report['macro avg']['f1-score'],
                'weighted_f1': report['weighted avg']['f1-score']
            }
            
            self.classification_results[name] = results[name]
        
        return results
    
    def evaluate_backtesting_performance(
        self,
        start_date: str,
        end_date: str,
        initial_capital: float = 10000,
        risk_params: Dict = None
    ) -> Dict[str, BacktestResults]:
        """
        Evaluate models through backtesting
        
        Args:
            start_date: Backtest start date
            end_date: Backtest end date
            initial_capital: Starting capital
            risk_params: Risk management parameters
        
        Returns:
            Dictionary of model_name -> backtest results
        """
        risk_params = risk_params or {
            'max_position': 1.0,
            'max_order_size': 0.1,
            'stop_loss': 0.002,
            'daily_loss_limit': 0.05
        }
        
        results = {}
        
        for name, model in self.models.items():
            logger.info(f"Running backtest for {name}")
            
            # Create strategy
            feature_config = {
                'n_features': 50,  # This should be loaded from model config
                'means': np.zeros(50),
                'stds': np.ones(50),
                'window_sizes': [10, 30, 60]
            }
            
            feature_engine = RealtimeFeatureEngine(feature_config)
            
            # For this evaluation, use the model as price predictor
            # and create a simple policy
            from ..ai.models import MarketMakingPolicy
            
            policy = MarketMakingPolicy(
                state_size=60,  # features + predictions + position state
                hidden_size=128
            ).to(self.device)
            
            strategy = AIMarketMaker(
                price_predictor=model,
                policy_network=policy,
                feature_engine=feature_engine,
                risk_params=risk_params,
                device=self.device
            )
            
            # Run backtest
            backtest_result = self.backtester.run(
                strategy=strategy,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                progress_bar=False
            )
            
            results[name] = backtest_result
            self.backtest_results[name] = backtest_result
        
        return results
    
    def evaluate_statistical_properties(self) -> Dict[str, Dict]:
        """
        Evaluate statistical properties of model predictions
        
        Returns:
            Dictionary of model_name -> statistical metrics
        """
        import h5py
        
        # Load test data
        with h5py.File(self.test_data_path, 'r') as f:
            X_test = f['features'][:1000]  # Sample for efficiency
        
        results = {}
        
        for name, model in self.models.items():
            logger.info(f"Evaluating statistical properties for {name}")
            
            model.eval()
            predictions = []
            confidences = []
            
            with torch.no_grad():
                for i in range(len(X_test)):
                    x = torch.FloatTensor(X_test[i:i+1]).to(self.device)
                    output = model(x)
                    
                    if isinstance(output, dict):
                        output = output['1s']
                    
                    probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                    pred_class = np.argmax(probs)
                    confidence = probs[pred_class]
                    
                    predictions.append(pred_class)
                    confidences.append(confidence)
            
            predictions = np.array(predictions)
            confidences = np.array(confidences)
            
            # Calculate statistics
            results[name] = {
                'prediction_distribution': {
                    'down': np.mean(predictions == 0),
                    'neutral': np.mean(predictions == 1),
                    'up': np.mean(predictions == 2)
                },
                'confidence_stats': {
                    'mean': np.mean(confidences),
                    'std': np.std(confidences),
                    'min': np.min(confidences),
                    'max': np.max(confidences),
                    'percentiles': {
                        '25': np.percentile(confidences, 25),
                        '50': np.percentile(confidences, 50),
                        '75': np.percentile(confidences, 75)
                    }
                },
                'prediction_entropy': self._calculate_entropy(predictions),
                'confidence_by_class': {
                    'down': np.mean(confidences[predictions == 0]) if np.any(predictions == 0) else 0,
                    'neutral': np.mean(confidences[predictions == 1]) if np.any(predictions == 1) else 0,
                    'up': np.mean(confidences[predictions == 2]) if np.any(predictions == 2) else 0
                }
            }
            
            self.statistical_results[name] = results[name]
        
        return results
    
    def _calculate_entropy(self, predictions: np.ndarray) -> float:
        """Calculate entropy of predictions"""
        _, counts = np.unique(predictions, return_counts=True)
        probs = counts / len(predictions)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        return entropy
    
    def generate_evaluation_report(self, output_path: str = "evaluation_report.html"):
        """
        Generate comprehensive evaluation report
        
        Args:
            output_path: Path to save report
        """
        import matplotlib.pyplot as plt
        from io import BytesIO
        import base64
        
        html_content = """
        <html>
        <head>
            <title>Model Evaluation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .metric { font-weight: bold; color: #2e86c1; }
                .section { margin: 30px 0; }
                h2 { color: #34495e; }
            </style>
        </head>
        <body>
            <h1>AI-HFT Model Evaluation Report</h1>
            <p>Generated on: {timestamp}</p>
        """
        
        html_content = html_content.format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        # Classification Performance Section
        html_content += "<div class='section'><h2>Classification Performance</h2>"
        html_content += "<table><tr><th>Model</th><th>Accuracy</th><th>Macro F1</th><th>Weighted F1</th></tr>"
        
        for name, metrics in self.classification_results.items():
            html_content += f"""
            <tr>
                <td>{name}</td>
                <td class='metric'>{metrics['accuracy']:.4f}</td>
                <td>{metrics['macro_f1']:.4f}</td>
                <td>{metrics['weighted_f1']:.4f}</td>
            </tr>
            """
        
        html_content += "</table></div>"
        
        # Backtesting Performance Section
        if self.backtest_results:
            html_content += "<div class='section'><h2>Backtesting Performance</h2>"
            html_content += "<table><tr><th>Model</th><th>Total Return</th><th>Sharpe Ratio</th><th>Max Drawdown</th><th>Win Rate</th></tr>"
            
            for name, result in self.backtest_results.items():
                metrics = result.metrics
                html_content += f"""
                <tr>
                    <td>{name}</td>
                    <td class='metric'>{metrics.get('total_return', 0)*100:.2f}%</td>
                    <td>{metrics.get('sharpe_ratio', 0):.2f}</td>
                    <td>{metrics.get('max_drawdown', 0)*100:.2f}%</td>
                    <td>{metrics.get('win_rate', 0)*100:.2f}%</td>
                </tr>
                """
            
            html_content += "</table></div>"
        
        # Statistical Properties Section
        if self.statistical_results:
            html_content += "<div class='section'><h2>Statistical Properties</h2>"
            
            for name, stats in self.statistical_results.items():
                html_content += f"<h3>{name}</h3>"
                html_content += "<table>"
                html_content += f"""
                <tr><td>Mean Confidence</td><td>{stats['confidence_stats']['mean']:.4f}</td></tr>
                <tr><td>Prediction Entropy</td><td>{stats['prediction_entropy']:.4f}</td></tr>
                <tr><td>Down Predictions</td><td>{stats['prediction_distribution']['down']*100:.1f}%</td></tr>
                <tr><td>Neutral Predictions</td><td>{stats['prediction_distribution']['neutral']*100:.1f}%</td></tr>
                <tr><td>Up Predictions</td><td>{stats['prediction_distribution']['up']*100:.1f}%</td></tr>
                """
                html_content += "</table>"
        
        html_content += "</body></html>"
        
        # Save report
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Evaluation report saved to {output_path}")
    
    def select_best_model(self, weights: Dict[str, float] = None) -> Tuple[str, nn.Module]:
        """
        Select best model based on weighted criteria
        
        Args:
            weights: Weights for different metrics
        
        Returns:
            Tuple of (model_name, model)
        """
        weights = weights or {
            'accuracy': 0.2,
            'sharpe_ratio': 0.4,
            'max_drawdown': 0.2,
            'win_rate': 0.2
        }
        
        scores = {}
        
        for name in self.models:
            score = 0
            
            # Classification score
            if name in self.classification_results:
                score += weights.get('accuracy', 0) * self.classification_results[name]['accuracy']
            
            # Backtesting scores
            if name in self.backtest_results:
                metrics = self.backtest_results[name].metrics
                
                # Normalize Sharpe ratio (cap at 3)
                sharpe = min(metrics.get('sharpe_ratio', 0), 3) / 3
                score += weights.get('sharpe_ratio', 0) * sharpe
                
                # Inverse drawdown (lower is better)
                drawdown = 1 - metrics.get('max_drawdown', 0)
                score += weights.get('max_drawdown', 0) * drawdown
                
                # Win rate
                score += weights.get('win_rate', 0) * metrics.get('win_rate', 0)
            
            scores[name] = score
        
        # Select best model
        best_name = max(scores, key=scores.get)
        logger.info(f"Selected best model: {best_name} with score {scores[best_name]:.4f}")
        
        return best_name, self.models[best_name]


class WalkForwardAnalysis:
    """Walk-forward analysis for robust model evaluation"""
    
    def __init__(
        self,
        trainer,
        data_generator,
        backtester: Backtester,
        window_months: int = 6,
        step_months: int = 1
    ):
        """
        Initialize walk-forward analysis
        
        Args:
            trainer: Model trainer instance
            data_generator: Training data generator
            backtester: Backtester instance
            window_months: Training window in months
            step_months: Step size in months
        """
        self.trainer = trainer
        self.data_generator = data_generator
        self.backtester = backtester
        self.window_months = window_months
        self.step_months = step_months
        
        self.results = []
    
    def run_analysis(
        self,
        start_date: str,
        end_date: str,
        model_config: Dict,
        training_config: Dict
    ) -> pd.DataFrame:
        """
        Run walk-forward analysis
        
        Args:
            start_date: Analysis start date
            end_date: Analysis end date
            model_config: Model configuration
            training_config: Training configuration
        
        Returns:
            DataFrame with analysis results
        """
        from dateutil.relativedelta import relativedelta
        
        current_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        while current_date + relativedelta(months=self.window_months + 1) <= end_dt:
            # Define periods
            train_start = current_date
            train_end = current_date + relativedelta(months=self.window_months)
            test_start = train_end
            test_end = test_start + relativedelta(months=self.step_months)
            
            logger.info(f"Walk-forward step: Train {train_start} to {train_end}, "
                       f"Test {test_start} to {test_end}")
            
            # Generate training data
            train_data = self.data_generator.generate_data(
                train_start.strftime("%Y-%m-%d"),
                train_end.strftime("%Y-%m-%d")
            )
            
            # Train model
            model = self.trainer.train(
                train_data['train'],
                train_data['val'],
                model_config,
                training_config
            )
            
            # Backtest on out-of-sample period
            result = self._evaluate_period(
                model,
                test_start.strftime("%Y-%m-%d"),
                test_end.strftime("%Y-%m-%d")
            )
            
            result['train_start'] = train_start
            result['train_end'] = train_end
            result['test_start'] = test_start
            result['test_end'] = test_end
            
            self.results.append(result)
            
            # Move forward
            current_date += relativedelta(months=self.step_months)
        
        # Create results DataFrame
        results_df = pd.DataFrame(self.results)
        
        # Calculate aggregate statistics
        logger.info(f"Walk-forward analysis complete. Average Sharpe: {results_df['sharpe_ratio'].mean():.4f}")
        
        return results_df
    
    def _evaluate_period(self, model: nn.Module, start_date: str, end_date: str) -> Dict:
        """Evaluate model on specific period"""
        # Create strategy
        from ..ai.features import RealtimeFeatureEngine
        from ..ai.models import MarketMakingPolicy
        
        feature_config = {
            'n_features': 50,
            'means': np.zeros(50),
            'stds': np.ones(50),
            'window_sizes': [10, 30, 60]
        }
        
        feature_engine = RealtimeFeatureEngine(feature_config)
        policy = MarketMakingPolicy(state_size=60, hidden_size=128)
        
        strategy = AIMarketMaker(
            price_predictor=model,
            policy_network=policy,
            feature_engine=feature_engine,
            risk_params={'max_position': 0.5, 'stop_loss': 0.002}
        )
        
        # Run backtest
        results = self.backtester.run(
            strategy=strategy,
            start_date=start_date,
            end_date=end_date,
            initial_capital=10000,
            progress_bar=False
        )
        
        return {
            'total_return': results.metrics.get('total_return', 0),
            'sharpe_ratio': results.metrics.get('sharpe_ratio', 0),
            'max_drawdown': results.metrics.get('max_drawdown', 0),
            'total_trades': results.metrics.get('total_trades', 0)
        }