"""Example: Train model and run backtest with integrated pipeline"""

import sys
sys.path.append('..')

from ai_hft_backtester import Backtester
from ai_hft_backtester.ai.features import FeatureEngineer, RealtimeFeatureEngine
from ai_hft_backtester.ai.models import LSTMPredictor, TransformerPredictor, EnsemblePredictor
from ai_hft_backtester.training.data_generator import TrainingDataGenerator
from ai_hft_backtester.training.trainer import ModelTrainer
from ai_hft_backtester.training.online_learning import OnlineLearner, AdaptiveEnsemble
from ai_hft_backtester.training.evaluation import ModelEvaluator, WalkForwardAnalysis
from ai_hft_backtester.strategies.ai_market_maker import AIMarketMaker
from ai_hft_backtester.data.loader import DataLoader
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main training and backtesting pipeline"""
    
    # 1. Data Preparation Phase
    logger.info("=== Phase 1: Data Preparation ===")
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer(config={
        'window_sizes': [10, 30, 60, 300, 600]
    })
    
    # Initialize data generator
    data_generator = TrainingDataGenerator(
        feature_engineer=feature_engineer,
        label_config={
            'horizons': [10, 50, 300, 600],  # 1s, 5s, 30s, 1m
            'threshold': 0.0002  # 2 basis points
        }
    )
    
    # Load historical data (this would be real data in production)
    logger.info("Loading and processing historical data...")
    # data_loader = DataLoader("./historical_data")
    # orderbook_data, trade_data = data_loader.load_data("BTCUSDT", "2023-01-01", "2023-12-31")
    
    # Process data and generate features (simplified for example)
    # for idx, snapshot in orderbook_data.iterrows():
    #     orderbook = create_orderbook_from_snapshot(snapshot)
    #     data_generator.process_orderbook_snapshot(orderbook, snapshot['timestamp'], snapshot['mid_price'])
    
    # Generate labels and create dataset
    # data_generator.generate_labels()
    # dataset_paths = data_generator.create_training_dataset(
    #     output_path="./datasets/btcusdt_2023",
    #     sequence_length=100,
    #     train_ratio=0.7,
    #     val_ratio=0.15
    # )
    
    # For demo purposes, we'll use dummy paths
    dataset_paths = {
        'train': './datasets/btcusdt_2023_train.h5',
        'val': './datasets/btcusdt_2023_val.h5',
        'test': './datasets/btcusdt_2023_test.h5'
    }
    
    # 2. Model Training Phase
    logger.info("\n=== Phase 2: Model Training ===")
    
    # Train multiple models
    models = {}
    
    # LSTM Model
    logger.info("Training LSTM model...")
    lstm_trainer = ModelTrainer(model_type="lstm", experiment_name="lstm_btcusdt")
    lstm_model = lstm_trainer.train(
        train_path=dataset_paths['train'],
        val_path=dataset_paths['val'],
        model_config={
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.2
        },
        training_config={
            'n_epochs': 50,
            'batch_size': 32,
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'use_wandb': False
        }
    )
    models['lstm'] = lstm_model
    
    # Transformer Model
    logger.info("Training Transformer model...")
    transformer_trainer = ModelTrainer(model_type="transformer", experiment_name="transformer_btcusdt")
    transformer_model = transformer_trainer.train(
        train_path=dataset_paths['train'],
        val_path=dataset_paths['val'],
        model_config={
            'd_model': 256,
            'n_heads': 8,
            'n_layers': 4,
            'dropout': 0.1
        },
        training_config={
            'n_epochs': 50,
            'batch_size': 16,
            'learning_rate': 0.0005,
            'weight_decay': 1e-5,
            'use_wandb': False
        }
    )
    models['transformer'] = transformer_model
    
    # 3. Model Evaluation Phase
    logger.info("\n=== Phase 3: Model Evaluation ===")
    
    # Initialize backtester
    backtester = Backtester(
        symbol="BTCUSDT",
        exchange="binance",
        data_path="./historical_data",
        latency_config={'location': 'tokyo'}
    )
    
    # Evaluate models
    evaluator = ModelEvaluator(
        models=models,
        test_data_path=dataset_paths['test'],
        backtester=backtester
    )
    
    # Classification performance
    classification_results = evaluator.evaluate_classification_performance()
    logger.info("Classification Results:")
    for model_name, metrics in classification_results.items():
        logger.info(f"{model_name}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['macro_f1']:.4f}")
    
    # Backtesting performance
    backtest_results = evaluator.evaluate_backtesting_performance(
        start_date="2024-01-01",
        end_date="2024-01-31",
        initial_capital=10000,
        risk_params={
            'max_position': 1.0,
            'max_order_size': 0.1,
            'stop_loss': 0.002,
            'daily_loss_limit': 0.05
        }
    )
    
    logger.info("\nBacktest Results:")
    for model_name, result in backtest_results.items():
        metrics = result.metrics
        logger.info(f"{model_name}: Return={metrics['total_return']*100:.2f}%, "
                   f"Sharpe={metrics['sharpe_ratio']:.2f}, "
                   f"MaxDD={metrics['max_drawdown']*100:.2f}%")
    
    # Statistical properties
    statistical_results = evaluator.evaluate_statistical_properties()
    
    # Generate evaluation report
    evaluator.generate_evaluation_report("model_evaluation_report.html")
    
    # Select best model
    best_model_name, best_model = evaluator.select_best_model()
    logger.info(f"\nSelected best model: {best_model_name}")
    
    # 4. Walk-Forward Analysis
    logger.info("\n=== Phase 4: Walk-Forward Analysis ===")
    
    # This is computationally intensive, so we'll skip in the example
    # walk_forward = WalkForwardAnalysis(
    #     trainer=lstm_trainer,
    #     data_generator=data_generator,
    #     backtester=backtester,
    #     window_months=6,
    #     step_months=1
    # )
    # 
    # wf_results = walk_forward.run_analysis(
    #     start_date="2023-01-01",
    #     end_date="2023-12-31",
    #     model_config={'hidden_size': 128, 'num_layers': 2},
    #     training_config={'n_epochs': 20, 'batch_size': 32}
    # )
    
    # 5. Online Learning Setup
    logger.info("\n=== Phase 5: Online Learning Setup ===")
    
    # Initialize online learner
    online_learner = OnlineLearner(
        base_model=best_model,
        learning_rate=0.0001,
        buffer_size=10000,
        update_frequency=100
    )
    
    # Initialize adaptive ensemble
    adaptive_ensemble = AdaptiveEnsemble(
        models=models,
        window_size=1000,
        adaptation_rate=0.01
    )
    
    # 6. Production Backtest with Online Learning
    logger.info("\n=== Phase 6: Production Backtest ===")
    
    # Create feature engine for real-time use
    feature_config = {
        'n_features': 50,
        'means': np.zeros(50),  # Would be computed from training data
        'stds': np.ones(50),   # Would be computed from training data
        'window_sizes': [10, 30, 60]
    }
    realtime_feature_engine = RealtimeFeatureEngine(feature_config)
    
    # Create production strategy with ensemble
    from ai_hft_backtester.ai.models import MarketMakingPolicy
    
    policy = MarketMakingPolicy(
        state_size=60,
        hidden_size=256
    )
    
    production_strategy = AIMarketMaker(
        price_predictor=best_model,  # Or use adaptive_ensemble
        policy_network=policy,
        feature_engine=realtime_feature_engine,
        risk_params={
            'max_position': 1.0,
            'max_order_size': 0.1,
            'stop_loss': 0.002,
            'daily_loss_limit': 0.05
        }
    )
    
    # Run final backtest
    final_results = backtester.run(
        strategy=production_strategy,
        start_date="2024-02-01",
        end_date="2024-02-28",
        initial_capital=10000,
        commission_rate=0.0002
    )
    
    # Print final results
    final_results.print_statistics()
    
    # Plot performance
    final_results.plot_performance()
    
    # Export results
    final_results.to_csv("final_backtest_results")
    
    logger.info("\n=== Training and Backtesting Complete ===")
    

if __name__ == "__main__":
    import numpy as np  # Add this for the example
    
    # Note: This is a demonstration script. In production, you would:
    # 1. Use real historical data
    # 2. Implement proper data validation
    # 3. Add more sophisticated feature engineering
    # 4. Include transaction cost models
    # 5. Implement proper risk management
    # 6. Add monitoring and alerting
    # 7. Use distributed computing for large-scale training
    
    logger.warning("This is a demonstration script. Ensure you have real data before running.")
    # main()  # Uncomment when you have real data