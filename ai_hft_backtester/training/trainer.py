"""Model training framework with integrated backtesting"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import h5py
from typing import Dict, List, Tuple, Optional, Callable
from datetime import datetime
import logging
import wandb
from tqdm import tqdm

from ..ai.models import LSTMPredictor, TransformerPredictor, MarketMakingPolicy
from ..core.backtester import Backtester
from ..strategies.ai_market_maker import AIMarketMaker


logger = logging.getLogger(__name__)


class MarketDataset(Dataset):
    """PyTorch dataset for market data"""
    
    def __init__(self, h5_path: str, transform: Optional[Callable] = None):
        """
        Initialize dataset
        
        Args:
            h5_path: Path to HDF5 file
            transform: Optional data transformation
        """
        self.h5_path = h5_path
        self.transform = transform
        
        # Load data info
        with h5py.File(h5_path, 'r') as f:
            self.length = len(f['features'])
            self.sequence_length = f.attrs['sequence_length']
            self.n_features = f.attrs['n_features']
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            features = f['features'][idx]
            labels = f['labels'][idx]
        
        if self.transform:
            features = self.transform(features)
        
        return torch.FloatTensor(features), torch.FloatTensor(labels)


class ModelTrainer:
    """Unified model training with backtesting validation"""
    
    def __init__(
        self,
        model_type: str = "lstm",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        experiment_name: str = None
    ):
        """
        Initialize trainer
        
        Args:
            model_type: Type of model to train
            device: Training device
            experiment_name: Name for experiment tracking
        """
        self.model_type = model_type
        self.device = device
        self.experiment_name = experiment_name or f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize tracking
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'backtest_sharpe': [],
            'backtest_return': []
        }
        
        # Model storage
        self.best_model_state = None
        self.best_val_loss = float('inf')
    
    def create_model(self, input_size: int, config: Dict) -> nn.Module:
        """
        Create model based on type
        
        Args:
            input_size: Number of input features
            config: Model configuration
        
        Returns:
            Model instance
        """
        if self.model_type == "lstm":
            model = LSTMPredictor(
                input_size=input_size,
                hidden_size=config.get('hidden_size', 128),
                num_layers=config.get('num_layers', 2),
                dropout=config.get('dropout', 0.2)
            )
        elif self.model_type == "transformer":
            model = TransformerPredictor(
                input_size=input_size,
                d_model=config.get('d_model', 256),
                n_heads=config.get('n_heads', 8),
                n_layers=config.get('n_layers', 4),
                dropout=config.get('dropout', 0.1)
            )
        elif self.model_type == "rl_policy":
            model = MarketMakingPolicy(
                state_size=input_size,
                hidden_size=config.get('hidden_size', 256),
                n_actions=config.get('n_actions', 25)
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return model.to(self.device)
    
    def train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module
    ) -> float:
        """
        Train for one epoch
        
        Args:
            model: Model to train
            train_loader: Training data loader
            optimizer: Optimizer
            criterion: Loss function
        
        Returns:
            Average training loss
        """
        model.train()
        total_loss = 0
        n_batches = 0
        
        for features, labels in tqdm(train_loader, desc="Training"):
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            if self.model_type == "transformer":
                outputs = model(features)
                # Use shortest horizon for training
                outputs = outputs['1s']
            else:
                outputs = model(features)
            
            # Calculate loss
            if len(labels.shape) == 3:  # Multi-horizon labels
                # Use first horizon for now
                loss = criterion(outputs, labels[:, 0, :])
            else:
                loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    def validate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> float:
        """
        Validate model
        
        Args:
            model: Model to validate
            val_loader: Validation data loader
            criterion: Loss function
        
        Returns:
            Average validation loss
        """
        model.eval()
        total_loss = 0
        n_batches = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                if self.model_type == "transformer":
                    outputs = model(features)
                    outputs = outputs['1s']
                else:
                    outputs = model(features)
                
                # Calculate loss
                if len(labels.shape) == 3:
                    loss = criterion(outputs, labels[:, 0, :])
                else:
                    loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                n_batches += 1
        
        return total_loss / n_batches
    
    def backtest_validation(
        self,
        model: nn.Module,
        backtester: Backtester,
        start_date: str,
        end_date: str,
        feature_config: Dict
    ) -> Dict[str, float]:
        """
        Validate model performance through backtesting
        
        Args:
            model: Trained model
            backtester: Backtester instance
            start_date: Backtest start date
            end_date: Backtest end date
            feature_config: Feature engineering configuration
        
        Returns:
            Backtest metrics
        """
        # Create strategy with trained model
        from ..ai.features import RealtimeFeatureEngine
        
        feature_engine = RealtimeFeatureEngine(feature_config)
        
        # For RL policy, we need the policy network
        if self.model_type == "rl_policy":
            policy_network = model
            # Create a dummy price predictor for now
            price_predictor = LSTMPredictor(
                input_size=feature_config['n_features'],
                hidden_size=64
            ).to(self.device)
        else:
            price_predictor = model
            # Create a simple policy network
            policy_network = MarketMakingPolicy(
                state_size=feature_config['n_features'] + 10,
                hidden_size=128
            ).to(self.device)
        
        strategy = AIMarketMaker(
            price_predictor=price_predictor,
            policy_network=policy_network,
            feature_engine=feature_engine,
            risk_params={
                'max_position': 0.5,
                'max_order_size': 0.05,
                'stop_loss': 0.001,
                'daily_loss_limit': 0.02
            },
            device=self.device
        )
        
        # Run backtest
        results = backtester.run(
            strategy=strategy,
            start_date=start_date,
            end_date=end_date,
            initial_capital=10000,
            progress_bar=False
        )
        
        return {
            'sharpe_ratio': results.metrics.get('sharpe_ratio', 0),
            'total_return': results.metrics.get('total_return', 0),
            'max_drawdown': results.metrics.get('max_drawdown', 0),
            'win_rate': results.metrics.get('win_rate', 0)
        }
    
    def train(
        self,
        train_path: str,
        val_path: str,
        model_config: Dict,
        training_config: Dict,
        backtester: Optional[Backtester] = None,
        backtest_dates: Optional[Tuple[str, str]] = None
    ):
        """
        Full training pipeline with backtesting validation
        
        Args:
            train_path: Path to training data
            val_path: Path to validation data
            model_config: Model configuration
            training_config: Training configuration
            backtester: Optional backtester for validation
            backtest_dates: Optional (start, end) dates for backtesting
        """
        # Initialize wandb
        if training_config.get('use_wandb', False):
            wandb.init(project="ai-hft-backtester", name=self.experiment_name)
            wandb.config.update({**model_config, **training_config})
        
        # Create datasets
        train_dataset = MarketDataset(train_path)
        val_dataset = MarketDataset(val_path)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=training_config.get('batch_size', 32),
            shuffle=True,
            num_workers=training_config.get('num_workers', 4)
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=training_config.get('batch_size', 32),
            shuffle=False,
            num_workers=training_config.get('num_workers', 4)
        )
        
        # Create model
        model = self.create_model(train_dataset.n_features, model_config)
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=training_config.get('learning_rate', 0.001),
            weight_decay=training_config.get('weight_decay', 1e-5)
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Training loop
        n_epochs = training_config.get('n_epochs', 100)
        
        for epoch in range(n_epochs):
            logger.info(f"Epoch {epoch+1}/{n_epochs}")
            
            # Train
            train_loss = self.train_epoch(model, train_loader, optimizer, criterion)
            
            # Validate
            val_loss = self.validate(model, val_loader, criterion)
            
            # Update scheduler
            scheduler.step(val_loss)
            
            # Log metrics
            self.metrics_history['train_loss'].append(train_loss)
            self.metrics_history['val_loss'].append(val_loss)
            
            logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Backtest validation
            if backtester and backtest_dates and epoch % 5 == 0:
                backtest_metrics = self.backtest_validation(
                    model,
                    backtester,
                    backtest_dates[0],
                    backtest_dates[1],
                    {'n_features': train_dataset.n_features, 'means': np.zeros(train_dataset.n_features), 'stds': np.ones(train_dataset.n_features)}
                )
                
                self.metrics_history['backtest_sharpe'].append(backtest_metrics['sharpe_ratio'])
                self.metrics_history['backtest_return'].append(backtest_metrics['total_return'])
                
                logger.info(f"Backtest Sharpe: {backtest_metrics['sharpe_ratio']:.4f}, "
                          f"Return: {backtest_metrics['total_return']*100:.2f}%")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = model.state_dict()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'metrics_history': self.metrics_history
                }, f"models/{self.experiment_name}_best.pth")
            
            # Log to wandb
            if training_config.get('use_wandb', False):
                log_dict = {
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'learning_rate': optimizer.param_groups[0]['lr']
                }
                
                if len(self.metrics_history['backtest_sharpe']) > 0:
                    log_dict['backtest_sharpe'] = self.metrics_history['backtest_sharpe'][-1]
                    log_dict['backtest_return'] = self.metrics_history['backtest_return'][-1]
                
                wandb.log(log_dict)
            
            # Early stopping
            if epoch > 20 and val_loss > min(self.metrics_history['val_loss'][-10:]):
                logger.info("Early stopping triggered")
                break
        
        # Load best model
        if self.best_model_state:
            model.load_state_dict(self.best_model_state)
        
        return model


class ReinforcementLearningTrainer:
    """Specialized trainer for RL models"""
    
    def __init__(
        self,
        environment,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize RL trainer
        
        Args:
            environment: Trading environment
            device: Training device
        """
        self.env = environment
        self.device = device
        
        # Training components
        self.policy_net = None
        self.target_net = None
        self.optimizer = None
        self.memory = []
        
    def train_dqn(
        self,
        n_episodes: int = 1000,
        batch_size: int = 32,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995
    ):
        """
        Train using Deep Q-Network
        
        Args:
            n_episodes: Number of training episodes
            batch_size: Batch size for experience replay
            gamma: Discount factor
            epsilon_start: Starting exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Exploration decay rate
        """
        epsilon = epsilon_start
        
        for episode in range(n_episodes):
            state = self.env.reset()
            total_reward = 0
            done = False
            
            while not done:
                # Select action
                if np.random.random() < epsilon:
                    action = self.env.action_space.sample()
                else:
                    with torch.no_grad():
                        q_values = self.policy_net(torch.FloatTensor(state).to(self.device))
                        action = q_values.argmax().item()
                
                # Take action
                next_state, reward, done, info = self.env.step(action)
                
                # Store experience
                self.memory.append((state, action, reward, next_state, done))
                
                # Sample and train
                if len(self.memory) > batch_size:
                    self._train_batch(batch_size, gamma)
                
                state = next_state
                total_reward += reward
            
            # Update epsilon
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            
            # Update target network
            if episode % 10 == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            logger.info(f"Episode {episode}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.4f}")
    
    def _train_batch(self, batch_size: int, gamma: float):
        """Train on a batch of experiences"""
        # Sample batch
        batch = np.random.choice(len(self.memory), batch_size, replace=False)
        
        states = torch.FloatTensor([self.memory[i][0] for i in batch]).to(self.device)
        actions = torch.LongTensor([self.memory[i][1] for i in batch]).to(self.device)
        rewards = torch.FloatTensor([self.memory[i][2] for i in batch]).to(self.device)
        next_states = torch.FloatTensor([self.memory[i][3] for i in batch]).to(self.device)
        dones = torch.FloatTensor([self.memory[i][4] for i in batch]).to(self.device)
        
        # Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + gamma * next_q * (1 - dones)
        
        # Calculate loss
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()