"""Online learning and model adaptation framework"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Deque
from collections import deque
from datetime import datetime, timedelta
import logging
from threading import Lock
import asyncio

from ..ai.models import LSTMPredictor, TransformerPredictor
from ..ai.features import RealtimeFeatureEngine
from .data_generator import StreamingDataGenerator


logger = logging.getLogger(__name__)


class OnlineLearner:
    """Online learning system for continuous model improvement"""
    
    def __init__(
        self,
        base_model: nn.Module,
        learning_rate: float = 0.0001,
        buffer_size: int = 10000,
        batch_size: int = 32,
        update_frequency: int = 100,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize online learner
        
        Args:
            base_model: Pre-trained model to adapt
            learning_rate: Learning rate for online updates
            buffer_size: Size of experience buffer
            batch_size: Batch size for updates
            update_frequency: How often to update model
            device: Computing device
        """
        self.model = base_model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        
        # Create online version with smaller learning rate
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )
        
        # Experience buffer
        self.buffer_size = buffer_size
        self.feature_buffer: Deque = deque(maxlen=buffer_size)
        self.label_buffer: Deque = deque(maxlen=buffer_size)
        self.prediction_buffer: Deque = deque(maxlen=buffer_size)
        
        # Performance tracking
        self.performance_window = deque(maxlen=1000)
        self.update_count = 0
        self.lock = Lock()
        
        # Model versioning
        self.model_versions = []
        self.current_version = 0
        
    def add_experience(
        self,
        features: np.ndarray,
        prediction: np.ndarray,
        actual_outcome: int,
        timestamp: int
    ):
        """
        Add new experience to buffer
        
        Args:
            features: Input features
            prediction: Model prediction
            actual_outcome: Actual market outcome
            timestamp: Event timestamp
        """
        with self.lock:
            self.feature_buffer.append(features)
            self.label_buffer.append(actual_outcome)
            self.prediction_buffer.append(prediction)
            
            # Track performance
            predicted_class = np.argmax(prediction)
            correct = predicted_class == actual_outcome
            self.performance_window.append(correct)
            
            # Check if should update
            if len(self.feature_buffer) >= self.batch_size and \
               len(self.feature_buffer) % self.update_frequency == 0:
                self._update_model()
    
    def _update_model(self):
        """Perform online model update"""
        # Sample batch from buffer
        indices = np.random.choice(
            len(self.feature_buffer),
            min(self.batch_size, len(self.feature_buffer)),
            replace=False
        )
        
        # Prepare batch
        features = torch.FloatTensor(
            np.array([self.feature_buffer[i] for i in indices])
        ).to(self.device)
        
        labels = torch.LongTensor(
            np.array([self.label_buffer[i] for i in indices])
        ).to(self.device)
        
        # Forward pass
        self.model.train()
        outputs = self.model(features)
        
        # Handle multi-output models
        if isinstance(outputs, dict):
            outputs = outputs['1s']  # Use shortest horizon
        
        # Calculate loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        self.model.eval()
        
        self.update_count += 1
        
        # Log performance
        accuracy = sum(self.performance_window) / len(self.performance_window)
        logger.info(f"Online update {self.update_count}: Loss={loss.item():.4f}, "
                   f"Recent accuracy={accuracy:.4f}")
        
        # Save model version periodically
        if self.update_count % 100 == 0:
            self._save_model_version()
    
    def _save_model_version(self):
        """Save current model version"""
        version = {
            'version': self.current_version,
            'timestamp': datetime.now(),
            'update_count': self.update_count,
            'model_state': self.model.state_dict(),
            'performance': sum(self.performance_window) / len(self.performance_window)
        }
        
        self.model_versions.append(version)
        self.current_version += 1
        
        # Keep only recent versions
        if len(self.model_versions) > 10:
            self.model_versions.pop(0)
    
    def get_prediction(self, features: torch.Tensor) -> np.ndarray:
        """Get prediction with current model"""
        self.model.eval()
        with torch.no_grad():
            output = self.model(features.to(self.device))
            if isinstance(output, dict):
                output = output['1s']
            return torch.softmax(output, dim=-1).cpu().numpy()
    
    def rollback_model(self, version: int):
        """Rollback to previous model version"""
        for v in self.model_versions:
            if v['version'] == version:
                self.model.load_state_dict(v['model_state'])
                logger.info(f"Rolled back to model version {version}")
                return
        logger.warning(f"Version {version} not found")


class AdaptiveEnsemble:
    """Ensemble that adapts weights based on recent performance"""
    
    def __init__(
        self,
        models: Dict[str, nn.Module],
        window_size: int = 1000,
        adaptation_rate: float = 0.01,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize adaptive ensemble
        
        Args:
            models: Dictionary of model_name -> model
            window_size: Performance tracking window
            adaptation_rate: Weight update rate
            device: Computing device
        """
        self.models = {name: model.to(device) for name, model in models.items()}
        self.device = device
        self.window_size = window_size
        self.adaptation_rate = adaptation_rate
        
        # Initialize equal weights
        self.weights = {name: 1.0 / len(models) for name in models}
        
        # Performance tracking
        self.performance_history = {name: deque(maxlen=window_size) for name in models}
        self.ensemble_performance = deque(maxlen=window_size)
        
    def predict(self, features: torch.Tensor) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Get ensemble prediction
        
        Args:
            features: Input features
        
        Returns:
            Tuple of (ensemble_prediction, individual_predictions)
        """
        individual_predictions = {}
        weighted_sum = None
        
        for name, model in self.models.items():
            model.eval()
            with torch.no_grad():
                output = model(features.to(self.device))
                if isinstance(output, dict):
                    output = output['1s']
                
                probs = torch.softmax(output, dim=-1).cpu().numpy()
                individual_predictions[name] = probs
                
                if weighted_sum is None:
                    weighted_sum = self.weights[name] * probs
                else:
                    weighted_sum += self.weights[name] * probs
        
        return weighted_sum, individual_predictions
    
    def update_performance(
        self,
        individual_predictions: Dict[str, np.ndarray],
        ensemble_prediction: np.ndarray,
        actual_outcome: int
    ):
        """
        Update performance tracking and adapt weights
        
        Args:
            individual_predictions: Predictions from each model
            ensemble_prediction: Ensemble prediction
            actual_outcome: Actual market outcome
        """
        # Track individual model performance
        for name, pred in individual_predictions.items():
            predicted_class = np.argmax(pred)
            correct = predicted_class == actual_outcome
            self.performance_history[name].append(correct)
        
        # Track ensemble performance
        ensemble_class = np.argmax(ensemble_prediction)
        ensemble_correct = ensemble_class == actual_outcome
        self.ensemble_performance.append(ensemble_correct)
        
        # Adapt weights based on recent performance
        self._adapt_weights()
    
    def _adapt_weights(self):
        """Adapt ensemble weights based on recent performance"""
        if len(self.performance_history[list(self.models.keys())[0]]) < 100:
            return  # Not enough data
        
        # Calculate recent accuracy for each model
        accuracies = {}
        for name in self.models:
            accuracy = sum(self.performance_history[name]) / len(self.performance_history[name])
            accuracies[name] = accuracy
        
        # Update weights using softmax of accuracies
        max_acc = max(accuracies.values())
        exp_scores = {name: np.exp((acc - max_acc) / 0.1) for name, acc in accuracies.items()}
        sum_exp = sum(exp_scores.values())
        
        # Smooth weight updates
        for name in self.models:
            new_weight = exp_scores[name] / sum_exp
            self.weights[name] = (1 - self.adaptation_rate) * self.weights[name] + \
                                self.adaptation_rate * new_weight
        
        # Log weight updates
        logger.debug(f"Updated ensemble weights: {self.weights}")
    
    def get_model_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics for all models"""
        stats = {}
        
        for name in self.models:
            if len(self.performance_history[name]) > 0:
                accuracy = sum(self.performance_history[name]) / len(self.performance_history[name])
            else:
                accuracy = 0.0
            
            stats[name] = {
                'accuracy': accuracy,
                'weight': self.weights[name],
                'samples': len(self.performance_history[name])
            }
        
        # Add ensemble stats
        if len(self.ensemble_performance) > 0:
            ensemble_accuracy = sum(self.ensemble_performance) / len(self.ensemble_performance)
        else:
            ensemble_accuracy = 0.0
        
        stats['ensemble'] = {
            'accuracy': ensemble_accuracy,
            'weight': 1.0,
            'samples': len(self.ensemble_performance)
        }
        
        return stats


class ContinuousTrainingPipeline:
    """Pipeline for continuous model training and deployment"""
    
    def __init__(
        self,
        base_model: nn.Module,
        feature_engineer: RealtimeFeatureEngine,
        training_interval: int = 3600,  # 1 hour
        min_samples: int = 1000
    ):
        """
        Initialize continuous training pipeline
        
        Args:
            base_model: Initial model
            feature_engineer: Feature extraction engine
            training_interval: Seconds between training runs
            min_samples: Minimum samples before training
        """
        self.base_model = base_model
        self.feature_engineer = feature_engineer
        self.training_interval = training_interval
        self.min_samples = min_samples
        
        # Data collection
        self.data_generator = StreamingDataGenerator(
            feature_engineer,
            window_size=10000,
            update_frequency=100
        )
        
        # Model management
        self.production_model = base_model
        self.candidate_model = None
        self.model_lock = Lock()
        
        # Training state
        self.last_training_time = datetime.now()
        self.training_data_buffer = []
        self.is_training = False
        
    async def process_tick(
        self,
        orderbook,
        mid_price: float,
        timestamp: int
    ) -> Optional[np.ndarray]:
        """
        Process new market tick
        
        Args:
            orderbook: Current orderbook
            mid_price: Current mid price
            timestamp: Tick timestamp
        
        Returns:
            Model prediction if available
        """
        # Generate training data
        training_data = self.data_generator.process_tick(orderbook, mid_price, timestamp)
        
        if training_data:
            features, label = training_data
            self.training_data_buffer.append((features, label))
        
        # Check if should trigger training
        if self._should_train():
            asyncio.create_task(self._train_new_model())
        
        # Get prediction from production model
        with self.model_lock:
            features = orderbook.get_features()
            full_features = self.feature_engineer.get_features(features)
            
            return self._get_model_prediction(full_features)
    
    def _should_train(self) -> bool:
        """Check if should trigger new training"""
        if self.is_training:
            return False
        
        time_since_last = (datetime.now() - self.last_training_time).seconds
        
        return (time_since_last >= self.training_interval and 
                len(self.training_data_buffer) >= self.min_samples)
    
    async def _train_new_model(self):
        """Train new model in background"""
        self.is_training = True
        logger.info("Starting background model training")
        
        try:
            # Create training dataset
            X = np.array([x[0] for x in self.training_data_buffer])
            y = np.array([x[1] for x in self.training_data_buffer])
            
            # Clone current model
            import copy
            new_model = copy.deepcopy(self.production_model)
            
            # Train with smaller learning rate
            optimizer = torch.optim.Adam(new_model.parameters(), lr=0.0001)
            criterion = nn.CrossEntropyLoss()
            
            # Simple training loop
            new_model.train()
            for epoch in range(10):
                # Convert to tensors
                X_tensor = torch.FloatTensor(X)
                y_tensor = torch.LongTensor(y)
                
                # Forward pass
                outputs = new_model(X_tensor)
                if isinstance(outputs, dict):
                    outputs = outputs['1s']
                
                loss = criterion(outputs, y_tensor)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Validate new model
            new_model.eval()
            validation_accuracy = self._validate_model(new_model, X, y)
            
            # Deploy if better
            if validation_accuracy > 0.55:  # Simple threshold
                with self.model_lock:
                    self.candidate_model = self.production_model
                    self.production_model = new_model
                logger.info(f"Deployed new model with validation accuracy: {validation_accuracy:.4f}")
            
            # Clear old training data
            self.training_data_buffer = self.training_data_buffer[-self.min_samples:]
            self.last_training_time = datetime.now()
            
        except Exception as e:
            logger.error(f"Error in background training: {e}")
        finally:
            self.is_training = False
    
    def _validate_model(self, model: nn.Module, X: np.ndarray, y: np.ndarray) -> float:
        """Validate model performance"""
        model.eval()
        with torch.no_grad():
            outputs = model(torch.FloatTensor(X))
            if isinstance(outputs, dict):
                outputs = outputs['1s']
            
            predictions = torch.argmax(outputs, dim=1).numpy()
            accuracy = np.mean(predictions == y)
        
        return accuracy
    
    def _get_model_prediction(self, features: np.ndarray) -> np.ndarray:
        """Get prediction from production model"""
        self.production_model.eval()
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            output = self.production_model(features_tensor)
            
            if isinstance(output, dict):
                output = output['1s']
            
            return torch.softmax(output, dim=-1).cpu().numpy()[0]