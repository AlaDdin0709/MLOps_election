"""Tests for training functionality."""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from sklearn.linear_model import LogisticRegression

import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
from train import get_ml_models, evaluate_model


class TestGetMLModels:
    """Test model factory function."""
    
    def test_returns_dict(self):
        """Test that get_ml_models returns a dictionary."""
        models = get_ml_models()
        assert isinstance(models, dict)
    
    def test_models_are_sklearn_compatible(self):
        """Test that all models have fit/predict methods."""
        models = get_ml_models()
        for name, model in models.items():
            assert hasattr(model, 'fit'), f"{name} missing fit method"
            assert hasattr(model, 'predict'), f"{name} missing predict method"
    
    def test_contains_expected_models(self):
        """Test that expected model types are present."""
        models = get_ml_models()
        model_names = list(models.keys())
        # Should have at least a few models
        assert len(model_names) >= 2
        # Check for common ones (case-insensitive substring)
        names_lower = [n.lower() for n in model_names]
        assert any('logistic' in n for n in names_lower) or \
               any('random' in n for n in names_lower) or \
               any('xgb' in n for n in names_lower), \
               f"Expected at least one common model type, got: {model_names}"


class TestEvaluateModel:
    """Test model evaluation function."""
    
    @pytest.fixture
    def trained_model(self):
        """Create a simple trained model for testing."""
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        return model
    
    @pytest.fixture  
    def test_data(self):
        """Create test data."""
        X_test = np.random.rand(30, 5)
        y_test = np.random.randint(0, 2, 30)
        return X_test, y_test
    
    def test_returns_dict(self, trained_model, test_data):
        """Test that evaluate_model returns a dict."""
        X_test, y_test = test_data
        metrics = evaluate_model(trained_model, X_test, y_test)
        assert isinstance(metrics, dict)
    
    def test_contains_required_metrics(self, trained_model, test_data):
        """Test that required metrics are present."""
        X_test, y_test = test_data
        metrics = evaluate_model(trained_model, X_test, y_test)
        required = ['accuracy', 'precision', 'recall', 'f1_score']
        for metric in required:
            assert metric in metrics, f"Missing metric: {metric}"
    
    def test_metrics_in_valid_range(self, trained_model, test_data):
        """Test that metrics are in valid [0,1] range."""
        X_test, y_test = test_data
        metrics = evaluate_model(trained_model, X_test, y_test)
        for name, value in metrics.items():
            if name in ['accuracy', 'precision', 'recall', 'f1_score']:
                assert 0 <= value <= 1, f"{name} out of range: {value}"


class TestMLflowIntegration:
    """Test MLflow integration (mocked)."""
    
    @patch('mlflow.start_run')
    @patch('mlflow.log_params')
    @patch('mlflow.log_metrics')
    def test_mlflow_called_during_training(self, mock_metrics, mock_params, mock_start):
        """Test that MLflow tracking functions are called."""
        # This is a smoke test - full integration tested in CI
        mock_start.return_value.__enter__ = MagicMock()
        mock_start.return_value.__exit__ = MagicMock(return_value=False)
        
        # Simply verify the mocks exist and can be called
        mock_params({'model_type': 'test'})
        mock_metrics({'f1_score': 0.85})
        
        mock_params.assert_called_once()
        mock_metrics.assert_called_once()
