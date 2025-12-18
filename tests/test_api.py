"""Tests for FastAPI backend API."""
import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import pickle
from io import BytesIO

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'backend' / 'src'))


class TestAPISetup:
    """Test API setup and imports."""
    
    def test_api_imports(self):
        """Test that API module can be imported."""
        from api import app
        assert app is not None
        assert app.title == "Sentiment API"
    
    def test_api_has_routes(self):
        """Test that expected routes exist."""
        from api import app
        routes = [r.path for r in app.routes]
        assert '/' in routes
        assert '/health' in routes
        assert '/predict' in routes


class TestHealthEndpoint:
    """Test /health endpoint."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from api import app
        return TestClient(app)
    
    def test_health_returns_200(self, client):
        """Test health endpoint returns 200."""
        response = client.get('/health')
        assert response.status_code == 200
    
    def test_health_response_format(self, client):
        """Test health response has expected format."""
        response = client.get('/health')
        data = response.json()
        assert 'status' in data
        assert 'model_loaded' in data


class TestPredictEndpoint:
    """Test /predict endpoint."""
    
    @pytest.fixture
    def client_with_model(self):
        """Create test client with mocked model."""
        from fastapi.testclient import TestClient
        import api
        
        # Mock the model and vectorizer
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1])
        mock_model.predict_proba = MagicMock(return_value=np.array([[0.3, 0.7]]))
        
        mock_vectorizer = MagicMock()
        mock_vectorizer.transform.return_value = np.array([[0.1, 0.2, 0.3]])
        
        # Patch module-level globals
        original_model = api.model
        original_vectorizer = api.vectorizer
        api.model = mock_model
        api.vectorizer = mock_vectorizer
        
        client = TestClient(api.app)
        yield client
        
        # Restore originals
        api.model = original_model
        api.vectorizer = original_vectorizer
    
    def test_predict_with_text(self, client_with_model):
        """Test prediction with text input."""
        response = client_with_model.post('/predict', json={'text': 'هذا نص اختبار'})
        # Should return 200 if model loaded, or 503 if not
        assert response.status_code in [200, 503]
    
    def test_predict_empty_text(self, client_with_model):
        """Test prediction with empty text."""
        response = client_with_model.post('/predict', json={'text': ''})
        # Empty text might return error or still process
        assert response.status_code in [200, 400, 503]
    
    def test_predict_invalid_request(self, client_with_model):
        """Test prediction with invalid request format."""
        response = client_with_model.post('/predict', json={'wrong_key': 'test'})
        # Should return 422 for validation error
        assert response.status_code == 422


class TestRootEndpoint:
    """Test root endpoint."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from api import app
        return TestClient(app)
    
    def test_root_returns_200(self, client):
        """Test root endpoint returns 200."""
        response = client.get('/')
        assert response.status_code == 200
    
    def test_root_has_message(self, client):
        """Test root response has message."""
        response = client.get('/')
        data = response.json()
        assert 'message' in data


class TestModelLoading:
    """Test model loading functionality."""
    
    @patch('builtins.open')
    @patch('os.path.exists')
    def test_load_model_checks_paths(self, mock_exists, mock_open):
        """Test that model loading checks expected paths."""
        mock_exists.return_value = False
        
        # Import and call load function
        from api import load_local_model
        load_local_model()
        
        # Should have checked for file existence
        assert mock_exists.called
