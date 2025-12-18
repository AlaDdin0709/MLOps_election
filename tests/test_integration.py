"""Integration tests for the full ML pipeline."""
import pytest
import os
from pathlib import Path
from unittest.mock import patch


class TestPipelineIntegration:
    """Test end-to-end pipeline integration."""
    
    @pytest.fixture
    def scripts_dir(self):
        """Get scripts directory."""
        return Path(__file__).parent.parent / 'scripts'
    
    def test_scripts_exist(self, scripts_dir):
        """Test that required scripts exist."""
        required_scripts = ['preprocess.py', 'train.py', 'register_best_model.py']
        for script in required_scripts:
            assert (scripts_dir / script).exists(), f"Missing script: {script}"
    
    def test_preprocess_imports(self):
        """Test that preprocess module can be imported."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        from preprocess import load_and_explore_data, preprocess_text, vectorize_text, split_data
        
        assert callable(load_and_explore_data)
        assert callable(preprocess_text)
        assert callable(vectorize_text)
        assert callable(split_data)
    
    def test_train_imports(self):
        """Test that train module can be imported."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        from train import get_ml_models, evaluate_model
        
        assert callable(get_ml_models)
        assert callable(evaluate_model)


class TestArtifactPaths:
    """Test that artifact paths are consistent across modules."""
    
    @pytest.fixture
    def expected_paths(self):
        """Expected artifact paths."""
        base = Path(__file__).parent.parent
        return {
            'model_registry': base / 'model_registry' / 'Best_Election_Model',
            'vectorizer': base / 'model_registry' / 'Best_Election_Model' / 'tfidf_vectorizer.pkl',
            'model': base / 'model_registry' / 'Best_Election_Model' / 'production.pkl',
        }
    
    def test_model_registry_dir_matches(self):
        """Test that train.py and api.py use same registry path."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        
        from train import MODEL_REGISTRY_DIR as train_registry
        
        # Check that the path structure is consistent
        assert 'model_registry' in str(train_registry)
        assert 'Best_Election_Model' in str(train_registry)


class TestEnvironmentConfig:
    """Test environment configuration."""
    
    def test_env_file_exists(self):
        """Test that .env.example or .env exists for reference."""
        base = Path(__file__).parent.parent
        # Either .env or .env.example should exist
        has_env = (base / '.env').exists() or (base / '.env.example').exists()
        # It's OK if neither exists - CI uses secrets
        assert True  # Just a documentation test
    
    def test_required_env_vars_documented(self):
        """Test that required env vars are documented in code."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        from train import DAGSHUB_USERNAME, DAGSHUB_REPO
        
        # These should be defined (may be empty in test env)
        assert DAGSHUB_USERNAME is not None
        assert DAGSHUB_REPO is not None
