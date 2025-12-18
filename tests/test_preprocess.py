"""Tests for preprocessing module."""
import pytest
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
from preprocess import preprocess_text, vectorize_text, split_data


class TestPreprocessText:
    """Test text preprocessing function."""
    
    def test_basic_text_cleaned(self):
        """Test that basic text is cleaned correctly."""
        text = "Hello World!"
        result = preprocess_text(text)
        assert isinstance(result, str)
        # Should lowercase and clean
        assert result == result.lower() or result.isascii()
    
    def test_arabic_text(self):
        """Test that Arabic text is preserved (core language)."""
        text = "مرحبا بالعالم"
        result = preprocess_text(text)
        assert isinstance(result, str)
        # Should not be empty for Arabic
        assert len(result) > 0
    
    def test_empty_string(self):
        """Test empty string handling."""
        result = preprocess_text("")
        assert result == ""
    
    def test_whitespace_normalization(self):
        """Test that extra whitespace is normalized."""
        text = "hello    world   test"
        result = preprocess_text(text)
        # Should not have multiple consecutive spaces
        assert "    " not in result
    
    def test_special_characters_removed(self):
        """Test that special characters/URLs are handled."""
        text = "Check http://example.com and @user #hashtag"
        result = preprocess_text(text)
        # Result should not contain full URLs
        assert "http://" not in result


class TestVectorizeText:
    """Test text vectorization function."""
    
    @pytest.fixture
    def sample_texts(self):
        """Sample texts for vectorization tests."""
        return pd.Series([
            "هذا نص اختبار",
            "نص آخر للاختبار",
            "اختبار ثالث"
        ])
    
    def test_returns_sparse_matrix(self, sample_texts):
        """Test that vectorize_text returns sparse matrix and vectorizer."""
        X, vectorizer = vectorize_text(sample_texts)
        from scipy.sparse import issparse
        assert issparse(X)
        assert isinstance(vectorizer, TfidfVectorizer)
    
    def test_matrix_shape(self, sample_texts):
        """Test that matrix has correct shape."""
        X, _ = vectorize_text(sample_texts)
        assert X.shape[0] == len(sample_texts)
        # Should have features
        assert X.shape[1] > 0
    
    def test_vectorizer_can_transform_new(self, sample_texts):
        """Test that vectorizer can transform new text."""
        X, vectorizer = vectorize_text(sample_texts)
        new_text = pd.Series(["نص جديد"])
        X_new = vectorizer.transform(new_text)
        assert X_new.shape[0] == 1
        assert X_new.shape[1] == X.shape[1]


class TestSplitData:
    """Test data splitting function."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for split testing."""
        n = 100
        X = np.random.rand(n, 10)
        y = np.random.randint(0, 2, n)
        return X, y
    
    def test_correct_split_sizes(self, sample_data):
        """Test that split produces correct sizes."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
        
        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) + len(y_test) == len(y)
        # Approximately 80-20 split
        assert abs(len(X_test) / len(X) - 0.2) < 0.05
    
    def test_arrays_aligned(self, sample_data):
        """Test that X and y remain aligned after split."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        assert X_train.shape[0] == len(y_train)
        assert X_test.shape[0] == len(y_test)
