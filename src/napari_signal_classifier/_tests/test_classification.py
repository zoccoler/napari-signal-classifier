import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path

from napari_signal_classifier._classification import (
    split_table_train_test, train_signal_classifier, predict_signal_labels,
    generate_sub_signals_table, train_sub_signal_classifier,
    generate_sub_signal_templates_from_annotations, predict_sub_signal_labels,
    _get_classifier_file_path
)


@pytest.fixture
def sample_signal_table():
    """Create a sample signal table for testing."""
    np.random.seed(42)
    n_labels = 10
    n_frames = 20
    
    data = []
    for label in range(n_labels):
        for frame in range(n_frames):
            # Create different patterns for different annotations
            if label < 5:  # Class 1
                intensity = 10 + 5 * np.sin(frame / 3) + np.random.normal(0, 0.5)
                annotation = 1
            else:  # Class 2
                intensity = 15 + 3 * np.cos(frame / 2) + np.random.normal(0, 0.5)
                annotation = 2
            
            data.append({
                'label': label,
                'frame': frame,
                'mean_intensity': intensity,
                'Annotations': annotation
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_sub_signal_table():
    """Create a sample table with sub-signal annotations."""
    np.random.seed(42)
    data = []
    
    for label in range(5):
        for frame in range(30):
            # Create sub-signals at different positions
            if 5 <= frame < 10:
                annotation = 1  # Sub-signal type 1
                intensity = 20 + 3 * np.sin(frame)
            elif 15 <= frame < 20:
                annotation = 2  # Sub-signal type 2
                intensity = 15 + 2 * np.cos(frame)
            else:
                annotation = 0  # Background
                intensity = 10 + np.random.normal(0, 0.5)
            
            data.append({
                'label': label,
                'frame': frame,
                'mean_intensity': intensity,
                'Annotations': annotation
            })
    
    return pd.DataFrame(data)


def test_get_classifier_file_path():
    """Test classifier file path generation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test with directory path
        path = _get_classifier_file_path(tmpdir)
        assert path.name == 'signal_classifier.pkl'
        assert path.parent == Path(tmpdir)
        
        # Test with None
        path = _get_classifier_file_path(None)
        assert path.name == 'signal_classifier.pkl'
        
        # Test with specific file path
        file_path = Path(tmpdir) / 'custom.pkl'
        path = _get_classifier_file_path(str(file_path))
        assert path == file_path


def test_split_table_train_test(sample_signal_table):
    """Test train/test split of signal table."""
    train, test = split_table_train_test(
        sample_signal_table, train_size=0.8, random_state=42
    )
    
    assert len(train) > 0
    assert len(test) > 0
    assert len(train) + len(test) == len(sample_signal_table)
    
    # Check that labels are properly split
    train_labels = train['label'].unique()
    test_labels = test['label'].unique()
    assert len(set(train_labels).intersection(set(test_labels))) == 0


def test_train_signal_classifier(sample_signal_table):
    """Test signal classifier training."""
    with tempfile.TemporaryDirectory() as tmpdir:
        classifier_path = train_signal_classifier(
            sample_signal_table,
            classifier_path=tmpdir,
            train_size=0.6,
            random_state=42,
            n_estimators=10
        )
        
        assert classifier_path is not None
        assert Path(classifier_path).exists()


def test_predict_signal_labels(sample_signal_table):
    """Test signal label prediction."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Train classifier
        classifier_path = train_signal_classifier(
            sample_signal_table,
            classifier_path=tmpdir,
            train_size=0.6,
            random_state=42,
            n_estimators=10
        )
        
        # Predict
        result_table = predict_signal_labels(
            sample_signal_table,
            classifier_path
        )
        
        assert 'Predictions' in result_table.columns
        assert result_table['Predictions'].dtype == int
        assert len(result_table) == len(sample_signal_table)


def test_train_sub_signal_classifier(sample_sub_signal_table):
    """Test sub-signal classifier training."""
    with tempfile.TemporaryDirectory() as tmpdir:
        classifier_path = train_sub_signal_classifier(
            sample_sub_signal_table,
            classifier_path=tmpdir,
            train_size=0.6,
            random_state=42,
            n_estimators=10
        )
        
        assert classifier_path is not None
        assert Path(classifier_path).exists()


def test_generate_sub_signal_templates_from_annotations(sample_sub_signal_table):
    """Test sub-signal template generation."""
    templates = generate_sub_signal_templates_from_annotations(
        sample_sub_signal_table
    )
    
    assert isinstance(templates, dict)
    assert len(templates) > 0
    assert all(isinstance(v, np.ndarray) for v in templates.values())
