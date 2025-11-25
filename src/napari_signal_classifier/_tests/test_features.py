import numpy as np
import pandas as pd
import pytest

from napari_signal_classifier._features import get_signal_features


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


def test_get_signal_features(sample_signal_table):
    """Test signal feature extraction."""
    features = get_signal_features(
        sample_signal_table,
        column_id='label',
        column_sort='frame',
        column_value='mean_intensity'
    )
    
    assert isinstance(features, pd.DataFrame)
    assert len(features) == len(sample_signal_table['label'].unique())
    assert features.shape[1] > 0  # Should have multiple feature columns
