import numpy as np
import pandas as pd
import pytest

from napari_signal_classifier._sub_signals import (
    SubSignal, SubSignalCollection, extract_sub_signals_by_annotations_from_arrays,
    extract_sub_signals_by_annotations
)


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


def test_subsignal_creation():
    """Test SubSignal object creation."""
    data = np.array([1, 2, 3, 4, 5])
    subsignal = SubSignal(data, category=1, label=0, start_frame=10, end_frame=14)
    
    assert len(subsignal.data) == 5
    assert subsignal.category == 1
    assert subsignal.label == 0
    assert subsignal.start_frame == 10
    assert subsignal.end_frame == 14
    assert subsignal.id > 0


def test_subsignal_overlaps():
    """Test subsignal overlap detection."""
    ss1 = SubSignal(np.array([1, 2, 3]), 1, 0, 5, 7)
    ss2 = SubSignal(np.array([4, 5, 6]), 1, 0, 6, 8)
    ss3 = SubSignal(np.array([7, 8, 9]), 1, 0, 15, 17)
    
    assert ss1.overlaps(ss2, threshold=0.3)
    assert not ss1.overlaps(ss3, threshold=0.3)


def test_subsignal_merge():
    """Test subsignal merging."""
    ss1 = SubSignal(np.array([1, 2, 3]), 1, 0, 5, 7)
    ss2 = SubSignal(np.array([4, 5, 6]), 1, 0, 7, 9)
    
    original_id = ss1.id
    ss1.merge(ss2)
    
    assert ss1.start_frame == 5
    assert ss1.end_frame == 9
    assert ss1.id != original_id  # ID should change after merge


def test_subsignal_interpolate():
    """Test subsignal interpolation."""
    data = np.array([1, 2, 3, 4, 5])
    subsignal = SubSignal(data, 1, 0, 0, 4)
    
    interpolated = subsignal.interpolate_samples(10)
    
    assert len(interpolated) == 10
    assert isinstance(interpolated, np.ndarray)


def test_subsignal_collection():
    """Test SubSignalCollection functionality."""
    collection = SubSignalCollection()
    
    ss1 = SubSignal(np.array([1, 2, 3]), 1, 0, 0, 2)
    ss2 = SubSignal(np.array([4, 5, 6, 7]), 2, 0, 5, 8)
    
    collection.add_sub_signal(ss1)
    collection.add_sub_signal(ss2)
    
    assert len(collection.sub_signals) == 2
    assert 1 in collection.categories
    assert 2 in collection.categories
    assert collection.max_length_per_category[1] == 3
    assert collection.max_length_per_category[2] == 4


def test_subsignal_collection_sort():
    """Test subsignal collection sorting."""
    collection = SubSignalCollection()
    
    ss1 = SubSignal(np.array([1, 2, 3]), 2, 0, 0, 2)
    ss2 = SubSignal(np.array([4, 5, 6]), 1, 0, 5, 7)
    
    collection.add_sub_signal(ss1)
    collection.add_sub_signal(ss2)
    collection.sort_by_category()
    
    assert collection.sub_signals[0].category == 1
    assert collection.sub_signals[1].category == 2


def test_subsignal_collection_merge():
    """Test subsignal collection merging."""
    collection = SubSignalCollection()
    
    ss1 = SubSignal(np.array([1, 2, 3]), 1, 0, 5, 7)
    ss2 = SubSignal(np.array([4, 5, 6]), 2, 0, 6, 8)
    
    collection.add_sub_signal(ss1)
    collection.add_sub_signal(ss2)
    
    original_count = len(collection.sub_signals)
    collection.merge_subsignals(merging_overlap_threshold=0.3)
    
    # Should merge overlapping signals
    assert len(collection.sub_signals) <= original_count


def test_extract_sub_signals_by_annotations_from_arrays():
    """Test sub-signal extraction from arrays."""
    signal_data = np.array([1, 1, 2, 2, 2, 1, 1, 3, 3, 1, 1])
    annotations = np.array([0, 1, 1, 1, 0, 0, 2, 2, 0, 0, 0])
    labels = np.zeros(11, dtype=int)
    frames = np.arange(11)
    
    collection = extract_sub_signals_by_annotations_from_arrays(
        signal_data, annotations, labels, frames
    )
    
    assert isinstance(collection, SubSignalCollection)
    assert len(collection.sub_signals) == 2  # Two annotated regions


def test_extract_sub_signals_by_annotations(sample_sub_signal_table):
    """Test sub-signal extraction from DataFrame."""
    collection = extract_sub_signals_by_annotations(
        sample_sub_signal_table,
        'mean_intensity',
        'label',
        'Annotations',
        'frame'
    )
    
    assert isinstance(collection, SubSignalCollection)
    assert len(collection.sub_signals) > 0
