import numpy as np
import pandas as pd
import pytest

from napari_signal_classifier._detection import (
    align_signals, detect_sub_signal_by_template,
    extract_sub_signals_by_templates, generate_template_mean,
    generate_templates_by_category, normalize)
from napari_signal_classifier._sub_signals import SubSignalCollection


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

            data.append(
                {
                    "label": label,
                    "frame": frame,
                    "mean_intensity": intensity,
                    "Annotations": annotation,
                }
            )

    return pd.DataFrame(data)


def test_normalize():
    """Test signal normalization."""
    signal = np.array([1, 2, 3, 4, 5])

    # Test z-score normalization
    normalized_z = normalize(signal, method="zscores")
    assert np.isclose(np.mean(normalized_z), 0, atol=1e-10)
    assert np.isclose(np.std(normalized_z), 1, atol=1e-10)

    # Test min-max normalization
    normalized_mm = normalize(signal, method="minmax")
    assert np.isclose(np.min(normalized_mm), 0)
    assert np.isclose(np.max(normalized_mm), 1)

    # Test invalid method
    with pytest.raises(ValueError):
        normalize(signal, method="invalid")


def test_align_signals():
    """Test signal alignment using DTW."""
    reference = np.sin(np.linspace(0, 2 * np.pi, 50))
    signal = np.sin(np.linspace(0, 2 * np.pi, 50) + 0.1)  # Slightly shifted

    aligned = align_signals(reference, signal, detrend=False)

    assert len(aligned) == len(reference)
    assert isinstance(aligned, np.ndarray)


def test_generate_template_mean():
    """Test template generation from replicates."""
    # Create similar signals with slight variations
    replicates = [
        np.sin(np.linspace(0, 2 * np.pi, 50)) + np.random.normal(0, 0.1, 50)
        for _ in range(5)
    ]

    template = generate_template_mean(replicates, detrend=False)

    assert len(template) == len(replicates[0])
    assert isinstance(template, np.ndarray)


def test_detect_sub_signal_by_template():
    """Test sub-signal detection using template matching."""
    # Create a composite signal with a known pattern
    template = np.sin(np.linspace(0, 2 * np.pi, 20))
    composite = np.concatenate(
        [
            np.random.normal(0, 0.5, 30),
            template,
            np.random.normal(0, 0.5, 30),
            template,
            np.random.normal(0, 0.5, 30),
        ]
    )

    peaks = detect_sub_signal_by_template(composite, template, threshold=0.5)

    assert len(peaks) >= 2  # Should detect at least 2 peaks
    assert isinstance(peaks, np.ndarray)


def test_extract_sub_signals_by_templates(sample_sub_signal_table):
    """Test sub-signal extraction using templates."""
    from napari_signal_classifier._classification import \
        generate_sub_signal_templates_from_annotations

    # Generate templates first
    templates = generate_sub_signal_templates_from_annotations(
        sample_sub_signal_table
    )

    # Extract sub-signals
    collection = extract_sub_signals_by_templates(
        sample_sub_signal_table,
        "mean_intensity",
        "label",
        "frame",
        templates,
        threshold=0.5,
    )

    assert isinstance(collection, SubSignalCollection)
    assert len(collection.sub_signals) > 0
