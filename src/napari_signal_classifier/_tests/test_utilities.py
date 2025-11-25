import numpy as np
import pandas as pd
import pytest

from napari_signal_classifier._utilities import (
    make_list_of_coefficients_names, extract_numbers_with_template,
    get_frequency_bands, plot_wavelet_coefficient_decomposition_levels
)


def test_make_list_of_coefficients_names():
    """Test wavelet coefficient names generation."""
    names = make_list_of_coefficients_names('db4', 3)
    
    assert len(names) == 4  # 1 approximation + 3 details
    assert names[0] == 'db4_cA_3'
    assert names[1] == 'db4_cD_3'
    assert names[-1] == 'db4_cD_1'


def test_extract_numbers_with_template():
    """Test number extraction from strings."""
    data_list = ['feature_1', 'feature_2.5', 'feature_3', 'other', 'feature_', 'feature_A']
    numbers = extract_numbers_with_template(data_list, 'feature_')
    
    assert len(numbers) == 5 # 'other' should be excluded
    assert '1' in numbers
    assert '2.5' in numbers


def test_get_frequency_bands():
    """Test frequency band calculation."""
    fcA, fcD = get_frequency_bands(2, 100)
    
    assert len(fcA) == 2
    assert len(fcD) == 2
    assert fcA[0] == 0
    assert fcD[1] > fcD[0]


def test_plot_wavelet_coefficient_decomposition_levels():
    """Test wavelet decomposition plotting."""
    # Create sample signals table
    signals = pd.DataFrame(
        np.random.randn(3, 100)  # 3 signals, 100 samples each
    )
    
    fig, ax = plot_wavelet_coefficient_decomposition_levels(
        signals, 'db4', 10, figsize=(10, 10)
    )
    
    assert fig is not None
    assert len(ax) > 0
