__version__ = "0.0.1"
from ._sample_data import make_sample_data
from ._widget import Napari_Train_And_Predict_Signal_Classifier
from ._features import calculate_statistics, calculate_crossings, calculate_entropy
from ._features import get_signal_features_table, get_wavelet_coefficients_features_table, get_signal_with_wavelets_features_table
from ._classification import train_and_predict_signal_classifier, train_signal_classifier, predict_signal_labels
from ._utilities import plot_wavelet_coefficient_decomposition_levels, get_frequency_bands


__all__ = (
    "make_sample_data",
    "Napari_Train_And_Predict_Signal_Classifier",
)
