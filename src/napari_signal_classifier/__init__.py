try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"


from ._widget import Napari_Train_And_Predict_Signal_Classifier, Napari_Train_And_Predict_Sub_Signal_Classifier
from ._features import get_signal_features
from ._classification import train_signal_classifier, predict_signal_labels, train_sub_signal_classifier, predict_sub_signal_labels


__all__ = (
    "Napari_Train_And_Predict_Signal_Classifier",
    "Napari_Train_And_Predict_Sub_Signal_Classifier",
)
