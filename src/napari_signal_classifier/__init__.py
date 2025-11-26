try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"


from ._classification import (predict_signal_labels, predict_sub_signal_labels,
                              train_signal_classifier,
                              train_sub_signal_classifier)
from ._features import get_signal_features
from ._widget import (Napari_Train_And_Predict_Signal_Classifier,
                      Napari_Train_And_Predict_Sub_Signal_Classifier)

__all__ = (
    "Napari_Train_And_Predict_Signal_Classifier",
    "Napari_Train_And_Predict_Sub_Signal_Classifier",
)
