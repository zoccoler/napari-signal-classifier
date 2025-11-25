from napari_signal_classifier._widget import Napari_Train_And_Predict_Signal_Classifier, Napari_Train_And_Predict_Sub_Signal_Classifier

def test_signal_classifier_widget(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = Napari_Train_And_Predict_Signal_Classifier(viewer)
    viewer.window.add_dock_widget(widget, area='right')
    dws = [key for key in viewer.window._dock_widgets.keys()]
    assert len(dws) == 2


def test_sub_signal_classifier_widget(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = Napari_Train_And_Predict_Sub_Signal_Classifier(viewer)
    viewer.window.add_dock_widget(widget, area='right')
    dws = [key for key in viewer.window._dock_widgets.keys()]
    assert len(dws) == 2
