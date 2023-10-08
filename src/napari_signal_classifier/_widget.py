from typing import TYPE_CHECKING

from qtpy import uic
from qtpy.QtCore import QEvent, QObject
from qtpy.QtWidgets import QWidget, QListWidgetItem
from magicgui.widgets import ComboBox
from pathlib import Path
from cmap import Colormap

from napari_signal_selector.interactive import InteractiveFeaturesLineWidget
from napari_signal_selector.utilities import get_custom_cat10based_cmap_list
from napari_skimage_regionprops._parametric_images import relabel_with_map_array

from napari_signal_classifier._classification import train_and_predict_signal_classifier
from napari_signal_classifier._features import get_signal_with_wavelets_features_table
from napari_signal_classifier._utilities import extract_numbers_with_template

from napari.utils import notifications
import napari

if TYPE_CHECKING:
    import napari


class Napari_Train_And_Predict_Signal_Classifier(QWidget):
    def __init__(self, napari_viewer, napari_plotter=None):
        super().__init__()
        self.viewer = napari_viewer
        if napari_plotter is None:
            # Get plotter from napari viewer
            for name, dockwidget, in self.viewer.window._dock_widgets.items():
                if (name.startswith('Signal Selector') or name == 'InteractiveFeaturesLineWidget') and isinstance(
                        dockwidget.widget(), InteractiveFeaturesLineWidget):
                    self.plotter = dockwidget.widget()
                    break
        self.plotter = napari_plotter
        # load the .ui file from the same folder as this python file
        uic.loadUi(Path(__file__).parent / "./_ui/napari_train_and_predict_signal_classfier.ui", self)
        # add magicgui widget to widget layout
        self._labels_combobox = ComboBox(
            choices=self._get_labels_layer_with_features,
            label='Labels layer:',
            tooltip='Select labels layer with features to train and predict',
        )
        self.viewer.layers.events.inserted.connect(self._labels_combobox.reset_choices)
        self.viewer.layers.events.removed.connect(self._labels_combobox.reset_choices)
        self.layout().insertWidget(0, self._labels_combobox.native)
        self.installEventFilter(self)

        # Set features options
        self._features_options = ['statistics', 'crossings', 'entropy']
        for choice in self._features_options:
            item = QListWidgetItem(choice)
            self._features_multi_select_widget.addItem(item)
            # Set the items as selected
            item.setSelected(True)

        # Set wavelet options
        self._wavelet_family_combobox.addItems(['db', 'sym', 'coif', 'bior', 'rbio', 'dmey', 'haar'])
        self._wavelet_family_combobox.currentIndexChanged.connect(self._on_wavelet_family_change)
        self._wavelet_family_combobox.setCurrentIndex(0)
        # Set wavelet initial order options from the first family
        self._on_wavelet_family_change(0)
        # Start with 'db4'
        self._wavelet_order_combobox.setCurrentIndex(3)

        self._run_button.clicked.connect(self._run)

    def eventFilter(self, obj: QObject, event: QEvent):
        if event.type() == QEvent.ParentChange:
            self._labels_combobox.parent_changed.emit(self.parent())

        return super().eventFilter(obj, event)

    def _get_labels_layer_with_features(self, combo_box):
        '''Get selected labels layer'''
        return [layer for layer in self.viewer.layers if isinstance(
            layer, napari.layers.Labels) and len(layer.features) > 0]

    def _on_wavelet_family_change(self, index):
        '''Update wavelet order options'''
        import pywt
        wavelet_family = self._wavelet_family_combobox.currentText()
        available_wavelet_orders = pywt.wavelist(wavelet_family)
        wavelet_order_list = extract_numbers_with_template(available_wavelet_orders, wavelet_family)
        self._wavelet_order_combobox.clear()
        self._wavelet_order_combobox.addItems([str(order) for order in wavelet_order_list])

    def _run(self):
        if self.plotter is None:
            print('Plotter not found')
            notifications.show_warning('Plotter not found')
            return
        selected_items = self._features_multi_select_widget.selectedItems()
        selected_features = []
        if selected_items:
            selected_features = [item.text() for item in selected_items]

        classifier_path = self._classifier_path_line_edit.text()
        waveletname = self._wavelet_family_combobox.currentText() + self._wavelet_order_combobox.currentText()
        include_orignal_signal = self._include_original_signal_checkbox.isChecked()
        annotations_column_name = 'Annotations'

        # Check if plotter has data
        if self.plotter.y_axis_key is None:
            print('Plot signals first')
            return
        else:
            y_column_name = self.plotter.y_axis_key
            x_column_name = self.plotter.x_axis_key
            object_id_column_name = self.plotter.object_id_axis_key
        # Get table from selected layer features
        table = self._labels_combobox.value.features

        table, classifier_path = train_and_predict_signal_classifier(
            table,
            classifier_path,
            features_names=selected_features,
            include_original_signal=include_orignal_signal,
            waveletname=waveletname,
            y_column_name=y_column_name,
            annotations_column_name=annotations_column_name)

        # Make new_labels image where each label is replaced by the prediction number
        labels_data = self._labels_combobox.value.data
        label_list = table.groupby(object_id_column_name).mean().reset_index()[object_id_column_name].values
        predictions_list = table.groupby(object_id_column_name).mean().reset_index()[
            'Predictions'].values.astype('uint8')
        prediction_labels = relabel_with_map_array(labels_data, label_list, predictions_list)
        # Update table with predictions
        self.viewer.layers.selection.active.features = table

        # Display layers in napari
        # Get signal features table
        signals_table = table.pivot(
            index=object_id_column_name,
            columns=x_column_name,
            values=y_column_name)
        signal_features_table = get_signal_with_wavelets_features_table(
            signals_table, waveletname, selected_features, include_orignal_signal)
        # Add signal features table as a new labels layer
        self.viewer.add_labels(labels_data, name='signal features', features=signal_features_table, visible=False)

        # Generate predicionts labels layer
        prediction_cmap = Colormap(get_custom_cat10based_cmap_list()).to_napari()
        predition_color_dict = {}
        for i in range(0, len(prediction_cmap.colors)):
            predition_color_dict[i] = prediction_cmap.colors[i]
        self.viewer.add_labels(prediction_labels, name='predictions', color=predition_color_dict)

        # Select plotter back
        for name, dockwidget, in self.viewer.window._dock_widgets.items():
            if name == 'InteractiveFeaturesLineWidget':
                dockwidget.raise_()
                break

        # Select back the labels layer
        # TODO: Identify labels layer in a different way
        self.viewer.layers.selection.active = self.viewer.layers['labels']

        # Re-plot with previous x_axis_key and y_axis_key
        self.plotter.y_axis_key = y_column_name
        self.plotter.x_axis_key = x_column_name
        self.plotter.object_id_axis_key = object_id_column_name

        # Update plot colors with predictions
        self.plotter.update_line_layout_from_column(column_name='Predictions')
