from typing import TYPE_CHECKING

from qtpy import uic
from qtpy.QtWidgets import QWidget
from pathlib import Path
from cmap import Colormap

from napari_signal_selector.interactive import InteractiveFeaturesLineWidget
from nap_plot_tools.cmap import get_custom_cat10based_cmap_list
from skimage.util import map_array

from napari_signal_classifier._classification import train_signal_classifier, predict_signal_labels, train_sub_signal_classifier, predict_sub_signal_labels
from napari_signal_classifier._features import get_signal_features

from napari.utils import notifications
from napari.utils import DirectLabelColormap
import napari
import numpy as np

if TYPE_CHECKING:
    import napari


class Napari_Train_And_Predict_Signal_Classifier(QWidget):
    def __init__(self, napari_viewer, napari_plotter=None):
        super().__init__()
        self.viewer = napari_viewer
        self.plotter = napari_plotter
        if self.plotter is None:
            # Get plotter from napari viewer or add a new one if not present
            for name, dockwidget, in self.viewer.window._dock_widgets.items():
                if (name.startswith('Signal Selector') or name == 'InteractiveFeaturesLineWidget') and isinstance(
                        dockwidget.widget(), InteractiveFeaturesLineWidget):
                    self.plotter = dockwidget.widget()
                    break
            if self.plotter is None:
                print('Plotter not found! Openning Signal Selector widget...')
                notifications.show_warning(
                    'Plotter not found! Openning Signal Selector widget...')
                dock_widget, widget = self.viewer.window.add_plugin_dock_widget(
                    plugin_name='napari-signal-selector', widget_name='Signal Selector and Annotator', tabify=True)
                self.plotter = widget

        # load the .ui file from the same folder as this python file
        uic.loadUi(Path(__file__).parent /
                   "./_ui/napari_train_and_predict_signal_classfier.ui", self)

        self.viewer.layers.events.inserted.connect(
            self._reset_combobox_choices)
        self.viewer.layers.events.removed.connect(self._reset_combobox_choices)

        self._run_button.clicked.connect(self._run)
        # Populate combobox if there are already layers
        self._reset_combobox_choices()

        self.signal_features_in_metadata = True

    def _get_labels_layer_with_features(self):
        '''Get selected labels layer'''
        return [layer for layer in self.viewer.layers if isinstance(
            layer, napari.layers.Labels) and len(layer.features) > 0]

    def _get_layer_by_name(self, layer_name):
        '''Get layer by name'''
        return [layer for layer in self.viewer.layers if layer.name == layer_name][0]

    def _reset_combobox_choices(self):
        # clear pyqt combobox choices
        self._labels_layer_combobox.clear()
        labels_layers = self._get_labels_layer_with_features()
        # Set choices in qt combobox
        self._labels_layer_combobox.addItems(
            [layer.name for layer in labels_layers])
        # Link layer name change event to this method
        for layer in labels_layers:
            # Clear previous connections
            layer.events.name.disconnect()
            layer.events.name.connect(self._reset_combobox_choices)

    def _run(self):
        if self.plotter is None:
            print('Plotter not found')
            notifications.show_warning('Plotter not found')
            return

        classifier_path = self._classifier_path_line_edit.text()
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
        current_labels_layer = self._get_layer_by_name(
            self._labels_layer_combobox.currentText())
        table = current_labels_layer.features
        # Get signal features table
        signal_features_table = get_signal_features(
            table, column_id=object_id_column_name,
            column_sort=x_column_name,
            column_value=y_column_name)

        # Add signal features table as metadata
        if self.signal_features_in_metadata:
            current_labels_layer.metadata['signal_features'] = signal_features_table

        # Train signal classifier
        clssifier_path = train_signal_classifier(
            table,
            classifier_path,
            x_column_name=x_column_name,
            y_column_name=y_column_name,
            object_id_column_name=object_id_column_name,
            annotations_column_name=annotations_column_name
        )
        if clssifier_path is None:
            return

        # Get absolute path and set it to string
        clssifier_path = Path(clssifier_path).absolute().as_posix()
        self._classifier_path_line_edit.setText(clssifier_path)
        # Run predictions
        table_with_predictions = predict_signal_labels(
            table,
            classifier_file_path=clssifier_path,
            x_column_name=x_column_name,
            y_column_name=y_column_name,
            object_id_column_name=object_id_column_name,
            signal_features_table=signal_features_table
        )

        # Make new_labels image where each label is replaced by the prediction number
        label_list = table_with_predictions.groupby(
            object_id_column_name).first().reset_index()[object_id_column_name].values
        predictions_list = table_with_predictions.groupby(object_id_column_name).first().reset_index()[
            'Predictions'].values.astype('uint8')
        prediction_labels = map_array(np.asarray(current_labels_layer.data),
                                      np.asarray(label_list),
                                      np.asarray(predictions_list))

        # Update table with predictions
        current_labels_layer.features = table_with_predictions

        # Generate predicionts labels layer
        prediction_cmap = Colormap(
            get_custom_cat10based_cmap_list()).to_napari()
        predition_color_dict = {}
        predition_color_dict[None] = prediction_cmap.colors[0]
        for i in range(0, len(prediction_cmap.colors)):
            predition_color_dict[i] = prediction_cmap.colors[i]
        prediction_cmap_napari = DirectLabelColormap(
            color_dict=predition_color_dict)
        self.viewer.add_labels(
            prediction_labels, name='Signal Predictions', colormap=prediction_cmap_napari, opacity=0.5)
        # Select plotter back
        for name, dockwidget, in self.viewer.window._dock_widgets.items():
            if name == 'InteractiveFeaturesLineWidget':
                dockwidget.raise_()
                break
        # Select back the labels layer
        self.viewer.layers.selection.active = current_labels_layer
        # Re-plot with previous x_axis_key and y_axis_key
        self.plotter.y_axis_key = y_column_name
        self.plotter.x_axis_key = x_column_name
        self.plotter.object_id_axis_key = object_id_column_name

        # Update plot colors with predictions
        self.plotter.update_line_layout_from_column(column_name='Predictions')
        self.plotter.show_annotations_button.setChecked(False)
        self.plotter.show_predictions_button.setChecked(True)


class Napari_Train_And_Predict_Sub_Signal_Classifier(QWidget):
    def __init__(self, napari_viewer, napari_plotter=None):
        super().__init__()
        self.viewer = napari_viewer
        self.plotter = napari_plotter
        if self.plotter is None:
            # Get plotter from napari viewer or add a new one if not present
            for name, dockwidget, in self.viewer.window._dock_widgets.items():
                if (name.startswith('Signal Selector') or name == 'InteractiveFeaturesLineWidget') and isinstance(
                        dockwidget.widget(), InteractiveFeaturesLineWidget):
                    self.plotter = dockwidget.widget()
                    break
            if self.plotter is None:
                print('Plotter not found! Openning Signal Selector widget...')
                notifications.show_warning(
                    'Plotter not found! Openning Signal Selector widget...')
                dock_widget, widget = self.viewer.window.add_plugin_dock_widget(
                    plugin_name='napari-signal-selector', widget_name='Signal Selector and Annotator', tabify=True)
                self.plotter = widget
        # load the .ui file from the same folder as this python file
        uic.loadUi(Path(__file__).parent /
                   "./_ui/napari_train_and_predict_sub_signal_classfier.ui", self)

        self.viewer.layers.events.inserted.connect(
            self._reset_combobox_choices)
        self.viewer.layers.events.removed.connect(self._reset_combobox_choices)

        self._run_button.clicked.connect(self._run)
        # Populate combobox if there are already layers
        self._reset_combobox_choices()

        self.signal_features_in_metadata = True

    def _get_labels_layer_with_features(self):
        '''Get selected labels layer'''
        return [layer for layer in self.viewer.layers if isinstance(
            layer, napari.layers.Labels) and len(layer.features) > 0]

    def _get_layer_by_name(self, layer_name):
        '''Get layer by name'''
        return [layer for layer in self.viewer.layers if layer.name == layer_name][0]

    def _reset_combobox_choices(self):
        # clear pyqt combobox choices
        self._labels_layer_combobox.clear()
        labels_layers = self._get_labels_layer_with_features()
        # Set choices in qt combobox
        self._labels_layer_combobox.addItems(
            [layer.name for layer in labels_layers])
        # Link layer name change event to this method
        for layer in labels_layers:
            # Clear previous connections
            layer.events.name.disconnect()
            layer.events.name.connect(self._reset_combobox_choices)

    def _run(self):
        if self.plotter is None:
            print('Plotter not found')
            notifications.show_warning('Plotter not found')
            return

        classifier_path = self._classifier_path_line_edit.text()
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
        current_labels_layer = self._get_layer_by_name(
            self._labels_layer_combobox.currentText())
        table = current_labels_layer.features
        # # Get signal features table
        signal_features_table = get_signal_features(
            table, column_id=object_id_column_name,
            column_sort=x_column_name,
            column_value=y_column_name)
        # Add signal features table as metadata
        if self.signal_features_in_metadata:
            current_labels_layer.metadata['signal_features'] = signal_features_table
        # Train signal classifier
        clssifier_path = train_sub_signal_classifier(
            table,
            classifier_path,
            x_column_name=x_column_name,
            y_column_name=y_column_name,
            object_id_column_name=object_id_column_name,
            annotations_column_name=annotations_column_name
        )
        # Get absolute path and set it to string
        clssifier_path = Path(clssifier_path).absolute().as_posix()
        self._classifier_path_line_edit.setText(clssifier_path)
        # Run predictions
        print('Classifier path is:', clssifier_path)
        table_with_predictions = predict_sub_signal_labels(
            table,
            classifier_file_path=clssifier_path,
            x_column_name=x_column_name,
            y_column_name=y_column_name,
            object_id_column_name=object_id_column_name,
        )
        # Update table with predictions
        current_labels_layer.features = table_with_predictions

        # Generate predicionts labels layer
        label_list = table.groupby(self.plotter.object_id_axis_key).first().reset_index()[
            self.plotter.object_id_axis_key].values
        if len(current_labels_layer.data.shape) == 2:
            prediction_labels = np.stack(
                [current_labels_layer.data] * len(table[self.plotter.x_axis_key].unique()), axis=0)
        else:
            prediction_labels = np.copy(current_labels_layer.data)
        for i in range(prediction_labels.shape[0]):
            prediction_list = table[table[self.plotter.x_axis_key] == i].sort_values(
                by=self.plotter.object_id_axis_key)['Predictions'].values
            prediction_labels[i] = map_array(np.asarray(
                prediction_labels[i]), np.asarray(label_list), np.array(prediction_list))

        prediction_cmap = Colormap(
            get_custom_cat10based_cmap_list()).to_napari()
        predition_color_dict = {}
        predition_color_dict[None] = prediction_cmap.colors[0]
        for i in range(0, len(prediction_cmap.colors)):
            predition_color_dict[i] = prediction_cmap.colors[i]
        prediction_cmap_napari = DirectLabelColormap(
            color_dict=predition_color_dict)
        self.viewer.add_labels(
            prediction_labels, name='Sub-Signal Predictions', colormap=prediction_cmap_napari, opacity=0.5)

        # Select plotter back
        for name, dockwidget, in self.viewer.window._dock_widgets.items():
            if name == 'InteractiveFeaturesLineWidget':
                dockwidget.raise_()
                break
        # Select back the labels layer
        self.viewer.layers.selection.active = current_labels_layer
        # Re-plot with previous x_axis_key and y_axis_key
        self.plotter.y_axis_key = y_column_name
        self.plotter.x_axis_key = x_column_name
        self.plotter.object_id_axis_key = object_id_column_name

        # Update plot colors with predictions
        self.plotter.update_line_layout_from_column(column_name='Predictions')
        self.plotter.show_annotations_button.setChecked(False)
        self.plotter.show_predictions_button.setChecked(True)
