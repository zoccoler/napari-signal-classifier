from typing import TYPE_CHECKING

from qtpy import uic
from qtpy.QtWidgets import QWidget
from magicgui.widgets import ComboBox
from pathlib import Path
from cmap import Colormap

from napari_signal_selector.interactive import InteractiveFeaturesLineWidget
from nap_plot_tools.cmap import get_custom_cat10based_cmap_list
from napari_skimage_regionprops._parametric_images import relabel_with_map_array

from napari_signal_classifier._classification import train_signal_classifier, predict_signal_labels
from napari_signal_classifier._features import get_signal_features
from napari_signal_classifier.shared import RESULTS_TABLES

from napari.utils import notifications
from napari.qt.threading import thread_worker
import napari

if TYPE_CHECKING:
    import napari

# # Global variable to store multiple results
# SIGNAL_FEATURES_TABLE_LIST = []

@thread_worker
def run_in_thread(table, column_id, column_sort, column_value):
    # Call the function from the other file
    return get_signal_features(table, column_id, column_sort, column_value)

# Assign result to the shared global dictionary in the shared module
def assign_to_variable(result, key):
    global RESULTS_TABLES
    RESULTS_TABLES[key] = result
    print(f"Table with key '{key}' has been processed and stored.")
    print(result.shape)

# Start worker and assign result with a key
def start_worker_with_key(key, table, column_id='label', column_sort='frame', column_value='mean_intensity'):
    worker = run_in_thread(table, column_id, column_sort, column_value)  # create "worker" object
    worker.returned.connect(lambda result: assign_to_variable(result, key))
    worker.start()

class Napari_Train_And_Predict_Signal_Classifier(QWidget):
    def __init__(self, napari_viewer, napari_plotter=None):
        super().__init__()
        self.viewer = napari_viewer
        if napari_plotter is None:
            # Get plotter from napari viewer or add a new one if not present
            for name, dockwidget, in self.viewer.window._dock_widgets.items():
                if (name.startswith('Signal Selector') or name == 'InteractiveFeaturesLineWidget') and isinstance(
                        dockwidget.widget(), InteractiveFeaturesLineWidget):
                    self.plotter = dockwidget.widget()
                    break
        self.plotter = napari_plotter
            # if self.plotter is None:
            #     napari_plotter = InteractiveFeaturesLineWidget(self.viewer)
            #     self.viewer.window.add_dock_widget(napari_plotter, area='right', tabify=True)
            #     self.plotter = napari_plotter
        # load the .ui file from the same folder as this python file
        uic.loadUi(Path(__file__).parent / "./_ui/napari_train_and_predict_signal_classfier.ui", self)

        self.viewer.layers.events.inserted.connect(self._reset_combobox_choices)
        self.viewer.layers.events.removed.connect(self._reset_combobox_choices)

        self._run_button.clicked.connect(self._run)
        # Populate combobox if there are already layers
        self._reset_combobox_choices()

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
        self._labels_layer_combobox.addItems([layer.name for layer in labels_layers])
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
        table = self._get_layer_by_name(
            self._labels_layer_combobox.currentText()).features
        
        # TODO: Attempting mult-threading
        # worker = get_signal_features(
        #     table, column_id=object_id_column_name,
        #     column_sort=x_column_name,
        #     column_value=y_column_name)  # create "worker" object

        # worker.returned.connect(signal_features_table)  # connect callback functions
        # worker.start()  # start the thread!


        start_worker_with_key('table1', table, column_id=object_id_column_name,
            column_sort=x_column_name,
            column_value=y_column_name)

        # Get signal features table
        # signal_features_table = get_signal_features(
        #     table, column_id=object_id_column_name,
        #     column_sort=x_column_name,
        #     column_value=y_column_name)
        labels_data = self._get_layer_by_name(
            self._labels_layer_combobox.currentText()).data
        # Add signal features table as a new labels layer
        self.viewer.add_labels(labels_data, name='Labels Layer with Signal Features', features=RESULTS_TABLES['table1'], visible=False)

        # Train signal classifier
        clssifier_path = train_signal_classifier(
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
        table_with_predictions = predict_signal_labels(
            table,
            classifier_path=clssifier_path,
            x_column_name=x_column_name,
            y_column_name=y_column_name,
            object_id_column_name=object_id_column_name,
            signal_features_table=RESULTS_TABLES['table1']
        )

        # Make new_labels image where each label is replaced by the prediction number
        label_list = table_with_predictions.groupby(object_id_column_name).first().reset_index()[object_id_column_name].values
        predictions_list = table_with_predictions.groupby(object_id_column_name).first().reset_index()[
            'Predictions'].values.astype('uint8')
        prediction_labels = relabel_with_map_array(labels_data, label_list, predictions_list)
        # Update table with predictions
        self._get_layer_by_name(
            self._labels_layer_combobox.currentText()).features = table_with_predictions

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
        self.viewer.layers.selection.active = self._get_layer_by_name(
            self._labels_layer_combobox.currentText())

        # Re-plot with previous x_axis_key and y_axis_key
        self.plotter.y_axis_key = y_column_name
        self.plotter.x_axis_key = x_column_name
        self.plotter.object_id_axis_key = object_id_column_name

        # Update plot colors with predictions
        self.plotter.update_line_layout_from_column(column_name='Predictions')
