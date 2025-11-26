# Import InteractiveFeaturesLineWidget based on napari-signal-selector version
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING

from cmap import Colormap
from packaging.version import parse as parse_version
from qtpy import uic
from qtpy.QtWidgets import QWidget

nss_version = parse_version(version("napari-signal-selector"))
if nss_version >= parse_version("0.1.0"):
    from napari_signal_selector._interactive import \
        InteractiveFeaturesLineWidget
else:
    from napari_signal_selector.interactive import (
        InteractiveFeaturesLineWidget,
    )

import napari
import numpy as np
from nap_plot_tools.cmap import get_custom_cat10based_cmap_list
from napari.utils import DirectLabelColormap, notifications
from skimage.util import map_array

from napari_signal_classifier._classification import (
    predict_signal_labels, predict_sub_signal_labels, train_signal_classifier,
    train_sub_signal_classifier)
from napari_signal_classifier._features import get_signal_features

if TYPE_CHECKING:
    import napari


class Napari_Train_And_Predict_Signal_Classifier(QWidget):
    def __init__(self, napari_viewer, napari_plotter=None):
        super().__init__()
        self.viewer = napari_viewer
        self.plotter = napari_plotter
        if self.plotter is None:
            # Get plotter from napari viewer or add a new one if not present
            for (
                name,
                dockwidget,
            ) in self.viewer.window._dock_widgets.items():
                if (
                    name.startswith("Signal Selector")
                    or name == "InteractiveFeaturesLineWidget"
                ) and isinstance(
                    dockwidget.widget(), InteractiveFeaturesLineWidget
                ):
                    self.plotter = dockwidget.widget()
                    break
            if self.plotter is None:
                print("Plotter not found! Openning Signal Selector widget...")
                notifications.show_warning(
                    "Plotter not found! Openning Signal Selector widget..."
                )
                dock_widget, widget = (
                    self.viewer.window.add_plugin_dock_widget(
                        plugin_name="napari-signal-selector",
                        widget_name="Signal Selector and Annotator",
                        tabify=True,
                    )
                )
                self.plotter = widget

        # load the .ui file from the same folder as this python file
        uic.loadUi(
            Path(__file__).parent
            / "./_ui/napari_train_and_predict_signal_classfier.ui",
            self,
        )

        self.viewer.layers.events.inserted.connect(
            self._reset_combobox_choices
        )
        self.viewer.layers.events.removed.connect(self._reset_combobox_choices)

        self._run_qpushbutton.clicked.connect(self._run)

        # Connect classifier type combobox to update widget visibility
        self._classifier_type_qcombobox.currentTextChanged.connect(
            self._on_classifier_type_changed
        )

        # Populate combobox if there are already layers
        self._reset_combobox_choices()

        self.signal_features_in_metadata = True

        # Initialize widget visibility based on selected classifier
        self._on_classifier_type_changed(
            self._classifier_type_qcombobox.currentText()
        )

    def _get_labels_layer_with_features(self):
        """Get selected labels layer"""
        return [
            layer
            for layer in self.viewer.layers
            if isinstance(layer, napari.layers.Labels)
            and len(layer.features) > 0
        ]

    def _get_layer_by_name(self, layer_name):
        """Get layer by name"""
        return [
            layer for layer in self.viewer.layers if layer.name == layer_name
        ][0]

    def _on_classifier_type_changed(self, classifier_type):
        """Update widget visibility based on selected classifier type.

        Parameters
        ----------
        classifier_type : str
            The type of classifier selected (e.g., 'RandomForest').
        """
        # Define which widgets are visible for each classifier type
        classifier_widgets = {
            "RandomForest": [
                (
                    self._classifier_path_qlineedit,
                    self._classifier_path_qlabel,
                ),
                (self._n_trees_qspinbox, self._n_trees_qlabel),
                (self._random_state_qspinbox, self._random_state_qlabel),
            ],
            # Future classifier types will be added here
        }

        # Hide all classifier-specific widgets first
        all_widget_pairs = set()
        for widget_pairs in classifier_widgets.values():
            all_widget_pairs.update(widget_pairs)

        for widget, label in all_widget_pairs:
            widget.setVisible(False)
            label.setVisible(False)

        # Show widgets for the selected classifier
        if classifier_type in classifier_widgets:
            for widget, label in classifier_widgets[classifier_type]:
                widget.setVisible(True)
                label.setVisible(True)

    def _reset_combobox_choices(self):
        # clear pyqt combobox choices
        self._labels_layer_qcombobox.clear()
        labels_layers = self._get_labels_layer_with_features()
        # Set choices in qt combobox
        self._labels_layer_qcombobox.addItems(
            [layer.name for layer in labels_layers]
        )
        # Link layer name change event to this method
        for layer in labels_layers:
            # Clear previous connections
            layer.events.name.disconnect()
            layer.events.name.connect(self._reset_combobox_choices)

    def _run(self):
        if self.plotter is None:
            print("Plotter not found")
            notifications.show_warning("Plotter not found")
            return

        classifier_path = self._classifier_path_qlineedit.text()
        annotations_column_name = "Annotations"

        # Check if plotter has data
        if self.plotter.y_axis_key is None:
            print("Plot signals first")
            return
        else:
            y_column_name = self.plotter.y_axis_key
            x_column_name = self.plotter.x_axis_key
            object_id_column_name = self.plotter.object_id_axis_key
        # Get table from selected layer features
        current_labels_layer = self._get_layer_by_name(
            self._labels_layer_qcombobox.currentText()
        )
        table = current_labels_layer.features
        # Get signal features table
        signal_features_table = get_signal_features(
            table,
            column_id=object_id_column_name,
            column_sort=x_column_name,
            column_value=y_column_name,
        )

        # Add signal features table as metadata
        if self.signal_features_in_metadata:
            current_labels_layer.metadata["signal_features"] = (
                signal_features_table
            )

        # Get classifier parameters from widgets
        n_estimators = self._n_trees_qspinbox.value()
        random_state = self._random_state_qspinbox.value()
        if random_state < 0:
            random_state = None
        train_size = (
            self._training_size_qhorizontalslider.value() / 100.0
        )  # Convert percentage to fraction
        if train_size == 0:
            notifications.show_warning("Training size cannot be zero.")
            return
        stratify = self._stratify_qcheckbox.isChecked()

        # Validate sufficient annotated samples for train/test split
        annotated_mask = table[annotations_column_name] != 0
        n_annotated_labels = table[annotated_mask][
            object_id_column_name
        ].nunique()
        n_classes = table[annotated_mask][annotations_column_name].nunique()
        test_size = 1.0 - train_size
        n_train_samples = int(n_annotated_labels * train_size)
        n_test_samples = int(n_annotated_labels * test_size)

        if n_train_samples < n_classes:
            notifications.show_warning(
                f"Insufficient annotated samples for training set. "
                f"You have {n_annotated_labels} annotated labels and {n_classes} classes. "
                f"With {train_size*100:.0f}% training size, the training set would have {n_train_samples} samples, "
                f"but needs at least {n_classes} (one per class). "
                f"Please increase training size or annotate more samples."
            )
            return

        if n_test_samples < n_classes:
            notifications.show_warning(
                f"Insufficient annotated samples for test set. "
                f"You have {n_annotated_labels} annotated labels and {n_classes} classes. "
                f"With {train_size*100:.0f}% training size, the test set would have {n_test_samples} samples, "
                f"but needs at least {n_classes} (one per class). "
                f"Please decrease training size or annotate more samples."
            )
            return

        print(
            f"Training signal classifier with {n_annotated_labels} annotated signals and {n_classes} classes..."
        )
        print(
            f"Classifier parameters: "
            f"classifier={self._classifier_type_qcombobox.currentText()}, "
            f"n_estimators={n_estimators}, random_state={random_state}, "
            f"train_size={train_size}, stratify={stratify}"
        )

        # Train signal classifier
        classifier_file_path, train_score, test_score = (
            train_signal_classifier(
                table,
                classifier_path,
                x_column_name=x_column_name,
                y_column_name=y_column_name,
                object_id_column_name=object_id_column_name,
                annotations_column_name=annotations_column_name,
                n_estimators=n_estimators,
                random_state=random_state,
                train_size=train_size,
                stratify=stratify,
            )
        )
        if classifier_file_path is None:
            return

        # Get absolute path and set it to string
        classifier_file_path = Path(classifier_file_path).absolute().as_posix()
        self._classifier_path_qlineedit.setText(classifier_file_path)
        print("Classifier path is:", classifier_file_path)

        # Save classifier parameters to JSON file
        import json

        params_dict = {
            "classifier_type": self._classifier_type_qcombobox.currentText(),
            "n_estimators": n_estimators,
            "random_state": random_state,
            "train_size": train_size,
            "stratify": stratify,
            "n_annotated_labels": n_annotated_labels,
            "n_classes": n_classes,
            "n_train_samples": n_train_samples,
            "n_test_samples": n_test_samples,
            "train_score": train_score,
            "test_score": test_score,
        }
        params_path = Path(classifier_file_path).with_suffix(".json")
        with open(params_path, "w") as f:
            json.dump(params_dict, f, indent=2)
        print(f"Classifier parameters saved to: {params_path}")

        # Run predictions
        table_with_predictions = predict_signal_labels(
            table,
            classifier_file_path=classifier_file_path,
            x_column_name=x_column_name,
            y_column_name=y_column_name,
            object_id_column_name=object_id_column_name,
            signal_features_table=signal_features_table,
        )

        # Make new_labels image where each label is replaced by the prediction number
        label_list = (
            table_with_predictions.groupby(object_id_column_name)
            .first()
            .reset_index()[object_id_column_name]
            .values
        )
        predictions_list = (
            table_with_predictions.groupby(object_id_column_name)
            .first()
            .reset_index()["Predictions"]
            .values.astype("uint8")
        )
        prediction_labels = map_array(
            np.asarray(current_labels_layer.data),
            np.asarray(label_list),
            np.asarray(predictions_list),
        )

        # Update table with predictions
        current_labels_layer.features = table_with_predictions
        print("Adding predictions labels layer...")
        # Generate predicionts labels layer
        prediction_cmap = Colormap(
            get_custom_cat10based_cmap_list()
        ).to_napari()
        predition_color_dict = {}
        predition_color_dict[None] = prediction_cmap.colors[0]
        for i in range(0, len(prediction_cmap.colors)):
            predition_color_dict[i] = prediction_cmap.colors[i]
        prediction_cmap_napari = DirectLabelColormap(
            color_dict=predition_color_dict
        )
        self.viewer.add_labels(
            prediction_labels,
            name="Signal Predictions",
            colormap=prediction_cmap_napari,
            opacity=0.5,
        )
        # Select plotter back
        for (
            name,
            dockwidget,
        ) in self.viewer.window._dock_widgets.items():
            if name == "InteractiveFeaturesLineWidget":
                dockwidget.raise_()
                break
        # Select back the labels layer
        self.viewer.layers.selection.active = current_labels_layer
        # Re-plot with previous x_axis_key and y_axis_key
        self.plotter.y_axis_key = y_column_name
        self.plotter.x_axis_key = x_column_name
        self.plotter.object_id_axis_key = object_id_column_name

        # Update plot colors with predictions
        self.plotter.update_line_layout_from_column(column_name="Predictions")
        self.plotter.show_annotations_button.setChecked(False)
        self.plotter.show_predictions_button.setChecked(True)


class Napari_Train_And_Predict_Sub_Signal_Classifier(QWidget):
    def __init__(self, napari_viewer, napari_plotter=None):
        super().__init__()
        self.viewer = napari_viewer
        self.plotter = napari_plotter
        if self.plotter is None:
            # Get plotter from napari viewer or add a new one if not present
            for (
                name,
                dockwidget,
            ) in self.viewer.window._dock_widgets.items():
                if (
                    name.startswith("Signal Selector")
                    or name == "InteractiveFeaturesLineWidget"
                ) and isinstance(
                    dockwidget.widget(), InteractiveFeaturesLineWidget
                ):
                    self.plotter = dockwidget.widget()
                    break
            if self.plotter is None:
                print("Plotter not found! Openning Signal Selector widget...")
                notifications.show_warning(
                    "Plotter not found! Openning Signal Selector widget..."
                )
                dock_widget, widget = (
                    self.viewer.window.add_plugin_dock_widget(
                        plugin_name="napari-signal-selector",
                        widget_name="Signal Selector and Annotator",
                        tabify=True,
                    )
                )
                self.plotter = widget
        # load the .ui file from the same folder as this python file
        uic.loadUi(
            Path(__file__).parent
            / "./_ui/napari_train_and_predict_sub_signal_classfier.ui",
            self,
        )

        self.viewer.layers.events.inserted.connect(
            self._reset_combobox_choices
        )
        self.viewer.layers.events.removed.connect(self._reset_combobox_choices)

        self._run_qpushbutton.clicked.connect(self._run)

        # Connect classifier type combobox to update widget visibility
        self._classifier_type_qcombobox.currentTextChanged.connect(
            self._on_classifier_type_changed
        )

        # Connect detrend checkbox to update smooth widget visibility
        self._detrend_qcheckbox.stateChanged.connect(self._on_detrend_changed)

        # Populate combobox if there are already layers
        self._reset_combobox_choices()

        self.signal_features_in_metadata = True

        # Initialize widget visibility based on selected classifier
        self._on_classifier_type_changed(
            self._classifier_type_qcombobox.currentText()
        )

        # Initialize smooth widget visibility based on detrend state
        self._on_detrend_changed(self._detrend_qcheckbox.isChecked())

    def _get_labels_layer_with_features(self):
        """Get selected labels layer"""
        return [
            layer
            for layer in self.viewer.layers
            if isinstance(layer, napari.layers.Labels)
            and len(layer.features) > 0
        ]

    def _get_layer_by_name(self, layer_name):
        """Get layer by name"""
        return [
            layer for layer in self.viewer.layers if layer.name == layer_name
        ][0]

    def _on_classifier_type_changed(self, classifier_type):
        """Update widget visibility based on selected classifier type.

        Parameters
        ----------
        classifier_type : str
            The type of classifier selected (e.g., 'RandomForest').
        """
        # Define which widgets are visible for each classifier type
        classifier_widgets = {
            "RandomForest": [
                (
                    self._classifier_path_qlineedit,
                    self._classifier_path_qlabel,
                ),
                (self._n_trees_qspinbox, self._n_trees_qlabel),
                (self._random_state_qspinbox, self._random_state_qlabel),
            ],
            # Add more classifier types here in the future
        }

        # Hide all classifier-specific widgets first
        all_widget_pairs = set()
        for widget_pairs in classifier_widgets.values():
            all_widget_pairs.update(widget_pairs)

        for widget, label in all_widget_pairs:
            widget.setVisible(False)
            label.setVisible(False)

        # Show widgets for the selected classifier
        if classifier_type in classifier_widgets:
            for widget, label in classifier_widgets[classifier_type]:
                widget.setVisible(True)
                label.setVisible(True)

    def _on_detrend_changed(self, is_checked):
        """Update smooth widget visibility based on detrend checkbox state.

        Parameters
        ----------
        is_checked : bool
            Whether the detrend checkbox is checked.
        """
        self._smooth_factor_qslider.setVisible(is_checked)
        self._smooth_factor_qlabel.setVisible(is_checked)

    def _reset_combobox_choices(self):
        # clear pyqt combobox choices
        self._labels_layer_qcombobox.clear()
        labels_layers = self._get_labels_layer_with_features()
        # Set choices in qt combobox
        self._labels_layer_qcombobox.addItems(
            [layer.name for layer in labels_layers]
        )
        # Link layer name change event to this method
        for layer in labels_layers:
            # Clear previous connections
            layer.events.name.disconnect()
            layer.events.name.connect(self._reset_combobox_choices)

    def _run(self):
        if self.plotter is None:
            print("Plotter not found")
            notifications.show_warning("Plotter not found")
            return

        classifier_path = self._classifier_path_qlineedit.text()
        annotations_column_name = "Annotations"

        # Check if plotter has data
        if self.plotter.y_axis_key is None:
            print("Plot signals first")
            return
        else:
            y_column_name = self.plotter.y_axis_key
            x_column_name = self.plotter.x_axis_key
            object_id_column_name = self.plotter.object_id_axis_key
        # Get table from selected layer features
        current_labels_layer = self._get_layer_by_name(
            self._labels_layer_qcombobox.currentText()
        )
        table = current_labels_layer.features
        # # Get signal features table
        signal_features_table = get_signal_features(
            table,
            column_id=object_id_column_name,
            column_sort=x_column_name,
            column_value=y_column_name,
        )
        # Add signal features table as metadata
        if self.signal_features_in_metadata:
            current_labels_layer.metadata["signal_features"] = (
                signal_features_table
            )

        # Get classifier parameters from widgets
        n_estimators = self._n_trees_qspinbox.value()
        random_state = self._random_state_qspinbox.value()
        if random_state < 0:
            random_state = None
        train_size = (
            self._training_size_qhorizontalslider.value() / 100.0
        )  # Convert percentage to fraction
        if train_size == 0:
            notifications.show_warning("Training size cannot be zero.")
            return
        detection_threshold = (
            self._detection_threshold_qslider.value() / 100.0
        )  # Convert to 0-1 range
        detrend = self._detrend_qcheckbox.isChecked()
        smooth = (
            self._smooth_factor_qslider.value() / 100.0
        )  # Convert to 0-1 range
        merging_overlap_threshold = (
            self._merging_overlap_threshold_qslider.value() / 100.0
        )  # Convert to 0-1 range
        stratify = self._stratify_qcheckbox.isChecked()

        # Validate sufficient annotated samples for train/test split
        annotated_mask = table[annotations_column_name] != 0
        labels_with_annotations = np.unique(
            table[annotated_mask][object_id_column_name].values
        )
        table_annotated = (
            table[table[object_id_column_name].isin(labels_with_annotations)]
            .sort_values(by=[object_id_column_name, x_column_name])
            .reset_index(drop=True)
        )

        # Extract sub-signals
        from napari_signal_classifier._sub_signals import \
            extract_sub_signals_by_annotations

        sub_signal_collection = extract_sub_signals_by_annotations(
            table_annotated,
            y_column_name,
            object_id_column_name,
            annotations_column_name,
            x_column_name,
        )
        n_classes = len(sub_signal_collection.categories)
        n_annotated_sub_signals = len(sub_signal_collection.sub_signals)
        test_size = 1.0 - train_size
        n_train_samples = int(n_annotated_sub_signals * train_size)
        n_test_samples = int(n_annotated_sub_signals * test_size)

        if n_train_samples < n_classes:
            notifications.show_warning(
                f"Insufficient annotated samples for training set. "
                f"You have {n_annotated_sub_signals} annotated sub-signals and {n_classes} classes. "
                f"With {train_size*100:.0f}% training size, the training set would have {n_train_samples} samples, "
                f"but needs at least {n_classes} (one per class). "
                f"Please increase training size or annotate more samples."
            )
            return

        if n_test_samples < n_classes:
            notifications.show_warning(
                f"Insufficient annotated samples for test set. "
                f"You have {n_annotated_sub_signals} annotated sub-signals and {n_classes} classes. "
                f"With {train_size*100:.0f}% training size, the test set would have {n_test_samples} samples, "
                f"but needs at least {n_classes} (one per class). "
                f"Please decrease training size or annotate more samples."
            )
            return

        print(
            f"Training sub-signal classifier with {n_annotated_sub_signals} annotated sub-signals "
            f"and {n_classes} classes..."
        )
        print(
            f"Classifier parameters: "
            f"classifier={self._classifier_type_qcombobox.currentText()}, "
            f"n_estimators={n_estimators}, random_state={random_state}, "
            f"Training Percentage: {train_size*100:.0f}%, "
            f"Training size: {n_train_samples}, "
            f"Test size: {n_test_samples}, "
            f"Stratify: {stratify}, "
            f"Detection Threshold: {detection_threshold}, "
            f"Detrend: {detrend}, Smooth: {smooth}, "
            f"Merging Overlap Threshold: {merging_overlap_threshold}."
        )

        # Train signal classifier
        classifier_file_path, train_score, test_score = (
            train_sub_signal_classifier(
                table,
                classifier_path,
                x_column_name=x_column_name,
                y_column_name=y_column_name,
                object_id_column_name=object_id_column_name,
                annotations_column_name=annotations_column_name,
                n_estimators=n_estimators,
                random_state=random_state,
                stratify=stratify,
                train_size=train_size,
                detrend=detrend,
                smooth=smooth,
            )
        )
        if classifier_file_path is None:
            return

        # Get absolute path and set it to string
        classifier_file_path = Path(classifier_file_path).absolute().as_posix()
        self._classifier_path_qlineedit.setText(classifier_file_path)
        print("Classifier path is:", classifier_file_path)

        # Save classifier parameters to JSON file
        import json

        params_dict = {
            "classifier_type": self._classifier_type_qcombobox.currentText(),
            "n_estimators": n_estimators,
            "random_state": random_state,
            "train_size": train_size,
            "stratify": stratify,
            "detection_threshold": detection_threshold,
            "detrend": detrend,
            "smooth": smooth,
            "merging_overlap_threshold": merging_overlap_threshold,
            "n_annotated_sub_signals": n_annotated_sub_signals,
            "n_classes": n_classes,
            "n_train_samples": n_train_samples,
            "n_test_samples": n_test_samples,
            "train_score": train_score,
            "test_score": test_score,
        }
        params_path = Path(classifier_file_path).with_suffix(".json")
        with open(params_path, "w") as f:
            json.dump(params_dict, f, indent=2)
        print(f"Classifier parameters saved to: {params_path}")

        # Run predictions
        table_with_predictions = predict_sub_signal_labels(
            table,
            classifier_file_path=classifier_file_path,
            x_column_name=x_column_name,
            y_column_name=y_column_name,
            object_id_column_name=object_id_column_name,
            detection_threshold=detection_threshold,
            merging_overlap_threshold=merging_overlap_threshold,
            detrend=detrend,
            smooth=smooth,
        )
        # Update table with predictions
        current_labels_layer.features = table_with_predictions
        print("Adding prediction Labels layer...")
        # Generate predicionts labels layer
        label_list = (
            table.groupby(self.plotter.object_id_axis_key)
            .first()
            .reset_index()[self.plotter.object_id_axis_key]
            .values
        )
        if len(current_labels_layer.data.shape) == 2:
            prediction_labels = np.stack(
                [current_labels_layer.data]
                * len(table[self.plotter.x_axis_key].unique()),
                axis=0,
            )
        else:
            prediction_labels = np.copy(current_labels_layer.data)
        for i in range(prediction_labels.shape[0]):
            prediction_list = (
                table[table[self.plotter.x_axis_key] == i]
                .sort_values(by=self.plotter.object_id_axis_key)["Predictions"]
                .values
            )
            prediction_labels[i] = map_array(
                np.asarray(prediction_labels[i]),
                np.asarray(label_list),
                np.array(prediction_list),
            )

        prediction_cmap = Colormap(
            get_custom_cat10based_cmap_list()
        ).to_napari()
        predition_color_dict = {}
        predition_color_dict[None] = prediction_cmap.colors[0]
        for i in range(0, len(prediction_cmap.colors)):
            predition_color_dict[i] = prediction_cmap.colors[i]
        prediction_cmap_napari = DirectLabelColormap(
            color_dict=predition_color_dict
        )
        self.viewer.add_labels(
            prediction_labels,
            name="Sub-Signal Predictions",
            colormap=prediction_cmap_napari,
            opacity=0.5,
        )

        # Select plotter back
        for (
            name,
            dockwidget,
        ) in self.viewer.window._dock_widgets.items():
            if name == "InteractiveFeaturesLineWidget":
                dockwidget.raise_()
                break
        # Select back the labels layer
        self.viewer.layers.selection.active = current_labels_layer
        # Re-plot with previous x_axis_key and y_axis_key
        self.plotter.y_axis_key = y_column_name
        self.plotter.x_axis_key = x_column_name
        self.plotter.object_id_axis_key = object_id_column_name

        # Update plot colors with predictions
        self.plotter.update_line_layout_from_column(column_name="Predictions")
        self.plotter.show_annotations_button.setChecked(False)
        self.plotter.show_predictions_button.setChecked(True)
