import numpy as np
from sklearn.ensemble import RandomForestClassifier
from napari_signal_classifier._features import get_signal_features
import joblib


def train_signal_classifier(table, classifier_path=None,
                            x_column_name='frame',
                            y_column_name='mean_intensity',
                            object_id_column_name='label',
                            annotations_column_name='Annotations'):
    # Get training data
    table_training = table[table[annotations_column_name] > 0]
    # Get signal features table
    signal_features_table_training = get_signal_features(
        table_training, column_id=object_id_column_name,
         column_sort=x_column_name,
         column_value=y_column_name)
     # Get annotations (one per signal in training set)
    annotations = table_training.groupby(object_id_column_name).first()['Annotations'].values
    # Train classifier with training set
    if classifier_path is None or classifier_path == '':
        # TODO: remove fixed random state
        random_state = 42
        classifier = RandomForestClassifier(random_state=random_state)
        classifier_path = 'signal_classifier.pkl'
    else:
        classifier = joblib.load(classifier_path)

    classifier.fit(signal_features_table_training, annotations)
    train_score = classifier.score(signal_features_table_training, annotations)
    joblib.dump(classifier, classifier_path)
    return classifier_path


def predict_signal_labels(table, classifier_path, 
                          x_column_name='frame',
                          y_column_name='mean_intensity',
                          object_id_column_name='label',):
    # Get signal features table
    signal_features_table = get_signal_features(
        table, column_id=object_id_column_name,
         column_sort=x_column_name,
         column_value=y_column_name)
    # Get classifier
    classifier = joblib.load(classifier_path)

    # Run predictions on all signals
    predictions = classifier.predict(signal_features_table)

    # Expand labels to match the number of frames
    number_of_frames = table[x_column_name].max() + 1
    predictions = np.repeat(predictions.tolist(), number_of_frames)

    # Add predictions to table
    table = table.sort_values([object_id_column_name, x_column_name])
    table['Predictions'] = predictions.astype(int)

    return table


def train_and_predict_signal_classifier(table, classifier_path=None, 
                                        x_column_name='frame',
                                        y_column_name='mean_intensity',
                                        object_id_column_name='label',
                                        annotations_column_name='Annotations'):
    classifier_path = train_signal_classifier(
        table,
        classifier_path,
        x_column_name,
        y_column_name,
        object_id_column_name,
        annotations_column_name)
    table = predict_signal_labels(
        table,
        classifier_path,
        x_column_name,
        y_column_name,
        object_id_column_name)
    return table, classifier_path


if __name__ == '__main__':
    # Load data
    import pandas as pd
    from pathlib import Path
    from skimage import io
    import napari
    from napari_signal_selector.interactive import InteractiveFeaturesLineWidget
    from napari_signal_classifier._widget import Napari_Train_And_Predict_Signal_Classifier

    parent_path = Path(__file__).parent.parent.parent
    print(parent_path)
    table = pd.read_csv(parent_path / 'notebooks/data/signals_annotated.csv')
    timelapse = io.imread(parent_path / 'notebooks/data/synthetic_image.tif')
    labels = io.imread(parent_path / 'notebooks/data/synthetic_labels.tif')

    # Create a viewer and add the image
    viewer = napari.Viewer()
    viewer.add_image(timelapse, name='timelapse')
    viewer.add_labels(labels, name='labels', features=table)

    # Create a plotter
    plotter = InteractiveFeaturesLineWidget(viewer)
    viewer.window.add_dock_widget(plotter, area='right')

    # Programatically select features and plot
    plotter.y_axis_key = 'mean_intensity'
    plotter.x_axis_key = 'frame'
    plotter.object_id_axis_key = 'label'

    widget = Napari_Train_And_Predict_Signal_Classifier(viewer, plotter)
    viewer.window.add_dock_widget(widget, area='right', tabify=True)

    napari.run()
