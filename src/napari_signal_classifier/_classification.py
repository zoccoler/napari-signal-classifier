import numpy as np
from sklearn.ensemble import RandomForestClassifier
from napari_signal_classifier._features import get_signal_with_wavelets_features_table
import joblib


def train_signal_classifier(table, classifier_path=None, features_names=[
                            'statistics', 'crossings', 'entropy'],
                            include_original_signal=True,
                            waveletname='db4',
                            x_column_name='frame',
                            y_column_name='mean_intensity',
                            object_id_column_name='label',
                            annotations_column_name='Annotations'):
    # Get training data
    table_training = table[table[annotations_column_name] > 0]
    # Reshape table, storing one signal per row
    signals_table_training = table_training.pivot(
        index=object_id_column_name,
        columns=x_column_name,
        values=y_column_name)
    # Get signal features table
    signal_features_table_training = get_signal_with_wavelets_features_table(
        signals_table_training, waveletname, features_names, include_original_signal)
    # Shape data for training
    X_train = signal_features_table_training.iloc[:, 1:]  # remove the first column (the object_id)
    # Get annotations (one per signal in training set)
    annotations = table_training.groupby(object_id_column_name).mean()['Annotations'].values
    # Train classifier with training set
    if classifier_path is None or classifier_path == '':
        # TODO: remove fixed random state
        random_state = 42
        classifier = RandomForestClassifier(random_state=random_state)
        classifier_path = 'signal_classifier.pkl'
    else:
        classifier = joblib.load(classifier_path)

    classifier.fit(X_train, annotations)
    train_score = classifier.score(X_train, annotations)
    joblib.dump(classifier, classifier_path)
    return classifier_path


def predict_signal_labels(table, classifier_path, features_names=[
                          'statistics', 'crossings', 'entropy'],
                          include_original_signal=True,
                          waveletname='db4',
                          x_column_name='frame',
                          y_column_name='mean_intensity',
                          object_id_column_name='label',):
    signals_table = table.pivot(
        index=object_id_column_name,
        columns=x_column_name,
        values=y_column_name)

    # Get signal features table
    signal_features_table = get_signal_with_wavelets_features_table(
        signals_table, waveletname, features_names, include_original_signal)
    # Shape data for predictions
    X_pred = signal_features_table.iloc[:, 1:]  # remove the first column (the object_id)

    # Get classifier
    classifier = joblib.load(classifier_path)

    # Run predictions on all signals
    Y_pred = classifier.predict(X_pred)

    # Expand labels to match the number of frames
    number_of_frames = table[x_column_name].max() + 1
    Y_pred = np.repeat(Y_pred.tolist(), number_of_frames)

    # Add predictions to table
    table = table.sort_values([object_id_column_name, x_column_name])
    table['Predictions'] = Y_pred.astype(int)

    return table


def train_and_predict_signal_classifier(table, classifier_path=None, features_names=[
                                        'statistics', 'crossings', 'entropy'],
                                        include_original_signal=True,
                                        waveletname='db4',
                                        x_column_name='frame',
                                        y_column_name='mean_intensity',
                                        object_id_column_name='label',
                                        annotations_column_name='Annotations'):
    classifier_path = train_signal_classifier(
        table,
        classifier_path,
        features_names,
        include_original_signal,
        waveletname,
        x_column_name,
        y_column_name,
        object_id_column_name,
        annotations_column_name)
    table = predict_signal_labels(
        table,
        classifier_path,
        features_names,
        include_original_signal,
        waveletname,
        x_column_name,
        y_column_name,
        object_id_column_name)
    return table, classifier_path


# if __name__ == '__main__':
#     # Load data

#     table = pd.read_csv('./data/signals_annotated.csv')
#     timelapse = io.imread('./data/synthetic_image.tif')
#     labels = io.imread('./data/synthetic_labels.tif')

#     # Create a viewer and add the image
#     viewer = napari.Viewer()
#     viewer.add_image(timelapse, name='timelapse')
#     viewer.add_labels(labels, name='labels', features=table)

#     # Create a plotter
#     plotter = InteractiveFeaturesLineWidget(viewer)
#     viewer.window.add_dock_widget(plotter, area='right')

#     # Programatically select features and plot
#     plotter.y_axis_key = 'mean_intensity'
#     plotter.x_axis_key = 'frame'
#     plotter.object_id_axis_key = 'label'

#     widget = Napari_Train_And_Predict_Signal_Classifier(viewer, plotter)
#     viewer.window.add_dock_widget(widget, area='right', tabify=True)

#     napari.run()
