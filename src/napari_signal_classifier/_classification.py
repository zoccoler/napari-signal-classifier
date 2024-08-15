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
                          object_id_column_name='label',
                          signal_features_table=None):
    # Get signal features table
    if signal_features_table is None:
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
