import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

from napari_signal_classifier._features import get_signal_features


def train_signal_classifier(table, classifier_path=None,
                            x_column_name='frame',
                            y_column_name='mean_intensity',
                            object_id_column_name='label',
                            annotations_column_name='Annotations',
                            random_state=None):
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
                          predictions_column_name='Predictions',
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
    table[predictions_column_name] = predictions.astype(int)

    return table

def generate_sub_signals_table(sub_signal_collection, resample=True):
    # Get max number of samples in sub-signals
    n_samples = max(sub_signal_collection.max_length_per_category.values())
    # Resample all sub-signals to the same number of samples
    table_sub_signals = pd.DataFrame([])
    for i, sub_signal in enumerate(sub_signal_collection.sub_signals):
        if resample:
            sub_table = pd.DataFrame(sub_signal.interpolate_samples(n_samples), columns=['mean_intensity_interpolated'])
            sub_table['frame_resampled'] = np.linspace(sub_signal.start_frame, sub_signal.end_frame, n_samples)
            sub_table['category'] = sub_signal.category
        else:
            sub_table = pd.DataFrame(sub_signal.data, columns=['mean_intensity'])
            sub_table['frame'] = np.arange(sub_signal.start_frame, sub_signal.end_frame + 1)
            sub_table['category'] = 0
            sub_table['template_category'] = sub_signal.category
        sub_table['sub_label'] = sub_signal.id
        sub_table['original_label'] = sub_signal.label
        sub_table['original_start_frame'] = sub_signal.start_frame
        sub_table['original_end_frame'] = sub_signal.end_frame
        
        table_sub_signals = pd.concat([table_sub_signals, sub_table], axis=0)
    # Re-order columns
    if resample:
        table_sub_signals = table_sub_signals[['sub_label', 'frame_resampled', 'mean_intensity_interpolated', 'category', 'original_label', 'original_start_frame', 'original_end_frame']]
    else:
        table_sub_signals = table_sub_signals[['sub_label', 'frame', 'mean_intensity', 'category', 'template_category', 'original_label', 'original_start_frame', 'original_end_frame']]
    return table_sub_signals

def train_sub_signal_classifier(table, classifier_path=None,
                            x_column_name='frame',
                            y_column_name='mean_intensity',
                            object_id_column_name='label',
                            annotations_column_name='Annotations', 
                            random_state=None):
    from napari_signal_classifier._sub_signals import extract_sub_signals_by_annotations
    # Get training data
    annotations_mask = table[annotations_column_name] != 0
    labels_with_annotations = np.unique(table[annotations_mask][object_id_column_name].values)
    table_training = table[table[object_id_column_name].isin(labels_with_annotations)].sort_values(by=[object_id_column_name, x_column_name]).reset_index(drop=True).iloc[:, 1:]

    # Extract sub-signals
    sub_signal_collection_train = extract_sub_signals_by_annotations(table_training, y_column_name, object_id_column_name, annotations_column_name, x_column_name)
    
    # Generate sub_signals table with resampling (all sub-signals to the same number of samples)
    sub_signals_table_training = generate_sub_signals_table(sub_signal_collection_train, resample=True)

    # Get sub_signal features table
    sub_signal_features_table_training = get_signal_features(
        sub_signals_table_training, column_id='sub_label',
         column_sort='frame_resampled',
         column_value='mean_intensity_interpolated')
    
    # Get annotations
    annotations = [sub_sig.category for sub_sig in sub_signal_collection_train.sub_signals]

    # Train classifier with training set
    if classifier_path is None or classifier_path == '':
        # TODO: remove fixed random state
        random_state = 42
        classifier = RandomForestClassifier(random_state=random_state)
        classifier_path = 'sub_signal_classifier.pkl'
    else:
        classifier = joblib.load(classifier_path)

    classifier.fit(sub_signal_features_table_training, annotations)
    train_score = classifier.score(sub_signal_features_table_training, annotations)
    joblib.dump(classifier, classifier_path)
    return classifier_path

def generate_sub_signal_templates_from_annotations(table, x_column_name='frame', y_column_name='mean_intensity', object_id_column_name='label', annotations_column_name='Annotations', detrend=False, smooth=0.1):
    from napari_signal_classifier._detection import generate_templates_by_category
    from napari_signal_classifier._sub_signals import extract_sub_signals_by_annotations
    annotations_mask = table[annotations_column_name] != 0
    labels_with_annotations = np.unique(table[annotations_mask][object_id_column_name].values)
    table_training = table[table[object_id_column_name].isin(labels_with_annotations)].sort_values(by=[object_id_column_name, x_column_name]).reset_index(drop=True).iloc[:, 1:]

    # Extract sub-signals by anntations
    sub_signal_collection_train = extract_sub_signals_by_annotations(table_training, y_column_name, object_id_column_name, annotations_column_name, x_column_name)
    # Generate sub-signal templates
    sub_signal_templates = generate_templates_by_category(sub_signal_collection_train, plot_results=False, detrend=detrend, smooth=smooth)
    return sub_signal_templates

def predict_sub_signal_labels(table, classifier_path, 
                          x_column_name='frame',
                          y_column_name='mean_intensity',
                          object_id_column_name='label',
                          annotations_column_name='Annotations', 
                          predictions_column_name='Predictions',
                          threshold=0.8,
                          sub_signal_templates=None,
                          sub_signal_features_table=None,
                          overlap=0.5,
                          detrend=False,
                          smooth=0.1):
    from napari_signal_classifier._detection import extract_sub_signals_by_templates
    # Generate sub-signal templates if not provided
    if sub_signal_templates is None:
        if annotations_column_name not in table.columns:
            raise ValueError(f'Annotations column {annotations_column_name} not found in table. Either provide sub_signal_templates directly or a valid annotations_column_name in the table to derive sub_signal_templates.')
        else:
            sub_signal_templates = generate_sub_signal_templates_from_annotations(table, x_column_name, y_column_name, object_id_column_name, annotations_column_name, detrend, smooth)

    # Extract sub-signals by templates
    sub_signal_collection = extract_sub_signals_by_templates(table, y_column_name, object_id_column_name, x_column_name, sub_signal_templates, threshold)

    # Merge sub-signals that overlap by more than overlap (50%by default) (they are likely the same sub_signal detected by different templates)
    sub_signal_collection.merge_subsignals(overlap_threshold=overlap)

    # Generate sub_signals table (no need to resample them since they were collected via template of fixed length)
    sub_signals_table = generate_sub_signals_table(sub_signal_collection, resample=False)

    # Calculate signal features table if not provided
    if sub_signal_features_table is None:
        sub_signal_features_table = get_signal_features(
            sub_signals_table, column_id='sub_label',
            column_sort='frame',
            column_value='mean_intensity')

    # Load classifier
    classifier = joblib.load(classifier_path)

    # Run predictions on all sub_signals and add them to the table
    predictions = classifier.predict(sub_signal_features_table)
    # Add predictions to sub_signals_table
    predictions_series = pd.Series(predictions, index=sub_signal_features_table.index)
    sub_signals_table['predicted_category'] = sub_signals_table['sub_label'].map(predictions_series)

    # Post-processing: removing duplicates
    # Ensure table_test_set is sorted by frame and original_label
    sub_signals_table = sub_signals_table.sort_values(by=['original_label', 'frame', 'original_start_frame'])

    # Step 1: Identify Duplicates (places where original_label and frame are the same, but different sub_labels were given)
    duplicates = sub_signals_table[sub_signals_table.duplicated(subset=['original_label', 'frame'], keep=False)]
    # Step 2: Reassign duplicate values
    duplicates_reassigned_series = duplicates.groupby(['original_label']).apply(reassign_duplicate_values)

    sub_signals_table = sub_signals_table.drop_duplicates(subset=['original_label', 'frame'])

    mapped_values = sub_signals_table[['original_label', 'frame']].apply(tuple, axis=1).map(duplicates_reassigned_series)
    sub_signals_table.loc[:, 'predicted_category'] = mapped_values.combine_first(sub_signals_table['predicted_category'])
    sub_signals_table['predicted_category'] = sub_signals_table['predicted_category'].astype(int)

    # Add predictions to table
    new_predictions_series = sub_signals_table.set_index(['original_label', 'frame'])['predicted_category']
    new_predictions_series.index.names = ['label', 'frame']
    mapped_values = table[['label', 'frame']].apply(tuple, axis=1).map(new_predictions_series)
    table['Predictions'] = mapped_values.fillna(0).astype(int)

    return table

def reassign_duplicate_values(table):
    categories = table['predicted_category'].values
    left_value = categories[0]
    right_value = categories[-1]
    # Create the new array
    new_categories = np.empty(len(categories) // 2, dtype=np.int64)

    # Fill the first half with the first value of the original array
    new_categories[:len(new_categories)//2] = left_value

    # Fill the second half with the second value of the original array
    new_categories[len(new_categories)//2:] = right_value

    table = table.set_index(['frame'])
    return pd.Series(new_categories, index=table.index[::2])