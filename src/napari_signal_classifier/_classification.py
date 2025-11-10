from pathlib import Path
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
    '''Train a signal classifier using annotated signals in the table.

    Parameters
    ----------
    table : pd.DataFrame
        Input table containing time-series data in long format.
    classifier_path : str, optional
        Path to save the trained classifier (default is None).
    x_column_name : str, optional
        Column name for sorting time points within each time-series (default is 'frame').
    y_column_name : str, optional
        Column name containing the time-series values (default is 'mean_intensity').
    object_id_column_name : str, optional
        Column name identifying different time-series (default is 'label').
    annotations_column_name : str, optional
        Column name containing the annotations for training (default is 'Annotations').
    random_state : int, optional
        Random state for reproducibility (default is None).
    
    Returns
    -------
    classifier_file_path : str
        Path where the trained classifier is saved.
    '''
    # Get training data
    table_training = table[table[annotations_column_name] > 0]
    # Get signal features table
    signal_features_table_training = get_signal_features(
        table_training, column_id=object_id_column_name,
        column_sort=x_column_name,
        column_value=y_column_name)
    # Get annotations (one per signal in training set)
    annotations = table_training.groupby(object_id_column_name).first()[
        'Annotations'].values
    # Train classifier with training set
    # TODO: remove fixed random state
    if random_state is None:
        random_state = 42
    classifier = RandomForestClassifier(random_state=random_state)
    
    if classifier_path is None or classifier_path == '':
        # Create a classifier file path with a unique name
        classifier_folder_path = Path.cwd()
        base_name = 'signal_classifier'
        classifier_file_path = classifier_folder_path / f'{base_name}.pkl'
        counter = 1
        while classifier_file_path.exists():
            classifier_file_path = classifier_folder_path / f'{base_name}_{counter}.pkl'
            counter += 1

    classifier.fit(signal_features_table_training, annotations)
    train_score = classifier.score(signal_features_table_training, annotations)
    # TODO: Report train_score somewhere
    joblib.dump(classifier, classifier_file_path)
    return classifier_file_path


def predict_signal_labels(table, classifier_file_path,
                          x_column_name='frame',
                          y_column_name='mean_intensity',
                          object_id_column_name='label',
                          predictions_column_name='Predictions',
                          signal_features_table=None):
    '''Predict signal labels using a trained classifier.

    Parameters
    ----------
    table : pd.DataFrame
        Input table containing time-series data in long format.
    classifier_file_path : str
        Path to the trained classifier.
    x_column_name : str, optional
        Column name for sorting time points within each time-series (default is 'frame').
    y_column_name : str, optional
        Column name containing the time-series values (default is 'mean_intensity').
    object_id_column_name : str, optional
        Column name identifying different time-series (default is 'label').
    predictions_column_name : str, optional
        Column name to store the predictions (default is 'Predictions').
    signal_features_table : pd.DataFrame, optional
        Pre-computed signal features table (default is None).

    Returns
    -------
    table : pd.DataFrame
        Input table with an additional column for predictions.
    '''
    # Get signal features table
    if signal_features_table is None:
        signal_features_table = get_signal_features(
            table, column_id=object_id_column_name,
            column_sort=x_column_name,
            column_value=y_column_name)
    # Get classifier
    classifier = joblib.load(classifier_file_path)

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
    '''Generate a table from a collection of sub-signals.

    Parameters
    ----------
    sub_signal_collection : SubSignalCollection
        Collection of sub-signals.
    resample : bool, optional
        Whether to resample all sub-signals to the same number of samples (default is True).

    Returns
    -------
    table_sub_signals : pd.DataFrame
        Table containing the sub-signals data.
    '''
    # Get max number of samples in sub-signals
    n_samples = max(sub_signal_collection.max_length_per_category.values())
    # Resample all sub-signals to the same number of samples
    table_sub_signals = pd.DataFrame([])
    for i, sub_signal in enumerate(sub_signal_collection.sub_signals):
        if resample:
            sub_table = pd.DataFrame(sub_signal.interpolate_samples(
                n_samples), columns=['mean_intensity'])
            sub_table['frame_resampled'] = np.linspace(
                sub_signal.start_frame, sub_signal.end_frame, n_samples)
            sub_table['category'] = sub_signal.category
        else:
            sub_table = pd.DataFrame(
                sub_signal.data, columns=['mean_intensity'])
            sub_table['frame'] = np.arange(
                sub_signal.start_frame, sub_signal.end_frame + 1)
            sub_table['category'] = 0
            sub_table['template_category'] = sub_signal.category
        sub_table['sub_label'] = sub_signal.id
        sub_table['original_label'] = sub_signal.label
        sub_table['original_start_frame'] = sub_signal.start_frame
        sub_table['original_end_frame'] = sub_signal.end_frame

        table_sub_signals = pd.concat([table_sub_signals, sub_table], axis=0)
    # Re-order columns
    if resample:
        table_sub_signals = table_sub_signals[['sub_label', 'frame_resampled', 'mean_intensity',
                                               'category', 'original_label', 'original_start_frame', 'original_end_frame']]
    else:
        table_sub_signals = table_sub_signals[['sub_label', 'frame', 'mean_intensity', 'category',
                                               'template_category', 'original_label', 'original_start_frame', 'original_end_frame']]
    return table_sub_signals


def train_sub_signal_classifier(table, classifier_path=None,
                                x_column_name='frame',
                                y_column_name='mean_intensity',
                                object_id_column_name='label',
                                annotations_column_name='Annotations',
                                random_state=None):
    '''Train a sub-signal classifier using annotated sub-signals in the table.

    Parameters
    ----------
    table : pd.DataFrame
        Input table containing time-series data in long format.
    classifier_path : str, optional
        Path to save the trained classifier (default is None).
    x_column_name : str, optional
        Column name for sorting time points within each time-series (default is 'frame').
    y_column_name : str, optional
        Column name containing the time-series values (default is 'mean_intensity').
    object_id_column_name : str, optional
        Column name identifying different time-series (default is 'label').
    annotations_column_name : str, optional
        Column name containing the annotations for training (default is 'Annotations').
    random_state : int, optional
        Random state for reproducibility (default is None).
    
    Returns
    -------
    classifier_file_path : str
        Path where the trained classifier is saved.
    '''
    from napari_signal_classifier._sub_signals import extract_sub_signals_by_annotations
    # Get training data
    annotations_mask = table[annotations_column_name] != 0
    labels_with_annotations = np.unique(
        table[annotations_mask][object_id_column_name].values)
    table_training = table[table[object_id_column_name].isin(labels_with_annotations)].sort_values(
        by=[object_id_column_name, x_column_name]).reset_index(drop=True)

    # Extract sub-signals
    sub_signal_collection_train = extract_sub_signals_by_annotations(
        table_training, y_column_name, object_id_column_name, annotations_column_name, x_column_name)

    # Generate sub_signals table with resampling (all sub-signals to the same number of samples)
    sub_signals_table_training = generate_sub_signals_table(
        sub_signal_collection_train, resample=True)

    # Get sub_signal features table
    sub_signal_features_table_training = get_signal_features(
        sub_signals_table_training, column_id='sub_label',
        column_sort='frame_resampled',
        column_value='mean_intensity')

    # Get annotations
    annotations = [
        sub_sig.category for sub_sig in sub_signal_collection_train.sub_signals]

    if random_state is None:
        random_state = 42
    classifier = RandomForestClassifier(random_state=random_state)
    
    if classifier_path is None or classifier_path == '':
        # Create a classifier file path with a unique name
        classifier_folder_path = Path.cwd()
        base_name = 'sub_signal_classifier'
        classifier_file_path = classifier_folder_path / f'{base_name}.pkl'
        counter = 1
        while classifier_file_path.exists():
            classifier_file_path = classifier_folder_path / f'{base_name}_{counter}.pkl'
            counter += 1

    classifier.fit(sub_signal_features_table_training, annotations)
    train_score = classifier.score(
        sub_signal_features_table_training, annotations)
    print(f"Training score: {train_score:.4f}")
    joblib.dump(classifier, classifier_file_path)
    return classifier_file_path


def generate_sub_signal_templates_from_annotations(table, x_column_name='frame', y_column_name='mean_intensity', object_id_column_name='label', annotations_column_name='Annotations', detrend=False, smooth=0.1):
    '''Generate sub-signal templates from annotated sub-signals in the table.

    Parameters
    ----------
    table : pd.DataFrame
        Input table containing time-series data in long format.
    x_column_name : str, optional
        Column name for sorting time points within each time-series (default is 'frame').
    y_column_name : str, optional
        Column name containing the time-series values (default is 'mean_intensity').
    object_id_column_name : str, optional
        Column name identifying different time-series (default is 'label').
    annotations_column_name : str, optional
        Column name containing the annotations for training (default is 'Annotations').
    detrend : bool, optional
        Whether to detrend the sub-signals when generating templates (default is False).
    smooth : float, optional
        Smoothing factor to apply when generating templates (default is 0.1).
    
    Returns
    -------
    sub_signal_templates : dict
        Dictionary containing the generated sub-signal templates by category.
    '''
    from napari_signal_classifier._detection import generate_templates_by_category
    from napari_signal_classifier._sub_signals import extract_sub_signals_by_annotations
    annotations_mask = table[annotations_column_name] != 0
    labels_with_annotations = np.unique(
        table[annotations_mask][object_id_column_name].values)
    table_training = table[table[object_id_column_name].isin(labels_with_annotations)].sort_values(
        by=[object_id_column_name, x_column_name]).reset_index(drop=True)

    # Extract sub-signals by anntations
    sub_signal_collection_train = extract_sub_signals_by_annotations(
        table_training, y_column_name, object_id_column_name, annotations_column_name, x_column_name)
    # Generate sub-signal templates
    sub_signal_templates = generate_templates_by_category(
        sub_signal_collection_train, detrend=detrend, smooth=smooth)
    return sub_signal_templates


def predict_sub_signal_labels(table, classifier_file_path,
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
    '''Predict sub-signal labels using a trained sub-signal classifier.

    Parameters
    ----------
    table : pd.DataFrame
        Input table containing time-series data in long format.
    classifier_file_path : str
        Path to the trained sub-signal classifier.
    x_column_name : str, optional
        Column name for sorting time points within each time-series (default is 'frame').
    y_column_name : str, optional
        Column name containing the time-series values (default is 'mean_intensity').
    object_id_column_name : str, optional
        Column name identifying different time-series (default is 'label').
    annotations_column_name : str, optional
        Column name containing the annotations for training (default is 'Annotations').
    predictions_column_name : str, optional
        Column name to store the predictions (default is 'Predictions').
    threshold : float, optional
        Similarity threshold for sub-signal detection (default is 0.8).
    sub_signal_templates : dict, optional
        Pre-computed sub-signal templates (default is None).
    sub_signal_features_table : pd.DataFrame, optional
        Pre-computed sub-signal features table (default is None).
    overlap : float, optional
        Overlap threshold for merging sub-signals (default is 0.5).
    detrend : bool, optional
        Whether to detrend the sub-signals when generating templates (default is False).
    smooth : float, optional
        Smoothing factor to apply when generating templates (default is 0.1).
    
    Returns
    -------
    table : pd.DataFrame
        Input table with an additional column for predictions.
    '''
    from napari_signal_classifier._detection import extract_sub_signals_by_templates
    # Generate sub-signal templates if not provided
    if sub_signal_templates is None:
        if annotations_column_name not in table.columns:
            raise ValueError(
                f'Annotations column {annotations_column_name} not found in table. Either provide sub_signal_templates directly or a valid annotations_column_name in the table to derive sub_signal_templates.')
        else:
            sub_signal_templates = generate_sub_signal_templates_from_annotations(
                table, x_column_name, y_column_name, object_id_column_name, annotations_column_name, detrend, smooth)

    # Extract sub-signals by templates
    sub_signal_collection = extract_sub_signals_by_templates(
        table, y_column_name, object_id_column_name, x_column_name, sub_signal_templates, threshold)

    # Merge sub-signals that overlap by more than overlap (50%by default) (they are likely the same sub_signal detected by different templates)
    sub_signal_collection.merge_subsignals(overlap_threshold=overlap)

    # Generate sub_signals table (no need to resample them since they were collected via template of fixed length)
    sub_signals_table = generate_sub_signals_table(
        sub_signal_collection, resample=False)

    # Calculate signal features table if not provided
    if sub_signal_features_table is None:
        sub_signal_features_table = get_signal_features(
            sub_signals_table, column_id='sub_label',
            column_sort='frame',
            column_value='mean_intensity')

    # Load classifier
    classifier = joblib.load(classifier_file_path)

    # Run predictions on all sub_signals and add them to the table
    predictions = classifier.predict(sub_signal_features_table)
    # Add predictions to sub_signals_table
    predictions_series = pd.Series(
        predictions, index=sub_signal_features_table.index)
    sub_signals_table['predicted_category'] = sub_signals_table['sub_label'].map(
        predictions_series)

    # Post-processing: removing duplicates
    # Ensure table_test_set is sorted by frame and original_label
    sub_signals_table = sub_signals_table.sort_values(
        by=['original_label', 'frame', 'original_start_frame'])

    # Step 1: Identify Duplicates (places where original_label and frame are the same, but different sub_labels were given)
    duplicates = sub_signals_table[sub_signals_table.duplicated(
        subset=['original_label', 'frame'], keep=False)]
    # Step 2: Reassign duplicate values
    duplicates_reassigned_series = duplicates.groupby(['original_label']).apply(
        _reassign_duplicate_values, include_groups=False)

    sub_signals_table = sub_signals_table.drop_duplicates(
        subset=['original_label', 'frame'])

    original_frame_and_label_indices = sub_signals_table[[
        'original_label', 'frame']].apply(tuple, axis=1)
    # This mapping makes a series with the newly assigned categories for the duplicated values and leaves NaNs for the non-duplicated values
    prediction_series_duplicates_reassigned_only = original_frame_and_label_indices.map(
        duplicates_reassigned_series)
    # Sets the index to the original index of the table_test_set
    prediction_series_duplicates_reassigned_only.index = original_frame_and_label_indices
    # We get the prediction series, set the index to the original index of the table_test_set and combine it with the 'prediction_series_duplicates_reassigned' to replace the NaNs with the original predictions while keeping the re-assigned values for the duplicates
    predictions_series = sub_signals_table['predicted_category']
    predictions_series.index = original_frame_and_label_indices
    sub_signals_table['predicted_category'] = prediction_series_duplicates_reassigned_only.combine_first(
        predictions_series).values
    sub_signals_table['predicted_category'] = sub_signals_table['predicted_category'].astype(
        int)

    # Add predictions to table
    new_predictions_series = sub_signals_table.set_index(
        ['original_label', 'frame'])['predicted_category']
    new_predictions_series.index.names = ['label', 'frame']
    mapped_values = table[['label', 'frame']].apply(
        tuple, axis=1).map(new_predictions_series)
    table['Predictions'] = mapped_values.fillna(0).astype(int)

    return table


def _reassign_duplicate_values(table):
    '''Reassign duplicate values in the 'predicted_category' of a table, first half with first value, second half with last value.

    Parameters
    ----------
    table : pd.DataFrame
        Input table containing duplicate entries.
    
    Returns
    -------
    pd.Series
        Series with reassigned values.
    '''
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
