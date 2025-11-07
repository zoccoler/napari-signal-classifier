

def get_signal_features(table, column_id='label', column_sort='frame', column_value='mean_intensity'):
    '''Extract time-series features from a table using tsfresh.

    Parameters
    ----------
    table : pd.DataFrame
        Input table containing time-series data in long format.
    column_id : str, optional
        Column name identifying different time-series (default is 'label').
    column_sort : str, optional
        Column name for sorting time points within each time-series (default is 'frame').
    column_value : str, optional
        Column name containing the time-series values (default is 'mean_intensity').
    Returns
    -------
    signal_features_table : pd.DataFrame
        DataFrame containing extracted features for each time-series.
    '''
    from tsfresh import extract_features
    from tsfresh.utilities.dataframe_functions import impute
    from tsfresh.feature_extraction import ComprehensiveFCParameters

    extraction_settings = ComprehensiveFCParameters()

    signal_features_table = extract_features(table,
                                             column_id=column_id,
                                             column_sort=column_sort,
                                             column_value=column_value,
                                             default_fc_parameters=extraction_settings,
                                             n_jobs=0,
                                             # we impute = remove all NaN features automatically
                                             impute_function=impute)
    return signal_features_table
