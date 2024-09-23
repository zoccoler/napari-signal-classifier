

def get_signal_features(table, column_id='label', column_sort='frame', column_value='mean_intensity'):
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
