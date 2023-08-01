from ._utilities import make_list_of_coefficients_names


def calculate_entropy(list_values):
    from collections import Counter
    from scipy import stats
    counter_values = Counter(list_values).most_common()
    probabilities = [elem[1] / len(list_values) for elem in counter_values]
    entropy = stats.entropy(probabilities)
    return entropy


def calculate_statistics(list_values):
    import numpy as np
    n5 = np.nanpercentile(list_values, 5)
    n25 = np.nanpercentile(list_values, 25)
    n75 = np.nanpercentile(list_values, 75)
    n95 = np.nanpercentile(list_values, 95)
    median = np.nanpercentile(list_values, 50)
    mean = np.nanmean(list_values)
    std = np.nanstd(list_values)
    var = np.nanvar(list_values)
    rms = np.nanmean(np.sqrt(list_values**2))
    return [n5, n25, n75, n95, median, mean, std, var, rms]


def calculate_crossings(list_values):
    import numpy as np
    zero_crossing_indices = np.nonzero(np.diff(np.array(list_values) > 0))[0]
    no_zero_crossings = len(zero_crossing_indices)
    mean_crossing_indices = np.nonzero(np.diff(np.array(list_values) > np.nanmean(list_values)))[0]
    no_mean_crossings = len(mean_crossing_indices)
    return [no_zero_crossings, no_mean_crossings]


def get_signal_features_table(signals_table, features_names=[
                              'statistics', 'crossings', 'entropy'], extra_features=None):
    import pandas as pd
    list_of_features_tables = []

    table_statistics = pd.DataFrame(signals_table.T.apply(calculate_statistics, axis=0).T)
    table_statistics.columns = [
        'percentile_5',
        'percentile_25',
        'percentile_75',
        'percentile_95',
        'median',
        'mean',
        'standard_deviation',
        'variance',
        'root-mean-square']
    list_of_features_tables.append(table_statistics)

    table_crossings = pd.DataFrame(signals_table.T.apply(calculate_crossings, axis=0).T)
    table_crossings.columns = ['n_zero_crossings', 'n_mean_crossings']
    list_of_features_tables.append(table_crossings)

    table_entropy = pd.DataFrame(signals_table.T.apply(calculate_entropy, axis=0).T)
    table_entropy.columns = ['entropy']
    list_of_features_tables.append(table_entropy)

    if extra_features is not None:
        for func in extra_features:
            table_extra = pd.DataFrame(signals_table.T.apply(func, axis=0).T)
            table_extra.columns = [func.__name__]
            list_of_features_tables.append(table_extra)

    table_features = pd.concat(list_of_features_tables, axis=1)
    return table_features


def get_wavelet_coefficients_features_table(signals_table, waveletname, features_names=[
                                            'statistics', 'crossings', 'entropy']):
    import pywt
    import pandas as pd
    # Get max level of discrete wavelet
    max_level_of_decomposition = pywt.dwt_max_level(signals_table.shape[1], waveletname)
    # Decompose signals into wavelets coefficients (returns a list of arrays,
    # whose rows are labels and columns are the coefficients amplitudes)
    list_wavelet_coefficients = pywt.wavedec(signals_table.values, waveletname, level=max_level_of_decomposition)
    list_of_wavelet_decomposition_level_names = make_list_of_coefficients_names(waveletname, max_level_of_decomposition)

    list_of_coeff_features_table = []
    # Get features from wavelet coefficients
    for coeff, decomp_level_name in zip(list_wavelet_coefficients, list_of_wavelet_decomposition_level_names):
        coeff_table = pd.DataFrame(coeff, index=signals_table.index)
        coeff_features_table = get_signal_features_table(coeff_table)
        coeff_features_table.columns = [decomp_level_name + '_' +
                                        column_name for column_name in coeff_features_table.columns]
        coeff_features_table.reset_index(inplace=True)
        list_of_coeff_features_table.append(coeff_features_table)

    # Concatenate all features tables
    wavelet_coefficients_features_table = pd.concat(list_of_coeff_features_table, axis=1)
    # Drop duplicated label column
    wavelet_coefficients_features_table = wavelet_coefficients_features_table.loc[:,
                                                                                  ~wavelet_coefficients_features_table.columns.duplicated()]

    return wavelet_coefficients_features_table


def get_signal_with_wavelets_features_table(signals_table, waveletname, features_names=[
        'statistics', 'crossings', 'entropy'], include_original_signal=True, extra_features=None):
    import pandas as pd
    signal_features_table = get_wavelet_coefficients_features_table(signals_table, waveletname, features_names)
    if include_original_signal:
        # appends features table from the original signal (with labels column dropped)
        signal_features_table = pd.concat([signal_features_table, get_signal_features_table(
            signals_table, features_names, extra_features).reset_index(drop=True)], axis=1)
    return signal_features_table


def concat_features_tables(list_of_features_tables):
    import pandas as pd
    # Resets index by dropping label column on all tables except the first one
    list_of_features_tables = [
        table.reset_index(
            drop=True) if i != 0 else table.reset_index() for i,
        table in enumerate(list_of_features_tables)]
    features_table = pd.concat(list_of_features_tables, axis=1)
    return features_table
