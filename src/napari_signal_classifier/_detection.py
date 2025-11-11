import numpy as np
import matplotlib.pyplot as plt
from dtaidistance import dtw, preprocessing


def normalize(signal, method='zscores'):
    '''Normalize a signal using the specified method.

    Parameters
    ----------
    signal : np.ndarray
        The input signal to be normalized.
    method : str, optional
        The normalization method to use ('zscores' or 'minmax'). Default is 'zscores'.

    Returns
    -------
    np.ndarray
        The normalized signal.
    '''
    if method == 'zscores':
        return (signal - np.mean(signal)) / np.std(signal)
    elif method == 'minmax':
        return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def align_signals(reference, signal, detrend=False, smooth=0.1):
    '''Align a signal to a reference using Dynamic Time Warping (DTW).

    Parameters
    ----------
    reference : np.ndarray
        The reference signal.
    signal : np.ndarray
        The signal to be aligned.
    detrend : bool, optional
        Whether to detrend the signals before alignment. Default is False.
    smooth : float, optional
        Smoothing factor for detrending. Default is 0.1.

    Returns
    -------
    np.ndarray
        The aligned signal.
    '''
    if detrend:
        signal_warp = preprocessing.differencing(signal, smooth=smooth)
        reference_warp = preprocessing.differencing(reference, smooth=smooth)
    else:
        signal_warp = signal
        reference_warp = reference
    alignment = dtw.warping_path_fast(reference_warp, signal_warp)

    aligned_signal = np.zeros_like(reference)
    for (i, j) in alignment:
        aligned_signal[i] = signal[j]
    return aligned_signal


def generate_template_mean(replicates, detrend=False, smooth=0.1):
    '''Generate a template signal from a list of replicates using median alignment.

    Parameters
    ----------
    replicates : list of np.ndarray
        List of signal replicates.
    detrend : bool, optional
        Whether to detrend the signals before alignment. Default is False.
    smooth : float, optional
        Smoothing factor for detrending. Default is 0.1.
    
    Returns
    -------
    np.ndarray
        The generated template signal.
    '''
    # Use the median signal as the initial reference
    median_signal = np.median(replicates, axis=0)

    # Align all replicates to the median signal
    aligned_replicates = [align_signals(
        median_signal, rep, detrend, smooth) for rep in replicates]

    # Compute the average to form the template
    template = np.mean(aligned_replicates, axis=0)
    return template


def generate_templates_by_category(sub_signal_collection, detrend=False, smooth=0.1):
    """
    Generate templates by category from a list of SignalSegment objects.

    Parameters
    ----------
    sub_signal_collection : SubSignalCollection
        The collection of signal segments.

    """
    # Sort the signal segments by category
    sub_signal_collection.sort_by_category()

    # Initialize variables
    templates_by_category = {}
    current_category = None
    sub_signals_with_current_category = []
    n_samples = max(sub_signal_collection.max_length_per_category.values())

    # Process each signal segment
    for sub_signal in sub_signal_collection.sub_signals:
        if sub_signal.category != current_category:
            # If the category changes, process the current category
            if sub_signals_with_current_category:
                template = generate_template_mean(
                    sub_signals_with_current_category, detrend, smooth)
                templates_by_category[current_category] = template
                sub_signals_with_current_category = []

            # Update the current category
            current_category = sub_signal.category

        # Normalize and resample the current segment
        # target_length = sub_signal_collection.max_length_per_category[current_category]
        resampled_norm_data = normalize(
            sub_signal.interpolate_samples(n_samples), method='zscores')
        sub_signals_with_current_category.append(resampled_norm_data)

    # Process the last category
    if sub_signals_with_current_category:
        template = generate_template_mean(
            sub_signals_with_current_category, detrend, smooth)
        templates_by_category[current_category] = template

    return templates_by_category


def detect_sub_signal_by_template(composite_signal, template, threshold, return_cross_corr=False, norm_method='zscores'):
    '''Detect sub-signals in a composite signal using cross-correlation with a template.

    Parameters
    ----------
    composite_signal : np.ndarray
        The composite signal in which to detect sub-signals.
    template : np.ndarray
        The template signal to use for detection.
    threshold : float
        The threshold for peak detection as a fraction of the maximum cross-correlation value.
    return_cross_corr : bool, optional
        Whether to return the cross-correlation array along with peak indices. Default is False.
    norm_method : str, optional
        The normalization method to use ('zscores' or 'minmax'). Default is 'zscores'.
    
    Returns
    -------
    peaks_indices : np.ndarray
        Indices of detected peaks in the composite signal.
    normalized_corr : np.ndarray, optional
        The normalized cross-correlation array (returned if return_cross_corr is True).
    '''
    from scipy import signal
    signal_norm = normalize(composite_signal, method=norm_method)
    template_norm = normalize(template, method=norm_method)
    # default method already chooses between fft and direct
    cross_corr = signal.correlate(signal_norm, template_norm, mode='same')

    if norm_method == 'minmax':
        # Normalizing cross-correlation by minmax directly
        normalized_corr = normalize(cross_corr, method=norm_method)
    elif norm_method == 'zscores':
        # Normalizing cross-correlation by energy of signal and template
        # Convolution of the squared composite signal with a window of ones
        fm2 = signal.correlate(
            signal_norm**2, np.ones_like(template_norm), mode='same')
        # Convolution of the composite signal with a window of ones
        fm = signal.correlate(
            signal_norm, np.ones_like(template_norm), mode='same')
        n = len(template)
        denominator = np.sqrt(fm2 - fm**2/n)
        normalized_corr = cross_corr/denominator

    threshold = np.max(normalized_corr) * threshold
    peaks_indices, _ = signal.find_peaks(normalized_corr, height=threshold)
    if return_cross_corr:
        return peaks_indices, normalized_corr
    return peaks_indices


def extract_sub_signals_by_templates(df, column_signal_value, column_signal_id, column_frame, sub_signal_templates, threshold, method='zscores'):
    '''Extract sub-signals from composite signals in a DataFrame using provided templates.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing composite signals in long format.
    column_signal_value : str
        Column name containing the composite signal values.
    column_signal_id : str
        Column name containing the signal ID (usually the label from image).
    column_frame : str
        Column name containing the frame information (time).
    sub_signal_templates : dict
        Dictionary of sub-signal templates by category.
    threshold : float
        Threshold for peak detection as a fraction of the maximum cross-correlation value.
    method : str, optional
        Normalization method to use ('zscores' or 'minmax'). Default is 'zscores'.
    
    Returns
    -------
    SubSignalCollection
        A collection of extracted sub-signals.
    '''
    from napari_signal_classifier._sub_signals import SubSignal, SubSignalCollection
    sub_signal_collection = SubSignalCollection()
    grouped_by_label = df.groupby(column_signal_id, sort=False)
    for label, sub_table in list(grouped_by_label):
        # column_signal_value = 'mean_intensity'
        composite_signal = sub_table[column_signal_value].values
        # for each composite signal, detect sub_signals using the templates
        for k, template in sub_signal_templates.items():
            peaks_indices = detect_sub_signal_by_template(
                composite_signal, template, threshold, norm_method=method)
            # Collect the sub_signals around each peak
            for peak_index in peaks_indices:
                # Add the sub_signal to the collection with the template category, original label and frame information
                start_index = int(peak_index-np.floor(len(template)/2))
                end_index = int(peak_index+np.floor(len(template)/2))
                # If sub_signal would start before composite signal, set start frame to 0
                if start_index < 0:
                    start_index = 0
                # If sub_signal would end after composite signal, set end frame to the last frame
                if end_index > len(composite_signal):
                    end_index = len(composite_signal)
                sub_signal = SubSignal(composite_signal[start_index:end_index],
                                       k,
                                       label,
                                       sub_table[column_frame].values[start_index],
                                       sub_table[column_frame].values[end_index-1]
                                       )
                sub_signal_collection.add_sub_signal(sub_signal)
    return sub_signal_collection
