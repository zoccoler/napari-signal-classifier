def make_list_of_coefficients_names(waveletname, max_level_of_decomposition):
    '''Create a list of wavelet coefficient names based on the wavelet name and maximum level of decomposition.
    '''
    list_coefficients_names = [waveletname +
                               '_cA_' + str(max_level_of_decomposition)]
    list_coefficients_names += [waveletname + '_cD_' +
                                str(i) for i in range(max_level_of_decomposition, 0, -1)]
    return list_coefficients_names


def extract_numbers_with_template(data_list, template):
    """Extract numbers from a list of strings based on a given template.

    Parameters
    ----------
    data_list : list
        List of strings to extract numbers from.
    template : str
        Template to match the numbers to. The template should be a string without numbers or dots.

    Returns
    -------
    list
        List of numbers extracted from the strings.
    """
    import re
    # Regular expression pattern to match numbers following the provided template
    pattern = rf'{template}(\d+(\.\d+)?)?'

    # List to store the extracted numbers as strings
    numbers_list = []

    # Loop through each string in the data_list and find the numbers using regex
    for item in data_list:
        match = re.search(pattern, item)
        if match:
            # Get the number part of the matched string
            number = match.group(1)
            if number is None:
                number = ''
            numbers_list.append(number)

    return numbers_list


def get_frequency_bands(decomp_level, sampling_frequency):
    '''Get frequency bands for approximation and detail coefficients at a given decomposition level.
    '''
    fcA = [0, (sampling_frequency / (2**(decomp_level + 1)))]
    fcD_decomp_level = [(sampling_frequency / (2**(decomp_level + 1))),
                        (sampling_frequency / (2**(decomp_level)))]
    return fcA, fcD_decomp_level


def plot_wavelet_coefficient_decomposition_levels(signals_table, waveletname, sampling_frequency, figsize=(15, 22)):
    '''Plot wavelet coefficient decomposition levels for a set of signals.
    
    Parameters
    ----------
    signals_table : pd.DataFrame
        DataFrame containing the signals to be decomposed.
    waveletname : str
        Name of the wavelet to be used for decomposition.
    sampling_frequency : float
        Sampling frequency of the signals.
    figsize : tuple, optional
        Figure size for the plot (default is (15, 22)).
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plots.
    ax : np.ndarray
        Array of axes objects for the plots.
    '''
    import pywt
    import matplotlib.pyplot as plt
    import numpy as np
    # Get decomposition levels
    max_level_of_decomposition = pywt.dwt_max_level(
        signals_table.shape[1], waveletname)
    # Get frequency bands

    frequency_band_list = [get_frequency_bands(
        max_level_of_decomposition, sampling_frequency)[0]]
    frequency_band_list += [get_frequency_bands(level, sampling_frequency)[1]
                            for level in reversed(range(1, max_level_of_decomposition + 1))]
    # Plot
    fig, ax = plt.subplots(max_level_of_decomposition + 1, 1, figsize=figsize)
    for i in range(signals_table.shape[0]):
        signal = signals_table.iloc[i, :]
        # Get coefficients
        coefficients = pywt.wavedec(
            signal, waveletname, level=max_level_of_decomposition, mode='per')
        for j, coef, freq_band in zip(range(len(coefficients)), coefficients, frequency_band_list):
            # First level is the approximation
            if j == 0:
                ax[j].plot(coef, label='label: ' + str(i))
                ax[j].set_title(
                    'Approximation Coefficient.\nFrequency band: ' + str(freq_band) + ' Hz')
            else:
                ax[j].plot(coef, label='label: ' + str(i))
                ax[j].set_title('Detail Coefficient: level ' + str(j) +
                                '.\nFrequency band: ' + str(freq_band) + ' Hz')
            ax[j].set_yticks([])

    plt.tight_layout()
    return fig, ax
