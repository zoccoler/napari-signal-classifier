import numpy as np
from scipy.interpolate import make_interp_spline
from collections import defaultdict

class SubSignal:
    _id_counter = 0  # Class variable to keep track of the last assigned ID
    def __init__(self, data, category, label, start_frame, end_frame):
        self.data = data  # The signal data as a numpy array
        self.category = category  # The class/annotation of the signal
        self.label = label  # The label of the signal
        self.start_frame = start_frame  # The starting frame of the signal segment
        self.end_frame = end_frame  # The ending frame of the signal segment
        self.id = SubSignal._get_next_id()  # Assign a unique ID

    @classmethod
    def _get_next_id(cls):
        cls._id_counter += 1
        return cls._id_counter

    def __repr__(self):
        return f"<SignalSegment id={self.id}, category={self.category}, label={self.label}, frames=[{self.start_frame}:{self.end_frame}), data_length={len(self.data)}>"

    def to_slice(self):
        return slice(self.start_frame, self.end_frame)

    def overlaps(self, other, threshold):
        range1 = set(range(self.start_frame, self.end_frame))
        range2 = set(range(other.start_frame, other.end_frame))
        intersection = len(range1.intersection(range2))
        union = len(range1.union(range2))
        jaccard_index = intersection / union if union else 0
        return jaccard_index > threshold

    def merge(self, other):
        new_start_frame = min(self.start_frame, other.start_frame)
        new_end_frame = max(self.end_frame, other.end_frame)
        new_length = new_end_frame + 1 - new_start_frame

        new_data = np.zeros(new_length)
        # Place the current data in the new_data array
        new_data[self.start_frame - new_start_frame:self.end_frame + 1 - new_start_frame] = self.data
        
        # Calculate the slice for the other data
        other_slice_start = other.start_frame - new_start_frame
        other_slice_end = other.end_frame + 1 - new_start_frame

        # Ensure the slices are correctly aligned
        new_data[other_slice_start:other_slice_end] = other.data

        self.data = new_data
        self.start_frame = new_start_frame
        self.end_frame = new_end_frame
        self.id = SubSignal._get_next_id()  # Assign a new unique ID to the merged result

    def interpolate_samples(self, n_samples):
        # Resample the signal segment to a fixed number of samples using spline interpolation
        x = np.arange(len(self.data))
        x_new = np.linspace(0, len(self.data), n_samples)
        spline = make_interp_spline(x, self.data, k=3)
        return spline(x_new)

class SubSignalCollection:
    def __init__(self):
        self.sub_signals = []
        self.categories = []
        self.max_length_per_category = defaultdict(int)

    def add_sub_signal(self, sub_signal):
        self.sub_signals.append(sub_signal)
        if sub_signal.category not in self.categories:
            self.categories.append(sub_signal.category)
        self.max_length_per_category[sub_signal.category] = max(self.max_length_per_category[sub_signal.category], len(sub_signal.data))
    
    def __repr__(self):
        return f"<SubSignalCollection signal categories={self.categories}, number of signals={len(self.sub_signals)}>"

    def sort_by_category(self):
        self.sub_signals = sorted(self.sub_signals, key=lambda x: x.category)

    def merge_subsignals(self, threshold):
        merged = []
        for subsignal in self.sub_signals:
            merged_with_existing = False
            for m in merged:
                if subsignal.label == m.label and subsignal.overlaps(m, threshold): # and subsignal.category != m.category
                    m.merge(subsignal)
                    m.category = f"{m.category}-{subsignal.category}"
                    merged_with_existing = True
                    break
            if not merged_with_existing:
                merged.append(subsignal)
        self.sub_signals = merged
    
def extract_sub_signals_by_annotations_from_arrays(signal_data, annotations, labels, frames):
    """
    Extracts sub-signals from a signal data array based on annotations.

    Parameters
    ----------
    signal_data : np.ndarray
        The signal data array.
    annotations : np.ndarray
        The annotations array.
    labels : np.ndarray
        The labels array.
    frames : np.ndarray
        The frames array.
    
    Returns
    -------
    list
        A list of SignalSegment objects.
    """
    sub_signal_collection = SubSignalCollection()
    current_signal_category = 0
    start_index = None

    for i, ann in enumerate(annotations):
        if ann > 0:  # Signal detected
            if current_signal_category == 0:  # New signal starts
                current_signal_category = ann
                start_index = i
            elif current_signal_category != ann:  # Different signal detected, save previous
                # Adjusted to include label and frame information
                sub_signal = SubSignal(signal_data[start_index:i], 
                                       current_signal_category, 
                                       labels[start_index], 
                                       frames[start_index], 
                                       frames[i])
                sub_signal_collection.add_sub_signal(sub_signal)
                
                current_signal_category = ann
                start_index = i
        else:  # Noise detected, save previous signal
            if current_signal_category > 0:
                sub_signal = SubSignal(signal_data[start_index:i],
                                       current_signal_category,
                                       labels[start_index],
                                       frames[start_index],
                                       frames[i])
                sub_signal_collection.add_sub_signal(sub_signal)
                
                current_signal_category = 0
                start_index = None
    
    # Handle the case where the last signal goes until the end
    if current_signal_category > 0:
        sub_signal = SubSignal(signal_data[start_index:],
                               current_signal_category,
                               labels[start_index],
                               frames[start_index],
                               frames[-1])
        sub_signal_collection.add_sub_signal(sub_signal)
    
    return sub_signal_collection

def extract_sub_signals_by_annotations(table, column_signal_value, column_signal_id, column_signal_annotation, column_frame):
    """
    Extracts sub-signals from a signal data array based on annotations.

    Parameters
    ----------
    table : pd.DataFrame
        The DataFrame containing the signal data.
    column_signal_value : str
        The column name containing the signal data.
    column_signal_id : str
        The column name containing the signal ID (usually the label from image).
    column_signal_annotation : str
        The column name containing the signal annotations.
    column_frame : str
        The column name containing the frame information (time).
    
    Returns
    -------
    list
        A list of SignalSegment objects.
    """
    signal_data = table[column_signal_value].values
    annotations = table[column_signal_annotation].values
    labels = table[column_signal_id].values
    frames = table[column_frame].values
    return extract_sub_signals_by_annotations_from_arrays(signal_data, annotations, labels, frames)