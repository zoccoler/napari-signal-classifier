import napari
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from skimage import morphology
from scipy import signal
from skimage.measure import label, regionprops_table
import pandas as pd
from napari_matplotlib.line import FeaturesLineWidget
from napari_signal_selector.line import InteractiveFeaturesLineWidget
from napari_signal_classifier._widget import Napari_Train_And_Predict_Signal_Classifier

seed = 42
np.random.seed(seed)



# Generate signals

freq = 5
time = np.linspace(0, 1, 500)
sawtooth = signal.sawtooth(2 * np.pi * freq * time) + 1
cosine = np.cos(2 * np.pi * freq * time) + 1
pulses = signal.square(2 * np.pi * freq * time) + 1

signals = [cosine, pulses, sawtooth]

# Generate objects

disk = morphology.disk(4)
square = morphology.square(9)
square[square == 1] = 2
diamond = morphology.diamond(4)
diamond[diamond == 1] = 3
objects_list = [disk, square, diamond]
image = np.zeros((100, 100))
object_order = []
for i in range(10):
    for j in range(3):
        n = np.random.randint(low=0, high=3, size=1)[0]
        obj = objects_list[n]
        image[i * 10: i * 10 + 9, 20 + j * 20:20 + j * 20 + 9] = obj
        object_order.append(n)


# Add signals to label image

label_image = label(image)
height, width = image.shape
time_points = time.shape[0]
time_lapse = np.zeros((time_points, height, width))
for i in range(label_image.max()):
    # print(object_order[i])
    # for label in [1, 2, 3]:
    mask = (label_image == i + 1)  # create boolean mask for current label
    signal = signals[object_order[i]]  # get corresponding signal for current label
    # Add random noise
    signal = signal + np.random.random(size=len(signal)) / 10
    # Add random offset
    signal = signal + np.random.random() * 5
    time_lapse[:, mask] = signal.reshape((time_points, -1))  # assign signal to region corresponding to current label

# Get intensities in a table

df = pd.DataFrame([])
for i in range(time_points):
    features = regionprops_table(label_image, intensity_image=time_lapse[i], properties=('label', 'mean_intensity'))
    features['frame'] = i
    # data = features.values()
    df = pd.concat([df, pd.DataFrame(features)])

array = np.zeros((label_image.max(), time_points))
for i in range(label_image.max()):
    intensities = df[df['label'] == i + 1]['mean_intensity'].values
    array[i, :] = intensities
time_array = np.stack([time] * array.shape[0], axis=0)

# Add to napari

metadata = {}

metadata['time-series-plugin'] = {}
metadata['time-series-plugin']['signals'] = array
metadata['time-series-plugin']['time'] = time_array

viewer = napari.Viewer()
averages = np.mean(time_lapse, axis=(1, 2))
viewer.add_image(time_lapse, metadata=metadata, name='time-lapse')

current_directory = Path(__file__).parent.resolve()
print(current_directory)

table = pd.read_csv(current_directory / '../src/napari_signal_classifier/data/signals_annotated.csv')

viewer.add_labels(label_image, features=table, name='labels')

# Call plotter

# plotter_widget = MetadataLine2DWidget(viewer)

# plotter_widget = FeaturesLineWidget(viewer)
plotter_widget = InteractiveFeaturesLineWidget(viewer)
viewer.window.add_dock_widget(plotter_widget)

plotter_widget.y_axis_key = 'mean_intensity'
plotter_widget.x_axis_key = 'frame'
plotter_widget.object_id_axis_key = 'label'

plotter_widget.update_line_layout_from_column(column_name='Annotations')

classifier_widget = Napari_Train_And_Predict_Signal_Classifier(viewer, plotter_widget)
viewer.window.add_dock_widget(classifier_widget, tabify=True, name='Signal Classifier Widget')

# TODO: Add signal classifier widget here !!!

napari.run()
