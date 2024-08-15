from pathlib import Path
import napari
from napari.utils import nbscreenshot
import numpy as np
import pandas as pd
from skimage.io import imread, imsave
# from napari_clusters_plotter._plotter import PlotterWidget
from napari_skimage_regionprops import add_table
from skimage.util import map_array
from napari_signal_selector.line import InteractiveFeaturesLineWidget
from scipy.stats import rankdata

image_path = r"C:\Users\mazo260d\Desktop\Fluo-N2DL-HeLa\01.tif"
labels_path = r"C:\Users\mazo260d\Desktop\Fluo-N2DL-HeLa\01_labels.tif"
tables_folder_path = r"C:\Users\mazo260d\Desktop\Fluo-N2DL-HeLa\01_track_tables"

image_path = Path(image_path)
labels_path = Path(labels_path)
image = imread(image_path).astype('uint16')
labels = imread(labels_path).astype('uint16')
print(image.shape)

Fiji_stack = imread(Path(r"C:\Users\mazo260d\Desktop\Fluo-N2DL-HeLa\01_Fiji_stack.tif"))

intensity_image = Fiji_stack[:,0]
label_image = Fiji_stack[:,1].astype('uint16')

tables_folder_path = Path(tables_folder_path)

table_list = []
for table_path in tables_folder_path.iterdir():
    if table_path.suffix == '.csv':
        if table_path.stem == 'edges':
            edges_table = pd.read_csv(table_path, skiprows=[1,2,3], encoding = "utf-8")
        elif table_path.stem == 'spots':
            spots_table = pd.read_csv(table_path, skiprows=[1,2,3], encoding = "utf-8")
        elif table_path.stem == 'tracks':
            tracks_table = pd.read_csv(table_path, skiprows=[1,2,3], encoding = "utf-8")

spots_and_tacks_table = pd.merge(left=spots_table, right=tracks_table, how='outer', on='TRACK_ID', suffixes=('_spots', '_tracks'))
spots_table = spots_and_tacks_table

def move_columns_to_beginning(df: "pandas.DataFrame", columns: list[str]):
    new_columns = columns + (df.columns.drop(columns).tolist())
    return df[new_columns]

spots_table['original_label'] = spots_table['MEDIAN_INTENSITY_CH2'].astype(int) # Getting the original labels by the median value of the "labels channel"
spots_table['frame'] = spots_table['FRAME'].astype(int)
# Sort by frame and original label
spots_table = spots_table.sort_values(by=['frame', 'original_label'])
# Add sequential unique labels after array is sorted
spots_table['label_unique'] = spots_table.reset_index().index + 1
spots_table['label'] = rankdata(spots_table['original_label'].values, method='dense')

# move columns
spots_table = move_columns_to_beginning(spots_table, columns=['frame', 'label', 'original_label', 'label_unique', 'TRACK_ID'])

# Remap labels to unique labels in a new label image
label_image_unique_labels = np.zeros_like(label_image)
for i in range(label_image.shape[0]):
    frame_mask = spots_table['frame'] == i
    # Remap labels to unique labels in a new label image
    original_labels_in_frame = spots_table.loc[frame_mask, 'original_label'].values
    new_labels_in_frame = spots_table.loc[frame_mask, 'label_unique'].values
    label_image_unique_labels[i] = map_array(label_image[i], original_labels_in_frame, new_labels_in_frame)

# Remove labels that are not in spots table
label_image2 = np.copy(label_image)
# Get label numbers that are in label image but not in spots table column "original_label"
labels_not_in_spots_table = np.setdiff1d(np.unique(label_image2), spots_table['original_label'].values)[1:]
for label in labels_not_in_spots_table:
    label_image2[label_image2 == label] = 0
# Remap labels to sequential labels in a new label image
label_image2 = map_array(label_image2, spots_table['original_label'].values, spots_table['label'].values)

viewer = napari.Viewer()
viewer.add_image(intensity_image)
labels_layer = viewer.add_labels(label_image2, features=spots_table)
add_table(labels_layer, viewer) # avoids current bug with status bar

widget = InteractiveFeaturesLineWidget(viewer)
viewer.window.add_dock_widget(widget, area='right')

napari.run()