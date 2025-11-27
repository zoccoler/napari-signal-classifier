# napari-signal-classifier

[![License BSD-3](https://img.shields.io/pypi/l/napari-signal-classifier.svg?color=green)](https://github.com/zoccoler/napari-signal-classifier/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-signal-classifier.svg?color=green)](https://pypi.org/project/napari-signal-classifier)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-signal-classifier.svg?color=green)](https://python.org)
[![tests](https://github.com/zoccoler/napari-signal-classifier/workflows/tests/badge.svg)](https://github.com/zoccoler/napari-signal-classifier/actions)
[![codecov](https://codecov.io/gh/zoccoler/napari-signal-classifier/branch/main/graph/badge.svg)](https://codecov.io/gh/zoccoler/napari-signal-classifier)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-signal-classifier)](https://napari-hub.org/plugins/napari-signal-classifier)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17727926.svg)](https://doi.org/10.5281/zenodo.17727926)

A napari plugin that classifies annotated signals stored in a table in the `.features` of a `Labels` layer using a classifier.

It also provides a sub-signal classifier that can be used to classify local patterns inside longer time-series. First it detects sub-sginals with a template matching algorithm based on annotated sub-signals. Then, it trains and run predictions with a chosen classifier. At the moment, only  [scikit-learn RandomForest classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) option is implemented.

This plugin employs and works in synergy with the [napari-signal-selector plugin](https://github.com/zoccoler/napari-signal-selector?tab=readme-ov-file#napari-signal-selector). Take a look at it to see how to annotate signals in a plotter linked to a napari `Labels` layer with signals table stored in the `.features` attribute.

[Jump to Intallation](#installation)

# Usage

## Napari Signal Classifier

After having annotated signals in the `.features` of a `Labels` layer (check the [napari-signal-selector plugin](https://github.com/zoccoler/napari-signal-selector?tab=readme-ov-file#napari-signal-selector)), you can train a signal classifier and predict the labels of unannotated signals.

Open the `Train and Predict Signal Classifier` widget from the napari menus in `Layers > Classify > Signal / Time-series > Train and Predict Signal Classifier`. The widget will appear in the right panel of napari (see image below). It will also cast the `Signal Selector and Annotator` widget from napari-signal-selector to display the signals (remember to update the comboboxes to display the signals).

![signal_classifier_widget](signal_classifier_widget.png)

1. Choose the right `Labels` layer in the `Labels Layer with Signals Table` field.
2. Choose the classifier (currently only RandomForest is implemented).
3. Optionally provide a path to a folder where to save the trained model. The model file name is unique and automatically generated. If you provide here a link to a `.pkl` file previously trained with this plugin, it will just run predictions using the provided classifier.
4. Select the number of trees (estimators) of the RandomForest.
5. Choose a number for the random state for reproducibility. Choosing `-1` will pass `None` to `random_state` of scikit-learn's [](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) and [RamdomForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html), meaning the outcome will not be deterministic.
6. Set the training percentage (the percentage of annotated signals that will be used for training, the rest will be used for testing the model and showing the accuracy) The initial value is 70%.
7. Check the "Stratify" checkbox to perform stratified splitting of the annotated signals into train and test sets. This is recommended when you have imbalanced classes in your annotations.
8. Click on "Train and Predict".

Be patient as several signal features will be calculated automatically from the signals and the classifier will be trained on the annotated signals training set and evaluated on the test set. Also, the labels of unannotated signals will be predicted. The predicted labels will be stored in a new column, called "Predictions", in the `.features` of the same `Labels` layer. All parameters and the model's accuracy on the train and test sets will be printed in the napari console and saved as a `.json` file in the same place and with the same name as the trained classifier. A new `Labels` layer will be created to show the predicted labels on the signals.

![demo](https://github.com/zoccoler/napari-signal-classifier/raw/main/images/signal_classifier_demo.gif)

The resulting `.features` table can be viewed via the native napari features table widget (`Layers > Visualize > Features Table Widget`) and exported to a `.csv` file from there for downstream analysis.

## Napari Sub-Signal Classifier

After having annotated signals in the `.features` of a `Labels` layer (check the [napari-signal-selector plugin](https://github.com/zoccoler/napari-signal-selector?tab=readme-ov-file#napari-signal-selector)), you can train a sub-signal classifier to classify local patterns (sub-signals) inside time-series.

Open the `Train and Predict Sub-Signal Classifier` widget from the napari menus in `Layers > Classify > Signal / Time-series > Train and Predict Sub-Signal Classifier`. The widget will appear in the right panel of napari. It will also cast the `Signal Selector and Annotator` widget from napari-signal-selector to display the signals (remember to update the comboboxes to display the signals).

![sub_signal_classifier_widget](sub_signal_classifier_widget.png)

Choose the same parameters as the [Napari Signal Classifier](#napari-signal-classifier), plus a few additional parameters related to sub-signal detection and merging: 
1. Choose the "Detection Threshold" for the template matching algorithm (default 0.8, ranging from 0 to 1). This can be seen as a sensitivity to detection parameter. Sub-signals will be disconsidered if peaks in the cross-correlation with template are lower than this value.
2. Choose whether to apply detrending to the annotated sub-signals (default unchecked). If checked, the first order derivative of each annotated sub-signal will be calculated and used for aligning sub-signals of the same class to generate the template.
3. If "Detrend" is checked, choose a smooth factor (default 0.1, ranging from 0 to 0.5) to be applied to the first order derivative. It corresponds to the fraction of the highest frequencies removed. 
4. Choose the "Merging Overlap Threshold" (default 0.5) for merging overlapping detected sub-signals. This is the minimal amount of overlap (Jaccard-index) in time between detected sub-signals to have them merged and considered a single detection. Increase this if you have sub-signals too close to each other and want to have them separated.
5. Click on "Train and Predict".

Be patient as a signal template for each class will be generated by the median of annotated sub-signals and a cross-correlation based template matching algorithm will be used to detect these templates in each unannotated time-series (signal). Several signal features will be calculated automatically from the detected sub-signals, a classifier will be trained on the annotated sub-signals training set and evaluated on the test set. Also, the detection + classification algorithms will be applied to all unannotated signals. The predicted labels will be stored in a new column in the `.features` of the `Labels` layer called "Predictions". All parameters and the model's accuracy on the train and test sets will be printed in the napari console and saved as a `.json` file in the same place and with the same name as the trained classifier. A new `Labels` layer will be created to show the predicted labels on the signals.

![demo](https://github.com/zoccoler/napari-signal-classifier/raw/main//images/sub_signal_classifier_demo.gif)

Again, the resulting `.features` table can be viewed via the native napari features table widget (`Layers > Visualize > Features Table Widget`) and exported to a `.csv` file from there for downstream analysis.

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

## Installation

You can install `napari-signal-classifier` via [pip]. Follow these steps from a terminal.

We recommend using `Miniforge` whenever possible. Click [here](https://conda-forge.org/download/) to choose the right download option for your OS.
**If you do not use `Miniforge`, you might need to replace the `mamba` term whenever you see it below with `conda`.**

Create a conda environment :

    mamba create -n nsc-env napari pyqt python=3.12
    
Activate the environment :

    mamba activate nsc-env

Install `napari-signal-classifier` via [pip] :

    pip install napari-signal-classifier

To install latest development version :

    pip install git+https://github.com/zoccoler/napari-signal-classifier.git


## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"napari-signal-classifier" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/zoccoler/napari-signal-classifier/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
