# napari-signal-classifier

[![License BSD-3](https://img.shields.io/pypi/l/napari-signal-classifier.svg?color=green)](https://github.com/zoccoler/napari-signal-classifier/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-signal-classifier.svg?color=green)](https://pypi.org/project/napari-signal-classifier)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-signal-classifier.svg?color=green)](https://python.org)
[![tests](https://github.com/zoccoler/napari-signal-classifier/workflows/tests/badge.svg)](https://github.com/zoccoler/napari-signal-classifier/actions)
[![codecov](https://codecov.io/gh/zoccoler/napari-signal-classifier/branch/main/graph/badge.svg)](https://codecov.io/gh/zoccoler/napari-signal-classifier)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-signal-classifier)](https://napari-hub.org/plugins/napari-signal-classifier)

A napari plugin that classifies annotated signals stored in a table in the .features of a Labels layer using scikit-learn RandomForest classifier.

It also provides a sub-signal classifier that can be used to classify sub-signals inside time-series. First it detects sub-sginals with a template matching algorithm and then classifies them also using scikit-learn RandomForest classifier.

This plugin employs and works in synergy with the [napari-signal-selector plugin](https://github.com/zoccoler/napari-signal-selector?tab=readme-ov-file#napari-signal-selector). Take a look at it to see how to annotate signals in a plotter linked to a napari Labels layer with the .features attribute.

# Quick Demo

## Napari Signal Classifier

![demo](https://github.com/zoccoler/napari-signal-classifier/raw/main/images/signal_classifier_demo.gif)

## Napari Sub-Signal Classifier

![demo](https://github.com/zoccoler/napari-signal-classifier/raw/main//images/sub_signal_classifier_demo.gif)

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

## Installation

You can install `napari-signal-classifier` via [pip]:

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
