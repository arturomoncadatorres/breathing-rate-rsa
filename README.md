# Estimating Breathing Rate from Respiratory Sinus Arrhythmia
The purpose of this repository is to provide the (Python) implementation of different methods to calculate (average) respiratory rate from heart rate variability, as shown in the paper

> Schäfer, Axel, and Karl W. Kratky. ["Estimation of Breathing Rate from Respiratory Sinus Arrhythmia: Comparison of Various Methods"](https://link.springer.com/article/10.1007/s10439-007-9428-1), Annals of Biomedical Engineering 36.3 (2008): 476-485.

Full credit goes to the authors.

<p align="center">
  <a href="#bar_chart-content">Content</a> •
  <a href="#world_map-roadmap">Roadmap</a> •
  <a href="#card_index_dividers-project-organization">Project Organization</a> •
  <a href="#page_with_curl-license">License</a> •
  <a href="#label-credits">Credits</a>
</p>

<figure class="aligncenter">
	<img width="800" src="./multimedia/respiration_animation.gif"/>
</figure>

<sub><sup>GIF for display purposes only. You can find the code for generating it [here](./multimedia/make_gif.py). Original lung GIF by [PresenterMedia](https://www.presentermedia.com/powerpoint-animation/human-lungs-pa-pid-3795)</sup></sub>

## :bar_chart: Content
* 1 - Data exploration (WIP)
* [2 - Data pre-processing](https://github.com/arturomoncadatorres/breathing-rate-rsa/blob/main/notebooks/02-preprocessing.ipynb): preparation of the signals needed for further analysis.
* [3 - Data analysis](https://github.com/arturomoncadatorres/breathing-rate-rsa/blob/main/notebooks/03-analysis.ipynb): showcase of the different methods used to calculate breathing rate. Each method includes a short explanation and visualizations.
* 4 - Statistical evaluation (WIP): comparison of performance of the different methods using various metrics.

## :world_map: Roadmap
Although I am more or less done with the implementation of the algorithms and the statistical methods for their validation, there are still quite a few things missing that I would like to work on. Namely:
* Create notebook for data exploration
* Create notebook for statistical evaluation
* Select segments with N annotations only
* Modularize visualizations
* Automate the whole pipeline using the `Makefile`


## :card_index_dividers: Project Organization


    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── multimedia         <- Multimedia for the project (and corresponding scripts)
    │
    ├── notebooks          <- Jupyter notebooks. Each notebook corresponds to a section of the pipeline.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make predictions
    │   │
    │   ├── signals        <- Scripts to process different type of signals
    │   │
    │   ├── statistics     <- Scripts used for statistical evaluation of the different methods
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


## :page_with_curl: License
This repository uses the MIT license

## :label: Credits
This repository was developed in [Spyder](https://www.spyder-ide.org/) (a fantastic open-source Python IDE) while leveraging [Jupytext](https://github.com/mwouts/jupytext) for easy creation of Jupyter Notebooks. It used [Cookiecutter](https://github.com/cookiecutter/cookiecutter) and the [cookiecutter data science project template](https://drivendata.github.io/cookiecutter-data-science/) #cookiecutterdatascience.

If you have any questions, comments, or feedback, please [open a discussion](https://github.com/arturomoncadatorres/breathing-rate-rsa/discussions). If there is a problem with the code (e.g., bug), please [open an issue](https://github.com/arturomoncadatorres/breathing-rate-rsa/issues). Moreover, you can always drop me a line on Twitter [(@amoncadatorres)](https://twitter.com/amoncadatorres).
