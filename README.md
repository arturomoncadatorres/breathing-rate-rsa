Estimating Breathing Rate from Respiratory Sinus Arrythmia
==============================

(Python) implementation of the paper ["Estimation of Breathing Rate from Respiratory Sinus Arrhythmia: Comparison of Various Methods"](https://link.springer.com/article/10.1007/s10439-007-9428-1) by Schafer and Kratky (2008).

<figure class="aligncenter">
	<img width="800" src="./multimedia/respiration_animation.gif"/>
</figure>

<sub><sup>GIF for display purposes only. You can find the code for generating it [here](./multimedia/make_gif.py). Original lung GIF by [PresenterMedia](https://www.presentermedia.com/powerpoint-animation/human-lungs-pa-pid-3795)</sup></sub>

## :bar_chart: Content
* [2 - Data pre-processing](https://github.com/arturomoncadatorres/breathing-rate-rsa/blob/main/notebooks/02-preprocessing.ipynb): 
* [3 - Data analysis](https://github.com/arturomoncadatorres/breathing-rate-rsa/blob/main/notebooks/03-analysis.ipynb): showcase of the different methods used to calculate breathing rate. Each method includes a short explanation and visualizations.

## :world_map: Roadmap
Although I am done with the implementation of the algorithms and the statistical methods for their validation, there are still quite a few things missing that I would like to work on. Namely:
* Data exploration
* Select segments with N annotations only
* Notebook for statistical evaluation
* Modularize visualizations
* Automate the whole pipeline using the `Makefile`


Project Organization
------------

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
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>