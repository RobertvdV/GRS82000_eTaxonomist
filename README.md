MSc Thesis topic: eTaxonomist: Explainable AI for biodiversity classification
==============================

With an estimated 50% of animal species yet to be discovered, and many going extinct before they are ever described, it is more important than ever to reinforce the effort of monitoring known species and describing new ones. As a first step towards automating this process, we need to gather as much information as possible about how expert taxonomists distinguish between different species. This information can then be used to teach an Artificial Intelligence (AI) model to communicate like a taxonomist.

Current AI systems based on deep learning allow for incredible feats, such as to automatically identify a plant or animal species based on an image (for instance, with the iNaturalist app). However, these models still have some serious drawbacks: 1) They are black boxes, meaning that it is very hard to understand the process that has led to a certain result. This prevents experts from spotting potential mistakes and amateurs to learn from the system; 2) They are quite rigid, and it is not straightforward to detect whether an image belongs to an unknown species or to add new species to the system.

This project aims at solving this problem by splitting the system into two agents that communicate using natural language. The first is a visual-language hybrid model that takes an image as input and generates descriptions of the image in natural language using the vocabulary of expert taxonomists, but is not aware of species names. The second is a pure language model that takes as input the description provided by the first and outputs the corresponding species name.

Goals of this project:
- Develop a web-crawling system able to retrieve web pages with species descriptions.
- Train a Natural Language Processing (NLP) model able to identify whether a paragraph is part of a species description or something else.
- Train an NLP model able to infer a species name based on partial descriptions.

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
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
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
