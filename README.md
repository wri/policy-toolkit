Policy-toolkit
==============================

This repository contains code and data for the Restoration Research & Monitoring team's initiative to automate the identification of financial incentives and disincentives across policy contexts.

The `notebooks` folder contains Jupyter and RMarkdown notebooks for setting up the environment, preprocessing data, and performing manual and automatic data labeling.

The `data` folder contains data at each stage of the pipeline, from raw to interim to processed. Raw data are simply PDFs of policy documents. The ETL pipeline results in three `.csv` files. The `gold_standard.csv` contains ~1,100 paragraphs labeled manually, `noisy_labels.csv` contains ~5,000 paragraphs labeled with Snorkel, `unlabeled.csv` contains the rest of the text data, split into paragraphs and generally cleaned.

   * gold_standard.csv: ID, country, policy, page, text, class
   * noisy_labels.csv: ID, country, policy, page, text, class
   * unlabeled.csv: ID, country, policy, page, text

Analyzing this data requires the following steps:
   * Refinement of Snorkel data programming
   * Tokenization and cleaning of data (e.g. converting numbers to <number> tags)
   * Removing punctuation

The ideal modeling pipeline would be:
   * Snorkel labels -> Fine tune BERT classifier -> Test on Gold Standard dataset

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
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
