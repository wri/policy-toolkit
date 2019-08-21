Policy-toolkit
==============================

This repository contains code and data for the Restoration Research & Monitoring team's initiative to automate the identification of financial incentives and disincentives across policy contexts.

## Notebooks

The `notebooks` folder contains Jupyter and RMarkdown notebooks for setting up the environment, preprocessing data, and performing manual and automatic data labeling.

   * 1-environment-setup: Set up jupyter environment (alternative to Docker)
   * 2-extract-transfer-load: Extract text and disaggregate to paragraphs
   * 3-data-labelling: Manual gold standard data creation
   * 4-automatic-data-labeling: Automatic data labeling with data programming in Snorkel
   * 5-roberta-classification: Embed paragraphs as features with roBERTa model
   * 6-end-model: Train a noise-aware end model with snorkel metal label classifier output

## Data

The `data` folder contains data at each stage of the pipeline, from raw to interim to processed. Raw data are simply PDFs of policy documents. The ETL pipeline results in two `.csv` files. The `gold_standard.csv` contains ~1,100 paragraphs labeled manually, and the `noisy_labels.csv` contains ~16,000 paragraphs (soon to be >30,000) labeled with Snorkel.

   * gold_standard.csv: ID, country, policy, page, text, class
   * noisy_labels.csv: ID, country, policy, page, text, (class distributions)
   
## Modeling ethos

This project uses data programming to algorithmically label training data based on a small, hand-made gold standard. Soft labels are assigned as probability distributions of label likelihood based on the weak algorithmic labels. These soft labels are used in a soft implementation of cross entropy.

Models are trained with algorithmically labeled samples and evaluated on the gold standard labels. The current pipeline is noisy labeling -> roBERTa encoding -> LSTM.

Future iterations will fine tune roBERTa, add additional feature engineering, and update the noisy labeling process.

## Roadmap

**Priorities for WRI team**
   * Second validation for gold standard
   * Refine snorkel data programming
   * Make the workflow from notebook to notebook more clear

**Priorities for Columbia team**
   * Pilot implementation of BabbleLabble [link](https://github.com/HazyResearch/babble)
   * Additional feature engineering including:
      * SpaCy dependency parsing
      * Named entity recognition
      * Topic modeling
      * Universal sentence encoder
      * Hidden markov model
      * DBPedia linking
   * Data augmentation with synonym replacement [link](https://www.snorkel.org/use-cases/02-spam-data-augmentation-tutorial)
   * Model augmentation with slicing functions [link](https://www.snorkel.org/use-cases/03-spam-data-slicing-tutorial)
   * Fine tune roBERTa on noisy labels
   * Massive multi task learning with snorkel 0.9
   * Named entity disambiguation from positive class paragraphs: (finance_type, finance_amount, funder, fundee)

## References

   * [BERT](https://arxiv.org/pdf/1810.04805.pdf)
   * [RoBERTA](https://arxiv.org/pdf/1907.11692.pdf)
   * [Additional feature engineering improves BERT performance](http://web.stanford.edu/class/cs224n/reports/default/15791958.pdf)
   * [Snorkel](https://dawn.cs.stanford.edu/pubs/snorkel-nips2016.pdf)
   * [Snorkel-metal](https://arxiv.org/pdf/1810.02840.pdf)
   * retrieveR ([github](https://github.com/wri/retrieveR), [paper](https://arxiv.org/pdf/1908.02425.pdf))
   * [NLP for policy analysis](https://web.stanford.edu/~jgrimmer/tad2.pdf)
   * [Topic modeling for policy](https://www.tandfonline.com/doi/full/10.1080/14693062.2019.1624252)
   * [Policy analysis and restoration](https://pdfs.semanticscholar.org/4bc7/af30a8ec6f325cd15da54cc5973f8be49240.pdf)

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── Dockerfile         <- Dockerfile to create environment
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
