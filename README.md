Policy-toolkit
==============================

This repository contains code and data for the Restoration Research & Monitoring team's initiative to automate the identification of financial incentives and disincentives across policy contexts.

The `notebooks` folder contains Jupyter and RMarkdown notebooks for setting up the environment, preprocessing data, and performing manual and automatic data labeling.

   * 1-environment-setup: Set up jupyter environment (alternative to Docker)
   * 2-extract-transfer-load: Extract text and disaggregate to paragraphs
   * 3-data-labelling: Manual gold standard data creation
   * 4-automatic-data-labeling: Automatic data labeling with data programming in Snorkel
   * 5-roberta-classification: Embed paragraphs as features with roBERTa model
   * 6-end-model: Train a noise-aware end model with snorkel metal label classifier output

The `data` folder contains data at each stage of the pipeline, from raw to interim to processed. Raw data are simply PDFs of policy documents. The ETL pipeline results in three `.csv` files. The `gold_standard.csv` contains ~1,100 paragraphs labeled manually, `noisy_labels.csv` contains ~5,000 paragraphs labeled with Snorkel, `unlabeled.csv` contains the rest of the text data, split into paragraphs and generally cleaned.

   * gold_standard.csv: ID, country, policy, page, text, class
   * noisy_labels.csv: ID, country, policy, page, text, class
   * unlabeled.csv: ID, country, policy, page, text

Analyzing this data requires the following steps:
   * Refinement of Snorkel data programming
   * Tokenization and cleaning of data (e.g. converting numbers to <number> tags)
   * Removing punctuation

Major to-dos include:
   * Refine snorkel data programming
   * Create gold standard classifier (vanilla RNN)
   * Create gold standard + roBERTa classifier
   * Create snorkel-metal classifier (simple embeddings)
   * Create roBERTA -> snorkel-metal classifier (LSTM of feature embeddings -> multitask head)

Priorities for columbia team:
   * Additional feature engineering (SpaCy dependency parsing, NER, topic modeling, USE, etc.)
   * Named entity disambiguation from positive class paragraphs: (finance_type, finance_amount, funder, fundee)
   
References:
   * [BERT](https://arxiv.org/pdf/1810.04805.pdf)
   * [RoBERTA](https://arxiv.org/pdf/1907.11692.pdf)
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
