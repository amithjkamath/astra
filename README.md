ASTRA: Atomic Surface Transformations for Radiotherapy Quality Assurance
==============================

This repository describes ASTRA: Atomic Surface Transformations for Radiotherapy quality Assurance.

Project Organization
------------
    
    ├── astra              <- Source code for use in this project.
    │   ├── data                  <- Scripts to download or generate data
    │   ├── model                 <- Classes to define the model architecture and losses.
    │   ├── training              <- Classes to handle network training.
    │   ├── utils                 <- Scripts utilities used during data generation or training
    │   ├── validation            <- Scripts for validating DLDP model
    │   ├── visualization         <- MATLAB scripts for creating visualizations
    │   ├── __init.py             <- Package boilerplate.
    │   ├── test_with_perturb.py  <- Main test script with perturbation/transformation.
    │   ├── test.py               <- Classic testing script for the model.
    │   └── train.py              <- Model training script.
    │
    ├── data               <- This folder is empty; reach out for more details.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- This folder is empty; reach out for more details.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── results            <- Empty here; run notebook files for results here.
    │
    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── requirements.txt   <- Generated with `pip freeze > requirements.txt`.
    └── setup.py           <- boilerplate for using with pip.
