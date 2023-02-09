ASTRA
==============================

This repository describes ASTRA: Atomic Surface Transformations for Radiotherapy quality Assurance.

To create a new test set, run the following scripts (in the astra/utils/ folder) in this specific order:
1. rtss_to_nifti.py
2. convert_dose_volume.py
3. resize_to_standard_dimensions.py

Then train the model using the train.py script (modifying the paths to the data files of course).

and finally, generate the ASTRA sensitivity maps using test_with_perturb.py

Project Organization
------------

    ├── LICENSE
    ├── README.md                               <- The top-level README for developers using this project.
    ├── data
    │   ├── processed                           <- The final, canonical data sets for modeling.
    │   └── raw                                 <- The original, immutable data dump.
    │
    ├── docs                                    <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models                                  <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks                               <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                                              the creator's initials, and a short `-` delimited description, e.g.
    │                                              `1.0-jqp-initial-data-exploration`.
    │
    ├── references                              <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── results                                <- Generated analysis as HTML, PDF, LaTeX, etc.
    │
    ├── requirements.txt                        <- The requirements file for reproducing the analysis environment, e.g.
    │                                              generated with `pip freeze > requirements.txt`
    │
    └── astra           <- Source code for use in this project.
       │
       ├── data                                <- Scripts to download or generate data
       │
       ├── model                               <- Classes to define the model architecture and losses.
       │
       ├── training                            <- Classes to handle network training.
       │
       ├── utils                               <- Scripts utilities used during data generation or training
       └── validation                          <- Classes to evaluate model performance.