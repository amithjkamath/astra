# ASTRA: Atomic Surface Transformations for Radiotherapy Quality Assurance

![EMBC 2023](https://img.shields.io/badge/Conference-EMBC%202023-blue)

This repository accompanies our [award-winning](https://embc.embs.org/2023/students/paper-competition/) paper:

**"ASTRA: Atomic Surface Transformations for Radiotherapy Quality Assurance"**  
Presented at the 45th IEEE Engineering in Medicine and Biology Conference (EMBC), 2023.

**Authors:** Amith Kamath, Robert Poel, Jonas Willmann, Ekin Ermis, Nicolaus Andratschke, Mauricio Reyes

See a short video description of this work here:

[<img src="https://i.ytimg.com/vi/vghlJh8ACOY/maxresdefault.jpg" width="50%">](https://youtu.be/vghlJh8ACOY "ASTRA")

🔗 [Project Website](https://amithjkamath.github.io/projects/2023-embc-astra/)

---

## Overview

ASTRA introduces a deep learning-based framework to assess the sensitivity of radiotherapy dose predictions to local variations in organ-at-risk (OAR) segmentations. By simulating atomic surface transformations, ASTRA provides clinicians with dose-aware sensitivity maps, highlighting regions where segmentation inaccuracies could significantly impact dose distributions.

---

## Key Contributions

- **Dose-Aware Sensitivity Mapping:** ASTRA predicts the potential impact of local segmentation variations on radiotherapy dose distributions, aiding in quality assurance.
- **Simulation of Segmentation Variability:** Introduces a method to simulate realistic local perturbations in OAR contours, reflecting inter-observer variability.
- **Clinical Applicability:** Demonstrated the utility of ASTRA in identifying critical regions in OARs susceptible to dose changes, facilitating informed decision-making in treatment planning.

---

## Methodology

- **Data:** Utilized a dataset of 100 glioblastoma patients, including CT scans, OAR segmentations, and corresponding dose distributions.
- **Perturbation Techniques:** Applied atomic surface transformations to OAR contours to simulate local segmentation variability.
- **Model Architecture:** Employed convolutional neural networks to predict dose sensitivity maps based on perturbed segmentations.
- **Evaluation Metrics:** Assessed the model's performance using metrics such as Mean Absolute Error (MAE) and Dose-Volume Histogram (DVH) differences.

---

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- MONAI
- NumPy
- SciPy
- Matplotlib

### Installation

```bash
git clone https://github.com/amithjkamath/astra.git
cd astra
pip install -r requirements.txt
```

If this is useful in your research, please consider citing:

    @inproceedings{kamath2023astra,
    title={ASTRA: Atomic Surface Transformations for Radiotherapy quality Assurance},
    author={Kamath, Amith and Poel, Robert and Willmann, Jonas and Ermis, Ekin and Andratschke, Nicolaus and Reyes, Mauricio},
    booktitle={45th IEEE Engineering in Medicine and Biology Conference (EMBC)},
    year={2023}
    }

## Credits
Major props to the code and organization in https://github.com/LSL000UD/RTDosePrediction, which is what this model is based on (looks like this repo is not maintained/available anymore!)
