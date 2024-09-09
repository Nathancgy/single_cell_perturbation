# Single Cell Perturbation Analysis Project Overview

This project focuses on analyzing single-cell perturbation data using advanced machine learning techniques. It includes data preprocessing, feature engineering, model training, and prediction for Quantitative Structure-Activity Relationship (QSAR) tasks and Convolutional Neural Network (CNN) based predictions.

# Table of Contents
1. Installation
2. Project Structure
3. Usage
4. Visualization
5. Configuration
6. Dependencies
7. Contributing
8. License
9. Contact

## 1. Installation

1. Clone this repository
```
git clone https://github.com/your-username/single-cell-perturbation-analysis.git
```

2. Install dependencies
```
pip install -r requirements.txt
```

## 2. Project Structure

```
single_cell_perturbation/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── config/
│   ├── SETTINGS.json
│   ├── train_config.json
│   └── test_config.json
├── data/
│   ├── de_train.parquet
│   ├── id_map.csv
│   ├── sample_submission.csv
│   ├── qsar_descriptors.xlsx
├── src/
│   ├── prepare_data.py
│   ├── data_vis.py
│   └── qsar_modeling/
│       ├── train_qsar.py
│       ├── predict_qsar.py
│       └── scale_qsar.py
├── CNN_pipeline/
│   ├── helper_functions.py
│   ├── models.py
│   ├── train.py
│   └── predict.py
├── QSAR_pipeline/
│   ├── helper_functions.py
│   ├── helper_classes.py
│   ├── models.py
│   ├── train.py
│   ├── predict.py
├── models/
│   ├── ChemBERTa-77M-MLM/
│   │   └── config.json
│   └── ChemBERTa-77M-MTR/
│       └── config.json
├── trained_models/
│   ├── CNN/
│   └── QSAR/
├── results/
│   ├── QSAR/
│   │   ├── MLM/
│   │   │   └── Conv/
│   │   └── MTR/
│   │       └── Conv/
│   └── QSAR_no_data_aug/
│       └── MLM/
│           └── Conv/
├── submissions/
└── entry_point.md
```

## 3. Usage

The project is divided into two main pipelines: CNN and QSAR. Refer to the entry_point.md file for detailed instructions on how to run the complete analysis.

1. Data Preparation
```
python prepare_data.py
```

2. QSAR Model Training and Prediction:
```
python QSAR_pipeline/train_qsar.py
python QSAR_pipeline/predict_qsar.py
python QSAR_pipeline/scale_qsar.py
```

3. CNN Pipeline:
```
python CNN_pipeline/train.py
python CNN_pipeline/predict.py
```

4. QSAR Pipeline:
```
python QSAR_pipeline/train.py
python QSAR_pipeline/predict.py
```

## 4.  Visualization
The `data_vis.py` file contains functions to visualize the data, including MRRMSE plots for different model configurations.

## 5. Configuration
The `config` directory contains JSON files for training and testing data configurations.

## 6. Dependencies
The `requirements.txt` file lists the dependencies for the project.

## 7. Contributing
We welcome contributions to this project. 

## 8. License
This project is licensed under the Apache License. See the LICENSE file for more details.

## 9. Contact
For any questions or inquiries, feel free to contact me at guangyu.chen40730-biph@basischina.com!
