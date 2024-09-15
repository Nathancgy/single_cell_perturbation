# Single Cell Perturbation Analysis Pipeline

## 1. Data Preparation
Run: `python prepare_data.py`
- Reads training data from TRAIN_RAW_DATA_DIR (specified in SETTINGS.json)
- Performs preprocessing steps:
  - Creates data augmentation features
  - One-hot encodes categorical variables
  - Generates ChemBERTa features for SMILES data
- Saves prepared data in TRAIN_DATA_AUG_DIR (specified in SETTINGS.json)

## QSAR Pipeline for prediction

## 1. QSAR Model Training and Prediction
Run: `python QSAR_pipeline/train_qsar.py`
- Reads SMILES data and calculates Morgan fingerprints
- Trains 362 Random Forest models for QSAR prediction
- Saves trained models in qsar_models directory

Run: `python QSAR_pipeline/predict_qsar.py`
- Loads trained QSAR models
- Predicts handcrafted features for test data
- Saves predictions as handcrafted_features.npy in TRAIN_DATA_AUG_DIR

Run: `python QSAR_pipeline/scale_qsar.py`
- Loads handcrafted features
- Scales features to [0, 1] range using min-max scaling
- Logs transformed features to file

## 2. QSAR Pipeline Training
Run: `python QSAR_pipeline/train.py`
- Similar to CNN pipeline, but uses QSAR-specific architecture
- Trains models for both MTR and MLM ChemBERTa types
- Saves trained models to MODEL_DIR/QSAR/

## 3. QSAR Pipeline Prediction
Run: `python QSAR_pipeline/predict.py`
- Reads test data and prepared features
- Loads trained QSAR models
- Specify the model type (MTR or MLM)
- Makes predictions using ensemble of models
- Saves predictions to SUBMISSION_DIR/QSAR_submission.csv

Note: Ensure all paths in SETTINGS.json are correctly set before running the pipeline.