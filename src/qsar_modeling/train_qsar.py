import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import AllChem
import joblib
import os
from tqdm import tqdm

# Function to calculate Morgan fingerprints
def calculate_fingerprints(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    return np.array(fp)

# Load the dataset
data = pd.read_excel('../../data/qsar_descriptors.xlsx')

# Prepare features (X) and targets (y)
X = np.array([calculate_fingerprints(smiles) for smiles in tqdm(data['SMILES'], desc="Calculating fingerprints")])
y = data.drop('SMILES', axis=1).values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a directory to save the models
os.makedirs('qsar_models', exist_ok=True)

# Train 362 random forest models
for i in tqdm(range(362), desc="Training models"):
    # Create and train the model
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train[:, i])
    
    # Save the model
    joblib.dump(rf_model, f'qsar_models/rf_model_{i}.joblib')

# After the model training loop, add:
feature_importance = np.zeros((362, X.shape[1]))
test_predictions = np.zeros((X_test.shape[0], 362))

for i in tqdm(range(362), desc="Extracting feature importance and test predictions"):
    rf_model = joblib.load(f'qsar_models/rf_model_{i}.joblib')
    feature_importance[i] = rf_model.feature_importances_
    test_predictions[:, i] = rf_model.predict(X_test)

np.save('qsar_models/feature_importance.npy', feature_importance)
np.save('qsar_models/test_predictions.npy', test_predictions)
np.save('qsar_models/test_actual.npy', y_test)

# Save fingerprints and SMILES for later use
np.save('qsar_models/fingerprints.npy', X)
pd.DataFrame({'SMILES': data['SMILES']}).to_csv('qsar_models/smiles.csv', index=False)

print("Additional data saved for visualization.")

print("Training completed. Models saved in 'qsar_models' directory.")