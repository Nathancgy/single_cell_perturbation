import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import joblib
from tqdm import tqdm

def calculate_fingerprints(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    return np.array(fp)

# Load the new SMILES data
new_data = pd.read_parquet('../../data/de_train.parquet')

# Calculate fingerprints for the new SMILES
X_new = np.array([calculate_fingerprints(smiles) for smiles in tqdm(new_data['SMILES'], desc="Calculating fingerprints")])

# Create an array to store predictions
predictions = np.zeros((len(X_new), 362))

# Load models and make predictions
for i in tqdm(range(362), desc="Making predictions"):
    # Load the model
    rf_model = joblib.load(f'qsar_models/rf_model_{i}.joblib')
    
    # Make predictions
    predictions[:, i] = rf_model.predict(X_new)

# Create a DataFrame with the predictions
result_df = pd.DataFrame(predictions, columns=[f'feature_{i}' for i in range(362)])

# Add the SMILES column to the result DataFrame
result_df['SMILES'] = new_data['SMILES']

# Save the results
result_df.to_csv('predicted_qsar_features.csv', index=False)

print("Predictions completed. Results saved in 'predicted_qsar_features.csv'.")