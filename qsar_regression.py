import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
def load_data(file_path):
    df = pd.read_csv(file_path)
    smiles = df.iloc[:, 0].values
    activities = df.iloc[:, 1:].values
    return smiles, activities

# Generate molecular fingerprints from SMILES
def generate_fingerprints(smiles):
    fingerprints = []
    for smile in smiles:
        mol = Chem.MolFromSmiles(smile)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            fingerprints.append(fp.ToBitString())
    return np.array([list(map(int, fp)) for fp in fingerprints])

# Perform PCA and select top features
def perform_pca(X, n_components=100):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca, pca

# Train QSAR regression model
def train_qsar_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, scaler, mse, r2

if __name__ == "__main__":
    # Load data
    file_path = "your_data.csv"  # Replace with your actual file path
    smiles, activities = load_data(file_path)
    
    # Generate fingerprints
    fingerprints = generate_fingerprints(smiles)
    
    # Perform PCA
    X_pca, pca = perform_pca(fingerprints)
    
    # Train QSAR model for each activity
    for i in range(activities.shape[1]):
        print(f"\nTraining model for activity {i+1}")
        model, scaler, mse, r2 = train_qsar_model(X_pca, activities[:, i])
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"R-squared Score: {r2:.4f}")
        
        # Print top 10 PCA components
        top_components = pca.components_[:10]
        print("\nTop 10 PCA components:")
        for j, component in enumerate(top_components):
            print(f"Component {j+1}: {component[:5]}...")
