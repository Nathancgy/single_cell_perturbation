import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score

def calculate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return list(Descriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))

def qsar_regression_pipeline(input_file, output_file, n_components=50):
    # Load data
    data = pd.read_excel(input_file)
    
    # Separate SMILES and activity columns
    smiles_col = data['SMILES']
    activity_cols = data.drop('SMILES', axis=1)
    
    # Calculate descriptors
    X = np.array([calculate_descriptors(smiles) for smiles in smiles_col])
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Random Forest model for each activity
    results = {}
    for col in activity_cols.columns:
        y = activity_cols[col].values
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        y_pred = rf_model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[col] = {'MSE': mse, 'R2': r2}
    
    # Print results
    for col, metrics in results.items():
        print(f"{col}: MSE = {metrics['MSE']:.4f}, R2 = {metrics['R2']:.4f}")
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create output DataFrame
    output_data = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components)])
    output_data['SMILES'] = smiles_col
    output_data = pd.concat([output_data, activity_cols], axis=1)
    
    # Save results
    output_data.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    input_file = "./data/qsar_descriptors.xlsx"
    output_file = "./data/qsar_pca_features.csv"
    n_components = 50  # Number of PCA components to keep
    
    qsar_regression_pipeline(input_file, output_file, n_components)