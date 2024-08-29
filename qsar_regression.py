import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score

def calculate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    morgan_gen = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    return np.array(morgan_gen)

def qsar_regression_pipeline(input_file, output_file, n_components=50):
    print("Loading data...")
    data = pd.read_excel(input_file)
    
    print("Calculating descriptors...")
    X = np.array([calculate_descriptors(smiles) for smiles in data['SMILES']])
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Training Random Forest models...")
    results = {}
    for i, col in enumerate(data.columns[1:], 1):  # Assuming SMILES is the first column
        print(f"Processing activity {i}/{len(data.columns)-1}: {col}")
        y = data[col].values
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
    
    print("Performing PCA...")
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create output DataFrame
    output_data = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components)])
    output_data['SMILES'] = data['SMILES']
    
    # Add all original biological activity columns
    output_data = pd.concat([output_data, data.drop('SMILES', axis=1)], axis=1)
    
    # Save results
    output_data.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    # Print information about PCA
    print(f"\nPCA Information:")
    print(f"Number of original molecular descriptors: {X.shape[1]}")
    print(f"Number of PCA components: {n_components}")
    print(f"Variance explained by PCA components: {pca.explained_variance_ratio_.sum():.2%}")
    print(f"\nNote: PCA was applied to molecular descriptors, not biological activity columns.")
    print(f"All {len(data.columns) - 1} original biological activity columns are preserved in the output file.")

if __name__ == "__main__":
    input_file = "./data/qsar_descriptors.xlsx"
    output_file = "./data/qsar_pca_features.csv"
    n_components = 50  # Number of PCA components to keep
    
    qsar_regression_pipeline(input_file, output_file, n_components)