from src.helper_classes import DataProcessor, PCAAnalysis
from src.regression_models import RegressionModels
import pandas as pd

def main():
    input_file = "./data/qsar_descriptors.xlsx"
    model_output_dir = "./models"

    processor = DataProcessor()
    X_scaled, y, smiles = processor.prepare_data(input_file)
    
    reg_models = RegressionModels()
    results = reg_models.train_models(X_scaled, y)
    for col, metrics in results.items():
        print(f"{col}: MSE = {metrics['MSE']:.4f}, R2 = {metrics['R2']:.4f}, "
              f"CV_MSE = {metrics['CV_MSE']:.4f} ± {metrics['CV_MSE_std']:.4f}")
    
    # PCA
    pca = PCAAnalysis()
    X_pca = pca.perform_pca(X_scaled)

    print(f"\nPCA Information:")
    print(f"Number of original molecular descriptors: {X_scaled.shape[1]}")
    print(f"Number of PCA components: {pca.n_components}")
    print(f"Variance explained by PCA components: {pca.get_explained_variance_ratio().sum():.2%}")
    
    # Get optimal number of components
    optimal_components = pca.get_optimal_components()
    print(f"Optimal number of components (95% variance explained): {optimal_components}")
    
    reg_models.save_models(model_output_dir)
    
    # Create output DataFrame with PCA components
    output_data = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(pca.n_components)])
    output_data['SMILES'] = smiles
    
    output_data = pd.concat([output_data, y], axis=1)
    output_file = "./data/qsar_pca_features.csv"
    output_data.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
