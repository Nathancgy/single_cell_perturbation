import torch
import torch.nn as nn
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

class LogCoshLoss(nn.Module):
    """Loss function for regression tasks"""
    def __init__(self):
        super().__init__()

    def forward(self, y_prime_t, y_t):
        """
        Compute the Log-Cosh loss between predicted and true values.

        Args:
            y_prime_t (torch.Tensor): Predicted values
            y_t (torch.Tensor): True values

        Returns:
            torch.Tensor: Computed Log-Cosh loss
        """
        ey_t = (y_t - y_prime_t)/3 # divide by 3 to avoid numerical overflow in cosh
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))
    
    
class Dataset:
    def __init__(self, data_x, data_y=None, handcrafted_features=None):
        self.data_x = torch.tensor(data_x, dtype=torch.float32)
        self.data_y = torch.tensor(data_y, dtype=torch.float32) if data_y is not None else None
        self.handcrafted_features = torch.tensor(handcrafted_features, dtype=torch.float32) if handcrafted_features is not None else None

    def __len__(self):
        return len(self.data_x)
    
    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        Args:
            idx (int): Index of the item to retrieve

        Returns:
            tuple: Returns (x, y, handcrafted_features) if both data_y and handcrafted_features are not None,
                   (x, y) if only data_y is not None,
                   (x, handcrafted_features) if only handcrafted_features is not None,
                   else returns x

        Raises:
            Exception: If there's an error accessing the data at the given index
        """
        try:
            x = self.data_x[idx]
            y = self.data_y[idx] if self.data_y is not None else None
            hf = self.handcrafted_features[idx] if self.handcrafted_features is not None else None

            if y is not None and hf is not None:
                return x, y, hf
            elif y is not None:
                return x, y
            elif hf is not None:
                return x, hf
            else:
                return x
        except Exception as e:
            print(f"Error accessing data at index {idx}: {str(e)}")
            raise
        
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).squeeze(-1))
        max_out = self.fc(self.max_pool(x).squeeze(-1))
        out = avg_out + max_out
        return torch.sigmoid(out).unsqueeze(-1)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=kernel_size//2)
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return torch.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, channels):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(channels)
        self.spatial_att = SpatialAttention()
        
    def forward(self, x):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def load_data(self, file_path):
        """
        Load data from Excel file.

        Args:
            file_path (str): Path to the Excel file

        Returns:
            pandas.DataFrame: Loaded data
        """
        print("Loading data...")
        return pd.read_excel(file_path)

    def calculate_descriptors(self, smiles):
        """
        Calculate Morgan fingerprints for a single SMILES string.

        Args:
            smiles (str): SMILES representation of a molecule

        Returns:
            numpy.ndarray: Morgan fingerprint as a bit vector
        """
        mol = Chem.MolFromSmiles(smiles)
        morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        return np.array(morgan_fp)

    def process_smiles(self, data):
        """
        Calculate descriptors for all SMILES in the dataset.

        Args:
            data (pandas.DataFrame): DataFrame containing SMILES column

        Returns:
            numpy.ndarray: Array of calculated descriptors
        """
        print("Calculating descriptors...")
        X = np.array([self.calculate_descriptors(smiles) for smiles in data['SMILES']])
        return X

    def scale_features(self, X):
        """
        Scale features using StandardScaler.

        Args:
            X (numpy.ndarray): Input features

        Returns:
            numpy.ndarray: Scaled features
        """
        print("Scaling features...")
        return self.scaler.fit_transform(X)

    def prepare_data(self, file_path):
        """
        Load data, calculate descriptors, and scale features.

        Args:
            file_path (str): Path to the Excel file

        Returns:
            tuple: Scaled features, target values, and SMILES strings
        """
        data = self.load_data(file_path)
        X = self.process_smiles(data)
        X_scaled = self.scale_features(X)
        y = data.drop('SMILES', axis=1)
        return X_scaled, y, data['SMILES']
    
class PCAAnalysis:
    def __init__(self, n_components=50):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)

    def perform_pca(self, X):
        """
        Perform PCA on the input features.

        Args:
            X (numpy.ndarray): Input features

        Returns:
            numpy.ndarray: PCA-transformed features
        """
        print("Performing PCA...")
        X_pca = self.pca.fit_transform(X)
        return X_pca

    def get_explained_variance_ratio(self):
        """
        Get the explained variance ratio for each component.

        Returns:
            numpy.ndarray: Explained variance ratio
        """
        return self.pca.explained_variance_ratio_

    def get_cumulative_variance_ratio(self):
        """
        Get the cumulative explained variance ratio.

        Returns:
            numpy.ndarray: Cumulative explained variance ratio
        """
        return np.cumsum(self.pca.explained_variance_ratio_)

    def get_optimal_components(self, threshold=0.95):
        """
        Get the number of components needed to explain a certain amount of variance.

        Args:
            threshold (float): Variance threshold (default: 0.95)

        Returns:
            int: Number of components needed to explain the specified variance
        """
        cumulative_variance = self.get_cumulative_variance_ratio()
        return np.argmax(cumulative_variance >= threshold) + 1

    def transform_data(self, X):
        """
        Transform data using the fitted PCA.

        Args:
            X (numpy.ndarray): Input features

        Returns:
            numpy.ndarray: PCA-transformed features
        """
        return self.pca.transform(X)