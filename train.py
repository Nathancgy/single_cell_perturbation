import warnings
from argparse import Namespace
import pandas as pd
import numpy as np
import json
import torch
from src.helper_functions import seed_everything, combine_features, train_validate

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

if __name__ == "__main__":

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")


    with open("./config/SETTINGS.json") as file:
        settings = json.load(file)
    with open("./config/train_config.json") as file:
        train_config = json.load(file)
        
    print("\nRead data and build features...")

    de_train = pd.read_parquet(settings["TRAIN_RAW_DATA_PATH"])
    xlist  = ['cell_type','sm_name']
    ylist = ['cell_type','sm_name','sm_lincs_id','SMILES','control']
    one_hot_train = pd.DataFrame(np.load(f'{settings["TRAIN_DATA_AUG_DIR"]}one_hot_train.npy'))
    y = de_train.drop(columns=ylist)

    mean_cell_type = pd.read_csv(f'{settings["TRAIN_DATA_AUG_DIR"]}mean_cell_type.csv')
    std_cell_type = pd.read_csv(f'{settings["TRAIN_DATA_AUG_DIR"]}std_cell_type.csv')
    mean_sm_name = pd.read_csv(f'{settings["TRAIN_DATA_AUG_DIR"]}mean_sm_name.csv')
    std_sm_name = pd.read_csv(f'{settings["TRAIN_DATA_AUG_DIR"]}std_sm_name.csv')
    quantiles_df = pd.read_csv(f'{settings["TRAIN_DATA_AUG_DIR"]}quantiles_cell_type.csv')

    # Print sizes of loaded features
    print(f"mean_cell_type size: {mean_cell_type.shape}")
    print(f"std_cell_type size: {std_cell_type.shape}")
    print(f"mean_sm_name size: {mean_sm_name.shape}")
    print(f"std_sm_name size: {std_sm_name.shape}")
    print(f"quantiles_df size: {quantiles_df.shape}")
    print(f"one_hot_train size: {one_hot_train.shape}")
    print(f"y size: {y.shape}")

    # Load handcrafted features
    handcrafted_features = np.load(f'{settings["TRAIN_DATA_AUG_DIR"]}handcrafted_features.npy')

    # Print size of handcrafted features
    print(f"handcrafted_features size: {handcrafted_features.shape}")

    for chemberta in ['MTR', 'MLM']:
        train_chem_feat = np.load(f'{settings["TRAIN_DATA_AUG_DIR"]}chemberta_train_{chemberta}.npy')
        train_chem_feat_mean = np.load(f'{settings["TRAIN_DATA_AUG_DIR"]}chemberta_train_mean_{chemberta}.npy')

        # Print sizes of ChemBERTa features
        print(f"train_chem_feat size ({chemberta}): {train_chem_feat.shape}")
        print(f"train_chem_feat_mean size ({chemberta}): {train_chem_feat_mean.shape}")

        X_vec = combine_features([mean_cell_type, std_cell_type, mean_sm_name, std_sm_name],\
                [train_chem_feat, train_chem_feat_mean], de_train, one_hot_train, handcrafted_features=handcrafted_features)
        X_vec_light = combine_features([mean_cell_type, mean_sm_name],\
                    [train_chem_feat, train_chem_feat_mean], de_train, one_hot_train, handcrafted_features=handcrafted_features)
        X_vec_heavy = combine_features([quantiles_df, mean_cell_type, mean_sm_name],\
                    [train_chem_feat, train_chem_feat_mean], de_train, one_hot_train, quantiles_df, handcrafted_features=handcrafted_features)
        
        # Print the dimensions of the combined features before training
        print(f"\nDimensions before training starts for {chemberta}:")
        print(f"X_vec dimensions: {X_vec.shape}")
        print(f"X_vec_light dimensions: {X_vec_light.shape}")
        print(f"X_vec_heavy dimensions: {X_vec_heavy.shape}")
        
        cell_types_sm_names = de_train[['cell_type', 'sm_name']]
        print("\nTraining starting...")
        trained_models = train_validate(X_vec, X_vec_light, X_vec_heavy, y, cell_types_sm_names, train_config, chemberta, handcrafted_features)
        print("\nDone " + chemberta)

        # You might want to save or further process the trained_models here