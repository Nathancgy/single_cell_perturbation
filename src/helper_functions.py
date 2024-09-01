import os
import json
import time
from argparse import Namespace
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from transformers import AutoModelForMaskedLM, AutoTokenizer
import random
from sklearn.model_selection import KFold as KF
from src.models import Conv
from src.helper_classes import Dataset

with open("./config/SETTINGS.json") as file:
    settings = json.load(file)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything():
    """
    Set random seeds for reproducibility across different libraries and components.
    
    This function sets a fixed seed for random number generators in Python's random module,
    NumPy, PyTorch, and CUDA (if available). It also sets some environment variables for
    further consistency.
    """
    seed = 42
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print('-----Seed Set!-----')
    
    
#### Data preprocessing utilities
def one_hot_encode(data_train, data_test, out_dir):
    """
    Perform one-hot encoding on training and test data and save the results.

    Args:
        data_train (array-like): Training data to be encoded.
        data_test (array-like): Test data to be encoded.
        out_dir (str): Directory to save the encoded data.

    This function fits a OneHotEncoder on the training data, transforms both
    training and test data, and saves the results as numpy arrays.
    """
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    encoder = OneHotEncoder()
    encoder.fit(data_train)
    train_features = encoder.transform(data_train)
    test_features = encoder.transform(data_test)
    np.save(f"{out_dir}/one_hot_train.npy", train_features.toarray().astype(float))
    np.save(f"{out_dir}/one_hot_test.npy", test_features.toarray().astype(float))        
        
def build_ChemBERTa_features(smiles_list, model):
    """
    Generate ChemBERTa features for a list of SMILES strings.

    Args:
        smiles_list (list): List of SMILES strings to generate features for.
        model (str): Name of the ChemBERTa model to use.

    Returns:
        tuple: Two numpy arrays containing the embeddings and mean embeddings.

    This function loads a pre-trained ChemBERTa model, tokenizes the SMILES strings,
    and generates embeddings for each molecule.
    """
    model_path = f"./models/ChemBERTa-77M-{model}"
    chemberta = AutoModelForMaskedLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chemberta = chemberta.to(device)
    chemberta.eval()
    
    embeddings = torch.zeros(len(smiles_list), 600)
    embeddings_mean = torch.zeros(len(smiles_list), 600)
    
    with torch.no_grad():
        for i, smiles in enumerate(tqdm(smiles_list)):
            encoded_input = tokenizer(smiles, return_tensors="pt", padding=False, truncation=True)
            encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
            model_output = chemberta(**encoded_input)
            embedding = model_output[0][::,0,::].cpu()
            embeddings[i] = embedding
            embedding = torch.mean(model_output[0], 1).cpu()
            embeddings_mean[i] = embedding

    return embeddings.numpy(), embeddings_mean.numpy()


def save_ChemBERTa_features(smiles_list, out_dir, model, on_train_data=False):
    """
    Generate and save ChemBERTa features for a list of SMILES strings.

    Args:
        smiles_list (list): List of SMILES strings to generate features for.
        out_dir (str): Directory to save the generated features.
        model (str): Name of the ChemBERTa model to use.
        on_train_data (bool): Flag to indicate if the features are for training data.

    This function generates ChemBERTa features and saves them as numpy arrays.
    """
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    emb, emb_mean = build_ChemBERTa_features(smiles_list, model)
    if on_train_data:
        np.save(f"{out_dir}/chemberta_train_" + model + ".npy", emb)
        np.save(f"{out_dir}/chemberta_train_mean_" + model + ".npy", emb_mean)
    else:
        np.save(f"{out_dir}/chemberta_test_" + model + ".npy", emb)
        np.save(f"{out_dir}/chemberta_test_mean_" + model + ".npy", emb_mean)                
                
def combine_features(data_aug_dfs, chem_feats, main_df, one_hot_dfs=None, quantiles_df=None, handcrafted_features=None):
    """
    Combine various features into a single long vector for each input pair.

    Args:
        data_aug_dfs (list): List of DataFrames containing augmented data.
        chem_feats (list): List of chemical features.
        main_df (DataFrame): Main DataFrame containing cell types and compound names.
        one_hot_dfs (DataFrame, optional): One-hot encoded features.
        quantiles_df (DataFrame, optional): DataFrame containing quantile information.
        handcrafted_features (numpy.ndarray, optional): Handcrafted features.

    Returns:
        numpy.ndarray: Combined features for each input pair.

    This function concatenates various features including one-hot encodings,
    augmented data, and chemical features into a single vector for each input pair.
    """
    new_vecs = []
    chem_feat_dim = 600
    if len(data_aug_dfs) > 0:
        add_len = sum(aug_df.shape[1]-1 for aug_df in data_aug_dfs)+chem_feat_dim*len(chem_feats)+one_hot_dfs.shape[1] if\
        one_hot_dfs is not None else sum(aug_df.shape[1]-1 for aug_df in data_aug_dfs)+chem_feat_dim*len(chem_feats)
    else:
        add_len = chem_feat_dim*len(chem_feats)+one_hot_dfs.shape[1] if\
        one_hot_dfs is not None else chem_feat_dim*len(chem_feats)
    if quantiles_df is not None:
        add_len += (quantiles_df.shape[1]-1)//3

    if handcrafted_features is not None:
        add_len += handcrafted_features.shape[1]

    for i in range(len(main_df)):
        if one_hot_dfs is not None:
            vec_ = (one_hot_dfs.iloc[i,:].values).copy()
        else:
            vec_ = np.array([])

        for df in data_aug_dfs:
            if 'cell_type' in df.columns:
                values = df[df['cell_type']==main_df.iloc[i]['cell_type']].values.squeeze()[1:].astype(float)
                vec_ = np.concatenate([vec_, values])
            else:
                assert 'sm_name' in df.columns
                values = df[df['sm_name']==main_df.iloc[i]['sm_name']].values.squeeze()[1:].astype(float)
                vec_ = np.concatenate([vec_, values])

        for chem_feat in chem_feats:
            vec_ = np.concatenate([vec_, chem_feat[i]])

        if handcrafted_features is not None:
            vec_ = np.concatenate([vec_, handcrafted_features[i]])

        final_vec = np.concatenate([vec_,np.zeros(add_len-vec_.shape[0],)])
        new_vecs.append(final_vec)

    return np.stack(new_vecs, axis=0).astype(float).reshape(len(main_df), 1, add_len)

def augment_data(x_, y_):
    """
    Augment the input data by setting random elements to zero.

    Args:
        x_ (numpy.ndarray): Input features.
        y_ (numpy.ndarray): Corresponding labels.

    Returns:
        tuple: Augmented features and corresponding labels.

    This function creates a copy of the input features and sets a random 30%
    of the elements to zero for each sample, effectively augmenting the dataset.
    """
    copy_x = x_.copy()
    new_x = []
    new_y = y_.copy()
    dim = x_.shape[2]
    k = int(0.3*dim)

    for i in range(x_.shape[0]):
        idx = random.sample(range(dim), k=k)
        copy_x[i,:,idx] = 0
        new_x.append(copy_x[i])

    return np.stack(new_x, axis=0), new_y

#### Metrics
def mrrmse_np(y_pred, y_true):
    """
    Calculate the Mean Root Mean Squared Error (MRRMSE) between predictions and true values.

    Args:
        y_pred (numpy.ndarray): Predicted values.
        y_true (numpy.ndarray): True values.

    Returns:
        float: The calculated MRRMSE.

    This function computes the MRRMSE, which is the mean of the root mean squared errors
    calculated for each sample across all targets.
    """
    return np.sqrt(np.square(y_true - y_pred).mean(axis=1)).mean()


#### Training utilities
def train_step(dataloader, model, opt, clip_norm):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    train_losses = []
    train_mrrmse = []

    for batch in dataloader:
        x = batch[0].to(device)
        target = batch[1].to(device)
        handcrafted_features = batch[2].to(device) if len(batch) > 2 else None

        loss = model(x, handcrafted_features, target)
        train_losses.append(loss.item())
        pred = model(x, handcrafted_features).detach().cpu().numpy()
        train_mrrmse.append(mrrmse_np(pred, target.cpu().numpy()))
        opt.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), clip_norm)
        opt.step()

    return np.mean(train_losses), np.mean(train_mrrmse)

def validation_step(dataloader, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    val_losses = []
    val_mrrmse = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch[0].to(device)
            target = batch[1].to(device)
            handcrafted_features = batch[2].to(device) if len(batch) > 2 else None

            loss = model(x, handcrafted_features, target)
            pred = model(x, handcrafted_features).cpu().numpy()
            val_mrrmse.append(mrrmse_np(pred, target.cpu().numpy()))
            val_losses.append(loss.item())

    return np.mean(val_losses), np.mean(val_mrrmse)


def train_function(model, x_train, y_train, x_val, y_val, info_data, config, clip_norm=1.0, handcrafted_features_train=None, handcrafted_features_val=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt = torch.optim.Adam(model.parameters(), lr=config["LEARNING_RATES"][0])
    model.to(device)
    results = {'train_loss': [], 'val_loss': [], 'train_mrrmse': [], 'val_mrrmse': [],
               'train_cell_type': info_data['train_cell_type'], 'val_cell_type': info_data['val_cell_type'], 
               'train_sm_name': info_data['train_sm_name'], 'val_sm_name': info_data['val_sm_name'], 'runtime': None}
   
    x_train_aug, y_train_aug = augment_data(x_train, y_train)
    x_train_aug = np.concatenate([x_train, x_train_aug], axis=0)
    y_train_aug = np.concatenate([y_train, y_train_aug], axis=0)
    
    # Convert input data to float32
    x_train_aug = x_train_aug.astype(np.float32)
    y_train_aug = y_train_aug.astype(np.float32)
    x_val = x_val.astype(np.float32)
    y_val = y_val.astype(np.float32)
    
    if handcrafted_features_train is not None:
        handcrafted_features_train_aug = np.concatenate([handcrafted_features_train, handcrafted_features_train], axis=0)
        handcrafted_features_train_aug = handcrafted_features_train_aug.astype(np.float32)
        handcrafted_features_val = handcrafted_features_val.astype(np.float32)
        train_dataset = Dataset(x_train_aug, y_train_aug, handcrafted_features_train_aug)
        val_dataset = Dataset(x_val, y_val, handcrafted_features_val)
    else:
        train_dataset = Dataset(x_train_aug, y_train_aug)
        val_dataset = Dataset(x_val, y_val)
    
    train_dataloader = DataLoader(train_dataset, num_workers=0, batch_size=128, shuffle=True)
    val_dataloader = DataLoader(val_dataset, num_workers=0, batch_size=256, shuffle=False)
    
    best_loss = np.inf
    best_weights = None
    t0 = time.time()

    for e in range(config["EPOCHS"]):
        try:
            loss, mrrmse = train_step(train_dataloader, model, opt, clip_norm)
            val_loss, val_mrrmse = validation_step(val_dataloader, model)
            results['train_loss'].append(float(loss))
            results['val_loss'].append(float(val_loss))
            results['train_mrrmse'].append(float(mrrmse))
            results['val_mrrmse'].append(float(val_mrrmse))
            if val_mrrmse < best_loss:
                best_loss = val_mrrmse
                best_weights = model.state_dict()
                print('BEST ----> ')
            print(f"{model.name} Epoch {e}, train_loss {round(loss,3)}, val_loss {round(val_loss, 3)}, val_mrrmse {val_mrrmse}")
        except Exception as e:
            print(f"Error during training: {str(e)}")
            break

    t1 = time.time()
    results['runtime'] = float(t1-t0)
    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model, results


def cross_validate_models(X, y, kf_cv, cell_types_sm_names, chemberta, config=None, scheme='initial', clip_norm=1.0, handcrafted_features=None):
    """
    Perform cross-validation for model training.

    Args:
        X (numpy.ndarray): Input features.
        y (pandas.DataFrame): Target values.
        kf_cv (sklearn.model_selection.KFold): K-Fold cross-validator.
        cell_types_sm_names (pandas.DataFrame): DataFrame containing cell types and small molecule names.
        chemberta (str): Type of ChemBERTa model used ('MLM' or 'MTR').
        config (dict, optional): Configuration parameters for training. Defaults to None.
        scheme (str, optional): Model scheme ('initial', 'light', or 'heavy'). Defaults to 'initial'.
        clip_norm (float, optional): Gradient clipping norm. Defaults to 1.0.
        handcrafted_features (numpy.ndarray, optional): Handcrafted features. Defaults to None.

    Returns:
        list: List of trained models for each fold.
    """
    trained_models = []

    for i,(train_idx,val_idx) in enumerate(kf_cv.split(X)):
        print(f"\nSplit {i+1}/{kf_cv.n_splits}...")
        x_train, x_val = X[train_idx], X[val_idx]
        y_train, y_val = y.values[train_idx], y.values[val_idx]
        
        if handcrafted_features is not None:
            handcrafted_features_train = handcrafted_features[train_idx]
            handcrafted_features_val = handcrafted_features[val_idx]
        else:
            handcrafted_features_train = None
            handcrafted_features_val = None
        
        info_data = {'train_cell_type': cell_types_sm_names.iloc[train_idx]['cell_type'].tolist(),
                    'val_cell_type': cell_types_sm_names.iloc[val_idx]['cell_type'].tolist(),
                    'train_sm_name': cell_types_sm_names.iloc[train_idx]['sm_name'].tolist(),
                    'val_sm_name': cell_types_sm_names.iloc[val_idx]['sm_name'].tolist()}
        
        model = Conv(scheme)
        model, results = train_function(model, x_train, y_train, x_val, y_val, info_data, config=config, clip_norm=clip_norm, 
                                        handcrafted_features_train=handcrafted_features_train, 
                                        handcrafted_features_val=handcrafted_features_val)
        trained_models.append(model)

        if not os.path.exists(f'{settings["MODEL_DIR"]}{chemberta}/{model.name}/'):
            os.mkdir(f'{settings["MODEL_DIR"]}{chemberta}/{model.name}/')
        torch.save(model.state_dict(), f'{settings["MODEL_DIR"]}{chemberta}/{model.name}/{scheme}_fold{i}.pt')
        if not os.path.exists(f'{settings["LOGS_DIR"]}{chemberta}/{model.name}/'):
            os.mkdir(f'{settings["LOGS_DIR"]}{chemberta}/{model.name}/')
        with open(f'{settings["LOGS_DIR"]}{chemberta}/{model.name}/{scheme}_fold{i}.json', 'w') as file:
            json.dump(results, file)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return trained_models

def train_validate(X_vec, X_vec_light, X_vec_heavy, y, cell_types_sm_names, config, chemberta, handcrafted_features):
    """
    Train and validate models using different input feature sets.

    Args:
        X_vec (numpy.ndarray): Input features for the 'initial' scheme.
        X_vec_light (numpy.ndarray): Input features for the 'light' scheme.
        X_vec_heavy (numpy.ndarray): Input features for the 'heavy' scheme.
        y (pandas.DataFrame): Target values.
        cell_types_sm_names (pandas.DataFrame): DataFrame containing cell types and small molecule names.
        config (dict): Configuration parameters for training.
        chemberta (str): Type of ChemBERTa model used ('MLM' or 'MTR').
        handcrafted_features (numpy.ndarray): Handcrafted features.

    Returns:
        dict: Dictionary containing trained models for each scheme.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    kf_cv = KF(n_splits=config["KF_N_SPLITS"], shuffle=True, random_state=42)
    trained_models = {'initial': [], 'light': [], 'heavy': []}

    if not os.path.exists(f'{settings["MODEL_DIR"]}'):
        os.mkdir(f'{settings["MODEL_DIR"]}')
    if not os.path.exists(f'{settings["MODEL_DIR"]}{chemberta}/'):
        os.mkdir(f'{settings["MODEL_DIR"]}{chemberta}/')
    if not os.path.exists(f'{settings["LOGS_DIR"]}'):
        os.mkdir(f'{settings["LOGS_DIR"]}')
    if not os.path.exists(f'{settings["LOGS_DIR"]}{chemberta}/'):   
        os.mkdir(f'{settings["LOGS_DIR"]}{chemberta}/')

    for scheme, clip_norm, input_features in zip(['initial', 'light', 'heavy'], config["CLIP_VALUES"], [X_vec, X_vec_light, X_vec_heavy]):
        seed_everything()
        models = cross_validate_models(input_features, y, kf_cv, cell_types_sm_names, chemberta, config=config, scheme=scheme, clip_norm=clip_norm, handcrafted_features=handcrafted_features)
        trained_models[scheme].extend(models)

    return trained_models

def inference_pytorch(model, dataloader):
    """
    Perform inference using a PyTorch model.

    Args:
        model (torch.nn.Module): Trained PyTorch model.
        dataloader (torch.utils.data.DataLoader): DataLoader containing the test data.

    Returns:
        numpy.ndarray: Array of model predictions.
    """
    device = next(model.parameters()).device
    model.eval()
    preds = []
 
    with torch.no_grad():
        for batch in dataloader:
            x = batch[0].to(device)
            handcrafted_features = batch[1].to(device) if len(batch) > 1 else None
            pred = model(x, handcrafted_features).cpu().numpy()
            preds.append(pred)
 
    return np.concatenate(preds, axis=0)

def average_prediction(dataloader, trained_models):
    """
    Compute the average prediction from multiple trained models.

    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader containing the test data.
        trained_models (list): List of trained PyTorch models.

    Returns:
        numpy.ndarray: Array of averaged predictions.
    """
    all_preds = []

    for model in trained_models:
        current_pred = inference_pytorch(model, dataloader)
        all_preds.append(current_pred)

    return np.stack(all_preds, axis=1).mean(axis=1)

def weighted_average_prediction(dataloader, trained_models, model_wise=[0.25, 0.35, 0.40], fold_wise=None):
    """
    Compute the weighted average prediction from multiple trained models.

    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader containing the test data.
        trained_models (list): List of trained PyTorch models.
        model_wise (list, optional): Weights for different model types. Defaults to [0.25, 0.35, 0.40].
        fold_wise (list, optional): Weights for different folds. Defaults to None.

    Returns:
        numpy.ndarray: Array of weighted average predictions.
    """
    all_preds = []

    for i, model in enumerate(trained_models):
        current_pred = inference_pytorch(model, dataloader)
        current_pred = model_wise[i%3] * current_pred
        if fold_wise:
            current_pred = fold_wise[i//3] * current_pred
        all_preds.append(current_pred)

    return np.stack(all_preds, axis=1).sum(axis=1)

def load_trained_models(chemberta, path=settings["MODEL_DIR"], kf_n_splits=5):
    """
    Load trained models from saved weights.

    Args:
        chemberta (str): Type of ChemBERTa model used ('MLM' or 'MTR').
        path (str, optional): Path to the directory containing model weights. Defaults to settings["MODEL_DIR"].
        kf_n_splits (int, optional): Number of splits in K-Fold cross-validation. Defaults to 5.

    Returns:
        dict: Dictionary containing loaded models for each scheme.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_models = {'initial': [], 'light': [], 'heavy': []}
 
    for scheme in ['initial', 'light', 'heavy']:
        for fold in range(kf_n_splits):
            for Model in [Conv]:
                model = Model(scheme)
                model_dir = f'{path}{chemberta}/{model.name}/'
                weights_filename = f'{scheme}_fold{fold}.pt'
                weights_path = os.path.join(model_dir, weights_filename)
                if os.path.exists(weights_path):
                    state_dict = torch.load(weights_path, map_location=device)
                    
                    # Adjust the size of the first linear layer's weights
                    old_linear_weight = state_dict['linear.0.weight']
                    new_linear_weight = torch.zeros(1024, model.adjusted_input_size, device=device)
                    
                    # Copy the old weights to the new tensor
                    min_width = min(old_linear_weight.size(1), new_linear_weight.size(1))
                    new_linear_weight[:, :min_width] = old_linear_weight[:, :min_width]
                    
                    state_dict['linear.0.weight'] = new_linear_weight
                    
                    # Adjust the size of the first linear layer's bias if necessary
                    if 'linear.0.bias' in state_dict and state_dict['linear.0.bias'].size(0) != 1024:
                        old_linear_bias = state_dict['linear.0.bias']
                        new_linear_bias = torch.zeros(1024, device=device)
                        min_size = min(old_linear_bias.size(0), new_linear_bias.size(0))
                        new_linear_bias[:min_size] = old_linear_bias[:min_size]
                        state_dict['linear.0.bias'] = new_linear_bias
                    
                    # Load the adjusted state dict
                    model.load_state_dict(state_dict, strict=False)
                    model.to(device)
                    trained_models[scheme].append(model)
 
    return trained_models
