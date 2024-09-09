import numpy as np
import os

# Assuming this script is in the root directory of your project
settings = {
    "TRAIN_DATA_AUG_DIR": "../../prepared_data/",
    "PROCESSED_DATA_DIR": "../../prepared_data/"
}

def scale_handcrafted_features():
    # Load the handcrafted features
    input_path = os.path.join(settings["TRAIN_DATA_AUG_DIR"], "handcrafted_features.npy")
    handcrafted_features = np.load(input_path, allow_pickle=True)

    print(f"Original handcrafted features shape: {handcrafted_features.shape}")
    print(f"Number of NaN values: {np.isnan(handcrafted_features).sum()}")
    print(f"Original range: [{np.nanmin(handcrafted_features)}, {np.nanmax(handcrafted_features)}]")
    print(f"Number of non-zero elements: {np.count_nonzero(handcrafted_features)}")
    print(f"Number of unique values: {len(np.unique(handcrafted_features))}")

    # Replace NaN values with 0 before scaling
    handcrafted_features = np.nan_to_num(handcrafted_features, nan=0.0)

    # Shift all values to be positive
    min_value = handcrafted_features.min()
    shifted_features = handcrafted_features - min_value + 1  # Add 1 to avoid log(0)

    # Apply log transformation
    log_features = np.log1p(shifted_features)  # log1p is log(1+x), which handles 0 values

    # Min-max scaling to [0, 1] range
    min_val = log_features.min()
    max_val = log_features.max()
    scaled_features = (log_features - min_val) / (max_val - min_val)

    print(f"Scaled handcrafted features shape: {scaled_features.shape}")
    print(f"Number of NaN values after scaling: {np.isnan(scaled_features).sum()}")
    print(f"New range: [{np.nanmin(scaled_features)}, {np.nanmax(scaled_features)}]")
    print(f"Number of non-zero elements after scaling: {np.count_nonzero(scaled_features)}")
    print(f"Number of unique values after scaling: {len(np.unique(scaled_features))}")

    # Create the output directory if it doesn't exist
    os.makedirs(settings["PROCESSED_DATA_DIR"], exist_ok=True)

    # Save the scaled features
    output_path = os.path.join(settings["PROCESSED_DATA_DIR"], "handcrafted_features_scaled.npy")
    np.save(output_path, scaled_features)

    print(f"Scaled handcrafted features saved to {output_path}")

if __name__ == "__main__":
    scale_handcrafted_features()