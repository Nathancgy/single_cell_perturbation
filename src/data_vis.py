import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_json_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def plot_mrrmse(data, title, ax):
    ax.plot(data['train_mrrmse'], label='Train MRRMSE')
    ax.plot(data['val_mrrmse'], label='Val MRRMSE')
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MRRMSE')
    ax.legend()

def visualize_results(folder_name):
    base_path = f'./results/{folder_name}'
    chemberta_types = ['MTR', 'MLM']
    input_types = ['initial', 'light', 'heavy']
    
    for chemberta in chemberta_types:
        for fold in range(5):
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(f'{folder_name} - {chemberta} - Fold {fold}', fontsize=16)
            
            for i, input_type in enumerate(input_types):
                file_path = f'{base_path}/{chemberta}/Conv/{input_type}_fold{fold}.json'
                if os.path.exists(file_path):
                    data = load_json_data(file_path)
                    plot_mrrmse(data, f'{input_type.capitalize()}', axes[i])
                else:
                    axes[i].text(0.5, 0.5, 'No data available', ha='center', va='center')
                    axes[i].set_title(f'{input_type.capitalize()}')
            
            plt.tight_layout()
            plt.savefig(f'{folder_name}_{chemberta}_fold{fold}.png')
            plt.close()

# Usage
visualize_results('QSAR_no_data_aug')