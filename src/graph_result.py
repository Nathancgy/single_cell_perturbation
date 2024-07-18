import os
import json
import numpy as np
import matplotlib.pyplot as plt

base_path = '../results/'
models = ['Conv', 'GRU', 'LSTM']
categories = ['initial', 'light', 'heavy']
folds = [0, 1, 2, 3, 4]

def average_lists(lists):
    return list(np.mean(np.array(lists), axis=0))

if __name__ == "__main__":

    for chemberta in ['MTR', 'MLM']:
        data_dict = {model: {category: {fold: {'train': [], 'val': []} for fold in folds} for category in categories} for model in models}
        folder_path = os.path.join(base_path, chemberta)
        
        for model in models:
            model_path = os.path.join(folder_path, model)
            for filename in os.listdir(model_path):
                if filename.endswith('.json'):
                    file_path = os.path.join(model_path, filename)
                    parts = filename.split('_')
                    category = parts[0]
                    fold = int(parts[1][4])
                    with open(file_path, 'r') as file:
                        data = json.load(file)
                    if 'train_mrrmse' in data and 'val_mrrmse' in data:
                        data_dict[model][category][fold]['train'].append(data['train_mrrmse'])
                        data_dict[model][category][fold]['val'].append(data['val_mrrmse'])
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, fold in enumerate(folds):
            all_train = []
            all_val = []
            for model in models:
                for category in categories:
                    if data_dict[model][category][fold]['train'] and data_dict[model][category][fold]['val']:
                        all_train.extend(data_dict[model][category][fold]['train'])
                        all_val.extend(data_dict[model][category][fold]['val'])

            avg_train = average_lists(all_train)
            avg_val = average_lists(all_val)

            axes[i].plot(avg_train, label='Average train', color='blue')
            axes[i].plot(avg_val, label='Average validation', linestyle='--', color='orange')
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel('MRRMSE')
            axes[i].set_title(f'Fold {fold + 1} ChemBERTa-77M-{chemberta}')
            axes[i].legend()
        
        # Remove the empty subplot if necessary
        if len(folds) < len(axes):
            fig.delaxes(axes[-1])
        
        plt.tight_layout()
        plt.savefig(f'../results/{chemberta}_folds.png')
        plt.show()
