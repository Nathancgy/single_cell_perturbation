import os
import json
import numpy as np
import matplotlib.pyplot as plt

base_path = 'results/'
categories = ['initial', 'light', 'heavy']
folds = [0, 1, 2, 3, 4]

def average_lists(lists):
    return list(np.mean(np.array(lists), axis=0))

if __name__ == "__main__":

    data_dict = {category: {fold: {'train': [], 'val': []} for fold in folds} for category in categories}
    for chemberta in ['MTR', 'MLM']:
        folder_path = os.path.join(base_path, chemberta)
        for filename in os.listdir(folder_path):
            if filename.endswith('.json'):
                file_path = os.path.join(folder_path, filename)
                parts = filename.split('_')
                category = parts[1]
                fold = int(parts[2][4])
                with open(file_path, 'r') as file:
                    data = json.load(file)
                if 'train_mrrmse' in data and 'val_mrrmse' in data:
                    data_dict[category][fold]['train'].append(data['train_mrrmse'])
                    data_dict[category][fold]['val'].append(data['val_mrrmse'])

        for fold in folds:
            plt.figure()
            all_train = []
            all_val = []
            for category in categories:
                if data_dict[category][fold]['train'] and data_dict[category][fold]['val']:
                    all_train.extend(data_dict[category][fold]['train'])
                    all_val.extend(data_dict[category][fold]['val'])

            avg_train = average_lists(all_train)
            avg_val = average_lists(all_val)

            plt.plot(avg_train, label='Average train', color='blue')
            plt.plot(avg_val, label='Average validation', linestyle='--', color='orange')
            plt.xlabel('Epoch')
            plt.ylabel('MRRMSE')
            plt.title(f'Fold {fold} ChemBERTa-77M-{chemberta}')
            plt.legend()
            plt.show()
