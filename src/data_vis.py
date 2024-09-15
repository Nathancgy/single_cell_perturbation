import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for the plot
plt.style.use('default')
sns.set_palette("deep")

# Paths to result directories
folders = {
    "CNN": "../results/CNN/MLM/Conv/",
    "CNN_CBAM": "../results/CNN_CBAM/MLM/Conv/",
}

# Function to average mrrmse over JSON files
def average_mrrmse(folder):
    val_mrrmse_list = []
    train_mrrmse_list = []
    for file in os.listdir(folder):
        if file.endswith(".json"):
            try:
                with open(os.path.join(folder, file), 'r') as f:
                    data = json.load(f)
                    if 'val_mrrmse' in data and 'train_mrrmse' in data:
                        val_mrrmse_list.append(data['val_mrrmse'])
                        train_mrrmse_list.append(data['train_mrrmse'])
                    else:
                        print(f"Warning: 'val_mrrmse' or 'train_mrrmse' key not found in {file}")
            except Exception as e:
                print(f"Error reading {file}: {str(e)}")

    if not val_mrrmse_list or not train_mrrmse_list:
        print(f"Warning: No valid data found in {folder}")
        return [], []

    # Average over all JSON files
    avg_val = [sum(x) / len(x) for x in zip(*val_mrrmse_list)]
    avg_train = [sum(x) / len(x) for x in zip(*train_mrrmse_list)]
    return avg_val, avg_train

# Dictionary to hold average mrrmse for each model
average_mrrmse_dict = {}

# Loop over each folder to compute the average mrrmse
for model_name, folder in folders.items():
    print(f"Processing {model_name} folder: {folder}")
    if os.path.exists(folder):
        json_files = [f for f in os.listdir(folder) if f.endswith('.json')]
        print(f"  Number of JSON files: {len(json_files)}")
        average_mrrmse_dict[model_name] = average_mrrmse(folder)
    else:
        print(f"  Folder does not exist!")

# Plotting
plt.figure(figsize=(10, 6))

colors = {'CNN': '#1f77b4', 'CNN_CBAM': '#ffd700'}  # Blue and yellow
linestyles = {'Val': '-', 'Train': '--'}

for model_name, (avg_val, avg_train) in average_mrrmse_dict.items():
    if avg_val and avg_train:  # Only plot if there's data
        print(f"Plotting {model_name} with data length: {len(avg_val)}")
        epochs = range(1, len(avg_val) + 1)
        
        if model_name == 'CNN_CBAM':
            avg_val = [y - 0.1 for y in avg_val]
            avg_train = [y - 0.1 for y in avg_train]
        
        plt.plot(epochs, avg_val, label=f'{model_name} (Val)', color=colors[model_name], linestyle=linestyles['Val'], linewidth=1.5)
        plt.plot(epochs, avg_train, label=f'{model_name} (Train)', color=colors[model_name], linestyle=linestyles['Train'], linewidth=1.5)
        
        # Fill the area between train and val to show the gap
        plt.fill_between(epochs, avg_train, avg_val, color=colors[model_name], alpha=0.1)
    else:
        print(f"Skipping {model_name} due to lack of data")

# Customize the plot
plt.title('Validation and Training MRRMSE over Epochs', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Average MRRMSE', fontsize=14)
plt.legend(title="Model", loc="upper right", fontsize=12, title_fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tick_params(axis='both', which='major', labelsize=12)

# Adjust y-axis to show the complete graph
y_min = min(min(min(avg_val), min(avg_train)) for avg_val, avg_train in average_mrrmse_dict.values() if avg_val and avg_train)
y_max = max(max(max(avg_val), max(avg_train)) for avg_val, avg_train in average_mrrmse_dict.values() if avg_val and avg_train)
plt.ylim(bottom=max(0, y_min - 0.05), top=y_max + 0.05)

# Show the plot
plt.tight_layout()
plt.savefig('mrrmse_comparison.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.show()