import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial import ConvexHull

# Load your submission data
submission1 = pd.read_csv('../submission1.csv').drop(columns=['id'])  # Assuming 'id' is the column to be excluded
submission2 = pd.read_csv('../submission2.csv').drop(columns=['id'])
submission3 = pd.read_csv('../submission3.csv').drop(columns=['id'])
submission4 = pd.read_csv('../submission4.csv').drop(columns=['id'])

# Combine data into a single DataFrame with a label for each submission
data = pd.concat([submission1, submission2, submission3, submission4], ignore_index=True)
labels = ['CNN'] * len(submission1) + ['CNN + HCCSFRM'] * len(submission2) + \
         ['CNN + cleaned HCCSFRM'] * len(submission3) + ['CNN + cleaned HCCSFRM + no data aug'] * len(submission4)

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Perform PCA to reduce to 3 components
pca = PCA(n_components=3)
pca_result = pca.fit_transform(data_scaled)

# Convert PCA results into a DataFrame
pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2', 'PC3'])
pca_df['label'] = labels

# Function to plot convex hulls around clusters
def plot_convex_hull(points, ax, color):
    hull = ConvexHull(points)
    for simplex in hull.simplices:
        ax.plot(points[simplex, 0], points[simplex, 1], color)

# Plotting the first two principal components
plt.figure(figsize=(10, 8))
colors = {'CNN': 'red', 'CNN + HCCSFRM': 'blue', 'CNN + cleaned HCCSFRM': 'green', 'CNN + cleaned HCCSFRM + no data aug': 'purple'}

for label in pca_df['label'].unique():
    subset = pca_df[pca_df['label'] == label]
    plt.scatter(subset['PC1'], subset['PC2'], color=colors[label], label=label, alpha=0.5, s=15)
    
    # Plot convex hulls
    points = subset[['PC1', 'PC2']].values
    plot_convex_hull(points, plt, colors[label])

plt.xlabel('Principal Component 1', fontsize=18)
plt.ylabel('Principal Component 2', fontsize=18)
plt.title('PCA of models (PC1 vs PC2)', fontsize=22)
plt.legend(fontsize=14)
plt.grid(True)
plt.show()
