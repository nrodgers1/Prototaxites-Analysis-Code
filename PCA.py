import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load the data
file_path = "xxx.xlsx"
data = pd.read_excel(file_path)
print("First 5 rows of data:\n", data.head())

# Extract X (spectral features) and y (one-hot encoded target)
X = data.iloc[:, 1:-2].values
y = data.iloc[:, -2:].values
y_labels = np.argmax(y, axis=1)
print("Labels:", y_labels)

# Perform PCA
n_components = min(X.shape[1], 10)  # Limit components to a max of 10 or less than features
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)

# Save transformed data
# 1. Save PCA loadings (components)
loadings = pd.DataFrame(pca.components_, columns=data.columns[1:-2])
loadings.index = [f"PC{i+1}" for i in range(n_components)]
loadings.to_csv("PCA_Loadings_PCA2.csv", index=True)

# 2. Save PCA scores with class labels
scores = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(n_components)])
scores["Class"] = np.argmax(y, axis=1)
scores.to_csv("PCA_Scores_PCA2.csv", index=False)

print("PCA loadings saved to 'PCA_Loadings.csv'")
print("PCA scores saved to 'PCA_Scores.csv'")

# Explained variance ratio
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# Plot Scree Plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_components + 1), explained_variance, marker='o', linestyle='--', label='Individual Explained Variance')
plt.plot(range(1, n_components + 1), cumulative_variance, marker='s', linestyle='-', label='Cumulative Explained Variance')
plt.xlabel('Principal Component Number')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot of PCA')
plt.legend()
plt.grid()
plt.show()

# Plot data on first two Principal Components
plt.figure(figsize=(10, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_labels, cmap='viridis', alpha=0.7, edgecolors='k')

# Annotate points with row numbers
for i in range(X_pca.shape[0]):  # Loop through each data point
    plt.text(X_pca[i, 0] + 0.0005, X_pca[i, 1] + 0.0005,  # Slight offset
             str(i), fontsize=8, ha='left', va='bottom', color='black')

# Labels and Title
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA: First Two Principal Components')
plt.grid()
plt.show()

# Print explained variance
for i, (var, cum_var) in enumerate(zip(explained_variance, cumulative_variance), 1):
    print(f"Principal Component {i}: Explained Variance = {var:.4f}, Cumulative Variance = {cum_var:.4f}")

# Extract loadings
pc1_loadings = loadings.loc["PC1"]
pc2_loadings = loadings.loc["PC2"]
pc3_loadings = loadings.loc["PC3"]
pc4_loadings = loadings.loc["PC4"]
pc5_loadings = loadings.loc["PC5"]

# Set feature labels (assuming they are wavelengths like 1450, 1500, etc.)
feature_labels = data.columns[1:-2].astype(float)  # Convert to float for numerical ticks

# Simplify x-axis to show ticks every 50 steps
target_positions = [1450, 1550, 1650, 1750, 2800, 3000]
tick_positions = [np.abs(feature_labels - value).argmin() for value in target_positions]
tick_labels = [f"{feature_labels[pos]:.2f}" for pos in tick_positions]

# Plot Loadings as Line Curves
plt.figure(figsize=(18, 6))

# PC1 Loadings
plt.subplot(1, 5, 1)
plt.plot(range(len(pc1_loadings)), pc1_loadings, color='steelblue', linewidth=2)
plt.title('PC1 Loadings')
plt.ylabel('Loading Value')
plt.xticks(tick_positions, tick_labels, rotation=45)  # Simplified x-axis labels

# PC2 Loadings
plt.subplot(1, 5, 2)
plt.plot(range(len(pc2_loadings)), pc2_loadings, color='orange', linewidth=2)
plt.title('PC2 Loadings')
plt.ylabel('Loading Value')
plt.xticks(tick_positions, tick_labels, rotation=45)  # Simplified x-axis labels

# PC3 Loadings
plt.subplot(1, 5, 3)
plt.plot(range(len(pc3_loadings)), pc3_loadings, color='mediumorchid', linewidth=2)
plt.title('PC3 Loadings')
plt.ylabel('Loading Value')
plt.xticks(tick_positions, tick_labels, rotation=45)  # Simplified x-axis labels

# PC4 Loadings
plt.subplot(1, 5, 4)
plt.plot(range(len(pc4_loadings)), pc4_loadings, color='mediumslateblue', linewidth=2)
plt.title('PC4 Loadings')
plt.ylabel('Loading Value')
plt.xticks(tick_positions, tick_labels, rotation=45)  # Simplified x-axis labels

# PC5 Loadings
plt.subplot(1, 5, 5)
plt.plot(range(len(pc5_loadings)), pc5_loadings, color='red', linewidth=2)
plt.title('PC5 Loadings')
plt.ylabel('Loading Value')
plt.xticks(tick_positions, tick_labels, rotation=45)  # Simplified x-axis labels



plt.tight_layout()
plt.show()

# --------------------- Bootstrap Stability Analysis -------------------------
import seaborn as sns
from sklearn.utils import resample
from scipy.spatial.distance import cosine
# Function to calculate cosine similarity between original and bootstrap components
def bootstrap_pca_cosine_similarity(X, n_components, n_bootstraps=100):
    pca_original = PCA(n_components=n_components)
    pca_original.fit(X)
    original_components = pca_original.components_

    cosine_similarities = np.zeros((n_components, n_components))

    for _ in range(n_bootstraps):
        # Resample the data
        X_resampled = resample(X)

        # Fit PCA on resampled data
        pca_bootstrap = PCA(n_components=n_components)
        pca_bootstrap.fit(X_resampled)
        bootstrap_components = pca_bootstrap.components_

        # Compute cosine similarity between original and bootstrap components
        for i in range(n_components):
            for j in range(n_components):
                similarity = 1 - cosine(original_components[i], bootstrap_components[j])
                cosine_similarities[i, j] += similarity

    # Average cosine similarity over all bootstraps
    cosine_similarities /= n_bootstraps
    return cosine_similarities


# Run bootstrap analysis
cosine_similarities = bootstrap_pca_cosine_similarity(X, n_components)

# Plot Heatmap of Cosine Similarities
plt.figure(figsize=(8, 6))
sns.heatmap(cosine_similarities, annot=True, cmap='coolwarm',
            xticklabels=[f'PC{i + 1}' for i in range(n_components)],
            yticklabels=[f'PC{i + 1}' for i in range(n_components)])
plt.title("Cosine Similarity of Principal Components (Bootstrap Stability)")
plt.xlabel("Bootstrap Components")
plt.ylabel("Original Components")
plt.show()
