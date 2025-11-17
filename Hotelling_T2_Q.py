import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import chi2

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
n_components = 4 #min(X.shape[1], 10)  # Limit components to a max of 10 or less than features
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)

# Reconstruct X from the PCA components
X_mean = np.mean(X, axis=0)
X_reconstructed = pca.inverse_transform(X_pca)

# Calculate Q-statistic (Squared Prediction Error)
Q = np.sum((X - X_reconstructed) ** 2, axis=1)

# Calculate Hotelling's T2 statistic
eigenvalues = pca.explained_variance_
T2 = np.sum((X_pca ** 2) / eigenvalues, axis=1)

# Define thresholds
alpha = 0.99  # Confidence level
T2_limit = chi2.ppf(alpha, df=n_components)  # T2 threshold based on Chi-square
Q_limit = np.percentile(Q, 100 * alpha)     # Q threshold at 99th percentile

# Identify and print outlier indices
print("Identifying outliers...")
outliers = (T2 > T2_limit) | (Q > Q_limit)
outlier_indices = np.where(outliers)[0]

print(f"Number of outliers detected: {len(outlier_indices)}")
print("Indices of outliers:", outlier_indices)

# Link outliers to the original data
if len(outlier_indices) > 0:
    print("\nOutlier Samples:")
    outlier_data = data.iloc[outlier_indices]
    print(outlier_data)
else:
    print("No outliers detected.")

# Plot Hotelling's T2 vs Q-statistic with row numbers annotated
plt.figure(figsize=(10, 7))
plt.scatter(Q, T2, c='blue', label="Data Points")

# Annotate points with row numbers (shifted slightly to avoid overlap)
for i, (q, t2) in enumerate(zip(Q, T2)):
    plt.text(q + 0.01 * max(Q), t2 + 0.01 * max(T2), str(i),
             fontsize=8, color='black', ha='center', va='center')

# Plot thresholds
plt.axhline(y=T2_limit, color='r', linestyle='--', label=f'T2 Limit ({T2_limit:.2f})')
plt.axvline(x=Q_limit, color='g', linestyle='--', label=f'Q Limit ({Q_limit:.2f})')

plt.xlabel("Q-statistic (Squared Prediction Error)")
plt.ylabel("Hotelling's T²")
plt.title("Hotelling's T² vs Q-Statistic (Outlier Detection)")
plt.legend()
plt.grid(True)
plt.show()
