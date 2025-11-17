import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Load the data
file_path = "xxx.xlsx"
data = pd.read_excel(file_path)

# Extract X (spectral features) and y (one-hot encoded target)
X = data.iloc[:, 1:-2].values
y = data.iloc[:, -2:].values
y_labels = np.argmax(y, axis=1)
print("Labels:", y_labels)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_labels, test_size=0.3, random_state=42, stratify=y_labels)

# Perform PCA on the training data
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Define the LDA model
lda = LDA()

# Generate learning curve
train_sizes, train_scores, test_scores = learning_curve(
    lda, X_train_pca, y_train, cv=5, scoring='accuracy', n_jobs=-1,
    train_sizes=np.linspace(0.3, 1.0, 10)
)

# Calculate mean and standard deviation for training and test scores
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# Plot learning curve
plt.figure(figsize=(8, 6))
plt.title("Learning Curve for LDA")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.grid()

# Plot training and test scores with error bars
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training Accuracy")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Validation Accuracy")

plt.legend(loc="best")
plt.show()
