import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                             recall_score, f1_score, matthews_corrcoef)
import seaborn as sns

# Load the data
file_path = "xxx.xlsx"
data = pd.read_excel(file_path)
print(data.head())

# Extract X (spectral features) and y (one-hot encoded target)
X = data.iloc[:, 1:-2].values
y = data.iloc[:, -2:].values
y_labels = np.argmax(y, axis=1)
print("Labels:", y_labels)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_labels, test_size=0.3, random_state=42, stratify=y_labels)

# Perform PCA on the training data (n_component as retained by stability analysis and scree plot)
pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Leave-One-Out Cross-Validation (LOOCV)
loo = LeaveOneOut()
train_true = []
train_pred = []
lda = LDA()

for train_index, val_index in loo.split(X_train_pca):
    X_train_inner, X_val_inner = X_train_pca[train_index], X_train_pca[val_index]
    y_train_inner, y_val_inner = y_train[train_index], y_train[val_index]

    # Fit the LDA model on the inner training set
    lda.fit(X_train_inner, y_train_inner)

    # Predict on the validation set
    y_val_pred = lda.predict(X_val_inner)
    train_pred.extend(y_val_pred)
    train_true.extend(y_val_inner)

# Calculate LOOCV performance
train_accuracy = accuracy_score(train_true, train_pred)
train_precision = precision_score(train_true, train_pred, average='weighted')
train_recall = recall_score(train_true, train_pred, average='weighted')
train_f1 = f1_score(train_true, train_pred, average='weighted')
train_mcc = matthews_corrcoef(train_true, train_pred)
print(f"LOOCV Accuracy on Resampled Training Set: {train_accuracy:.2f}")
print(f"LOOCV Precision on Resampled Training Set: {train_precision:.2f}")
print(f"LOOCV Recall on Resampled Training Set: {train_recall:.2f}")
print(f"LOOCV F1-Score on Resampled Training Set: {train_f1:.2f}")
print(f"LOOCV Matthews Correlation Coefficient (MCC) on Resampled Training Set: {train_mcc:.2f}")

# Train the model on the resampled training set and predict on the test set
lda.fit(X_train_pca, y_train)
y_test_pred = lda.predict(X_test_pca)

#Performance on the test set
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, average='weighted')
test_recall = recall_score(y_test, y_test_pred, average='weighted')
test_f1 = f1_score(y_test, y_test_pred, average='weighted')
test_mcc = matthews_corrcoef(y_test, y_test_pred)
print(f"Test set Accuracy: {test_accuracy:.2f}")
print(f"Test set Precision: {test_precision:.2f}")
print(f"Test set Recall: {test_recall:.2f}")
print(f"Test set F1-Score: {test_f1:.2f}")
print(f"Test set Matthews Correlation Coefficient (MCC): {test_mcc:.2f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix (Test Set):")
print(conf_matrix)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y_labels),
            yticklabels=np.unique(y_labels))
plt.title("Confusion Matrix (Test Set)")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.show()

# Plot LDA Projections on First Two Discriminant Axes
X_train_proj = lda.transform(X_train_pca)
plt.figure(figsize=(8, 6))

colors = ['red', 'blue']
labels = np.unique(y_train)
if X_train_proj.shape[1] > 1:
    # If more than one LDA component is available
    for i, label in enumerate(labels):
        plt.scatter(X_train_proj[y_train == label, 0], X_train_proj[y_train == label, 1],
                    label=f"Class {label}", color=colors[i])
    plt.xlabel("LDA Component 1")
    plt.ylabel("LDA Component 2")
else:
    # If only one LDA component is available
    for i, label in enumerate(labels):
        plt.scatter(X_train_proj[y_train == label, 0], np.zeros_like(X_train_proj[y_train == label, 0]),
                    label=f"Class {label}", color=colors[i])
    plt.xlabel("LDA Component 1")
    plt.ylabel("Constant (No Second Component)")

for i in range(X_train_proj.shape[0]):
    if X_train_proj.shape[1] > 1:  # More than one LDA component
        plt.text(X_train_proj[i, 0], X_train_proj[i, 1] + 0.001,
                 str(i), fontsize=8, ha='center', va='bottom', color='black')
    else:  # Only one LDA component
        plt.text(X_train_proj[i, 0], 0 + 0.001,  # Y-coordinate set to 0
                 str(i), fontsize=8, ha='center', va='bottom', color='black')

plt.title("LDA Projection (Training Set)")
plt.xlabel("LDA Component 1")
plt.ylabel("LDA Component 2")
plt.legend()
plt.show()

# Radar Plot for performance metrics
def radar_plot(metrics, train_values, test_values):
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    train_values += train_values[:1]
    test_values += test_values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Plot for training set
    ax.fill(angles, train_values, color='green', alpha=0.25, label="Training")
    ax.plot(angles, train_values, color='green', linewidth=2)

    # Plot for test set
    ax.fill(angles, test_values, color='blue', alpha=0.25, label="Test")
    ax.plot(angles, test_values, color='blue', linewidth=2)

    # Labels and ticks
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=12, color="black")
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=10)

    ax.set_ylim(0, 1) # Ensure outer circle is always 1.0

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    plt.show()

# Metrics and their values
metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "MCC"]
train_values = [train_accuracy, train_precision, train_recall, train_f1, train_mcc]
test_values = [test_accuracy, test_precision, test_recall, test_f1, test_mcc]

# Generate the radar plot
radar_plot(metrics, train_values, test_values)


#-----------------------------------------------
from sklearn.utils import resample
# Bootstrap with LOOCV to analyze model stability
def bootstrap_loocv_stability(X_train_pca, y_train, model, n_bootstraps=20):
    """
    Perform bootstrapping with Leave-One-Out Cross-Validation to assess model stability.

    Parameters:
        X_train (array-like): Training features.
        y_train (array-like): Training labels.
        model (object): Machine learning model instance.
        n_bootstraps (int): Number of bootstrap iterations.

    Returns:
        dict: Stability metrics (mean, std) for each performance metric.
    """
    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'mcc': []}

    for i in range(n_bootstraps):
        # Bootstrap resampling
        X_resampled, y_resampled = resample(X_train_pca, y_train, replace=True, random_state=i)

        # Initialize LOOCV
        loo = LeaveOneOut()
        train_true = []
        train_pred = []

        for train_index, val_index in loo.split(X_resampled):
            X_train_inner, X_val_inner = X_resampled[train_index], X_resampled[val_index]
            y_train_inner, y_val_inner = y_resampled[train_index], y_resampled[val_index]

            # Fit the model on the inner training set
            model.fit(X_train_inner, y_train_inner)

            # Predict on the validation set
            y_val_pred = model.predict(X_val_inner)
            train_pred.extend(y_val_pred)
            train_true.extend(y_val_inner)

        # Compute metrics for this bootstrap iteration
        metrics['accuracy'].append(accuracy_score(train_true, train_pred))
        metrics['precision'].append(precision_score(train_true, train_pred, average='weighted'))
        metrics['recall'].append(recall_score(train_true, train_pred, average='weighted'))
        metrics['f1'].append(f1_score(train_true, train_pred, average='weighted'))
        metrics['mcc'].append(matthews_corrcoef(train_true, train_pred))

    # Calculate mean and standard deviation for each metric
    stability_results = {metric: (np.mean(values), np.std(values)) for metric, values in metrics.items()}

    return stability_results

# Perform bootstrap LOOCV stability analysis
n_bootstraps = 20
lda_model = LDA()
stability_results = bootstrap_loocv_stability(X_train_pca, y_train, lda_model, n_bootstraps=n_bootstraps)

# Print results
print("\nBootstrap LOOCV Stability Results:")
for metric, (mean, std) in stability_results.items():
    print(f"{metric.capitalize()}: Mean = {mean:.2f}, Std = {std:.2f}")

# Visualize stability
metrics = list(stability_results.keys())
means = [stability_results[metric][0] for metric in metrics]
stds = [stability_results[metric][1] for metric in metrics]

plt.figure(figsize=(10, 6))
plt.bar(metrics, means, yerr=stds, capsize=5, alpha=0.75)
plt.title("Bootstrap LOOCV Stability (Mean ± Std)")
plt.ylabel("Performance Metric")
plt.show()
