import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score, recall_score,
                             f1_score, matthews_corrcoef, roc_auc_score)
import seaborn as sns
from imblearn.over_sampling import SMOTE

# Load the data
file_path = "xxx.xlsx"  # Update with your file path
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
print(f"Number of PCA components retained: {X_train_pca.shape[1]}")

# Apply SMOTE resampling to balance the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_pca, y_train)

print(f"Original Training Set Class Distribution: {np.bincount(y_train)}")
print(f"Resampled Training Set Class Distribution: {np.bincount(y_train_resampled)}")

# Leave-One-Out Cross-Validation (LOOCV) on the resampled training set + grid search for best parameters
gamma_values = [0.001, 0.01, 0.1, 1]
C_values = [0.1, 1, 10, 100]
best_params = None
best_accuracy = 0
train_true = []
train_pred = []

print("Performing LOOCV to find the best hyperparameters (C, gamma) on resampled training data...")
for C in C_values:
    for gamma in gamma_values:
        loo = LeaveOneOut()
        svm = SVC(C=C, gamma=gamma, kernel='rbf')
        accuracies = []

        for train_index, val_index in loo.split(X_train_resampled):
            X_train_inner, X_val_inner = X_train_resampled[train_index], X_train_resampled[val_index]
            y_train_inner, y_val_inner = y_train_resampled[train_index], y_train_resampled[val_index]
            svm.fit(X_train_inner, y_train_inner)
            y_val_pred = svm.predict(X_val_inner)
            accuracies.append(accuracy_score(y_val_inner, y_val_pred))

        mean_accuracy = np.mean(accuracies)
        print(f"C={C}, gamma={gamma}: LOOCV Accuracy = {mean_accuracy:.4f}")
        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_params = {'C': C, 'gamma': gamma}

print(f"Best hyperparameters found: {best_params} with LOOCV Accuracy: {best_accuracy:.4f}")

# Refit the best model on the entire resampled training set
best_svm = SVC(C=best_params['C'], gamma=best_params['gamma'], kernel='rbf')
loo = LeaveOneOut()
train_true = []
train_pred = []

for train_index, val_index in loo.split(X_train_resampled):
    X_train_inner, X_val_inner = X_train_resampled[train_index], X_train_resampled[val_index]
    y_train_inner, y_val_inner = y_train_resampled[train_index], y_train_resampled[val_index]
    best_svm.fit(X_train_inner, y_train_inner)
    y_val_pred = best_svm.predict(X_val_inner)
    train_pred.extend(y_val_pred)
    train_true.extend(y_val_inner)

# Calculate LOOCV performance for the best parameters
train_accuracy = accuracy_score(train_true, train_pred)
train_precision = precision_score(train_true, train_pred, average='weighted')
train_recall = recall_score(train_true, train_pred, average='weighted')
train_f1 = f1_score(train_true, train_pred, average='weighted')
train_mcc = matthews_corrcoef(train_true, train_pred)
print(f"LOOCV Accuracy on Resampled Training Set: {train_accuracy:.4f}")
print(f"LOOCV Precision on Resampled Training Set: {train_precision:.4f}")
print(f"LOOCV Recall on Resampled Training Set: {train_recall:.4f}")
print(f"LOOCV F1-Score on Resampled Training Set: {train_f1:.4f}")
print(f"LOOCV Matthews Correlation Coefficient (MCC) on Resampled Training Set: {train_mcc:.4f}")

# Train final SVM model with the best parameters on the resampled training set and predict on the test set
svm_final = SVC(**best_params)
svm_final.fit(X_train_resampled, y_train_resampled)
y_test_pred = svm_final.predict(X_test_pca)

# Performance on the test set
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, average='weighted')
test_recall = recall_score(y_test, y_test_pred, average='weighted')
test_f1 = f1_score(y_test, y_test_pred, average='weighted')
test_mcc = matthews_corrcoef(y_test, y_test_pred)

print("\nFinal Model Performance on Test Set:")
print(f"Accuracy: {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall: {test_recall:.4f}")
print(f"F1-Score: {test_f1:.4f}")
print(f"Matthews Correlation Coefficient (MCC): {test_mcc:.4f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y_labels), yticklabels=np.unique(y_labels))
plt.title("Confusion Matrix (Test Set)")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
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

# Decision boundary

# Transform the SMOTE-resampled data using 2D PCA
pca_2d = PCA(n_components=2)
X_train_resampled_2d = pca_2d.fit_transform(X_train_resampled)
X_test_2d = pca_2d.transform(X_test_pca)

# Also transform the original training data (for row numbering)
X_train_original_2d = pca_2d.transform(X_train_pca)

# Meshgrid for decision boundary
h = .02  # Step size in the mesh
x_min, x_max = X_train_resampled_2d[:, 0].min() - 1, X_train_resampled_2d[:, 0].max() + 1
y_min, y_max = X_train_resampled_2d[:, 1].min() - 1, X_train_resampled_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Train the SVM model on the SMOTE-resampled 2D data
svm_boundaries = SVC(**best_params)
svm_boundaries.fit(X_train_resampled_2d, y_train_resampled)

Z = svm_boundaries.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
plt.scatter(X_train_resampled_2d[:, 0], X_train_resampled_2d[:, 1], c=y_train_resampled, edgecolor='k', cmap='viridis', marker='o', label="SMOTE Resampled")

# Overlay the original training data with row numbers and slight shift
for i, (x, y) in enumerate(X_train_original_2d):
    plt.text(x + 0.05, y + 0.05, str(i), fontsize=8, color='red', ha='center', va='center')

plt.title(f"SVM Decision Boundaries with SMOTE (C={best_params['C']}, gamma={best_params['gamma']})")
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(["Decision Boundary", "SMOTE Resampled Data"])
plt.show()

#-----------------------------------------------
from sklearn.utils import resample
# Bootstrap with LOOCV to analyze model stability
def bootstrap_loocv_stability(X_train_resampled, y_train_resampled, model, n_bootstraps=20):
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
        X_resampled, y_resampled = resample(X_train_resampled, y_train_resampled, replace=True, random_state=i)

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
svm_final_model = SVC(**best_params)
stability_results = bootstrap_loocv_stability(X_train_resampled, y_train_resampled, svm_final_model, n_bootstraps=n_bootstraps)

# Print results
print("\nBootstrap LOOCV Stability Results:")
for metric, (mean, std) in stability_results.items():
    print(f"{metric.capitalize()}: Mean = {mean:.4f}, Std = {std:.4f}")

# Visualize stability
metrics = list(stability_results.keys())
means = [stability_results[metric][0] for metric in metrics]
stds = [stability_results[metric][1] for metric in metrics]

plt.figure(figsize=(10, 6))
plt.bar(metrics, means, yerr=stds, capsize=5, alpha=0.75)
plt.title("Bootstrap LOOCV Stability (Mean ± Std)")
plt.ylabel("Performance Metric")
plt.show()
