
# File loading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import random
import pickle
import math
from scipy import interp
from scipy.stats import mannwhitneyu
from itertools import permutations
from tqdm import tqdm
from sklearn import decomposition, manifold, metrics
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA, FactorAnalysis, TruncatedSVD, KernelPCA, FastICA, NMF, SparsePCA
from sklearn.manifold import TSNE, MDS, Isomap
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, plot_confusion_matrix, roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score, recall_score, precision_score, r2_score, confusion_matrix
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed


def data_load(file_loc):
    """
    Load data from a CSV file and perform modifications.

    Parameters
    ----------
    file_loc : str
        The file path of the CSV file.
        - CSV file features
            First column - sample ID
            Second column - phenotype of interest
            All other columns - gene expressions
            First row ['sampleID', 'output', 'gene_name_1', 'gene_name_2', etc.]          

    Returns
    -------
    dat : pandas DataFrame
        The loaded data.

    """

    # Read the CSV file into a pandas DataFrame
    dat = pd.read_csv(file_loc, index_col=0)

    # Modify column names
    col_names = list(dat.columns)
    col_names[0] = "output"  # Update the first column name to "output"
    dat.columns = [col_names]

    # Print the shape of the data
    print("data shape:",dat.shape)

    return dat


def train_test_splitting(data, t_size = 0.2, rand_state = 12):
    """
    Perform train-test split on the data.

    Parameters
    ----------
    data : pandas DataFrame
        The input data.

    Returns
    -------
    trainX_scaled : numpy array
        Scaled training features.

    testX_scaled : numpy array
        Scaled testing features.

    y_train : pandas Series
        Training labels.

    y_test : pandas Series
        Testing labels.

    """
    dat = data.copy()

    # train test split
    #X = dat.drop(['output'], axis=1)
    X = dat.drop('output', axis=1, level=0)  # Drop the 'output' column on the first level of the multi-index

    y = dat['output']

    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=t_size, stratify=y, random_state=rand_state)

    # Squeeze the labels to remove unnecessary dimensions
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()

    # Print the shapes of the train and test sets
    print('x_train shape:',X_train.shape,'   y_train shape:', y_train.shape)
    print('x_test shape :',X_test.shape,'   y_test shape:', y_test.shape)

    # Standardize the features using StandardScaler
#     sc = StandardScaler()
#     scaler = sc.fit(X_train)
#     trainX_scaled = scaler.transform(X_train)
#     testX_scaled = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test
    
def visualize_training_data(X_train, y_train):
    """
    Perform initial visualization of the training data using PCA and t-SNE.

    Parameters
    ----------
    X_train : numpy array
        The training features.

    y_train : pandas Series
        The training labels.

    """

    # PCA plot
    pca = decomposition.PCA(n_components=2)
    pca.fit(X_train)
    pca_out = pca.transform(X_train)

    # Encode labels as integers
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(y_train)

    # Generate colormap based on the number of unique labels
    num_labels = len(np.unique(encoded_labels))
    colormap = plt.cm.get_cmap('bwr', num_labels)

    # Create a figure with subplots for PCA and t-SNE visualizations
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('Initial visualization of training data')

    # Plot PCA visualization with dynamic colors
    axs[0].scatter(pca_out[:, 0], pca_out[:, 1], c=encoded_labels, cmap=colormap, alpha=0.5)
    axs[0].set_title('PCA')
    axs[0].set_xlabel('Principal Component 1')
    axs[0].set_ylabel('Principal Component 2')

    # Plot t-SNE visualization with dynamic colors
    tsne = manifold.TSNE(n_components=2, learning_rate='auto', init='random', perplexity=30, random_state=0)
    tsne_out = tsne.fit_transform(X_train)
    axs[1].scatter(tsne_out[:, 0], tsne_out[:, 1], c=encoded_labels, cmap=colormap, alpha=0.5)
    axs[1].set_title('t-SNE')
    axs[1].set_xlabel('t-SNE Component 1')
    axs[1].set_ylabel('t-SNE Component 2')

    # Create a colorbar to show label color mapping
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=colormap), ax=axs.ravel().tolist())
    cbar.set_ticks(np.arange(num_labels))
    cbar.set_ticklabels(label_encoder.inverse_transform(np.arange(num_labels)))

    # Adjust spacing between subplots
   # plt.tight_layout()

    # Show the plot
    plt.show()


def model_evaluation(model, X_train, y_train, folds=5):
    """
    Perform k-fold cross-validation and plot stacked AUROC and AUPRC curves.

    Parameters
    ----------
    model : object
        The model to be evaluated.

    X_train : numpy array
        The training features.

    y_train : numpy array
        The training labels.

    folds : int, default=5
        The number of cross-validation folds.

    """

    # Create empty lists to store fold performances
    tprs = []
    aucs = []
    precisions = []
    recalls = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('Model Evaluation')

    # Define color map for fold lines
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(folds)]

    for fold in tqdm(range(folds)):
        # Split data into training and validation sets for the current fold
        X_train_fold, X_valid_fold, y_train_fold, y_valid_fold = train_test_split(
            X_train, y_train, test_size=0.2, random_state=fold)

        # Train the model on the training fold
        model.fit(X_train_fold, y_train_fold)

        # Perform predictions on the validation fold
        y_scores_fold = model.predict_proba(X_valid_fold)[:, 1]

        # Calculate AUROC and AUPRC for the current fold
        fpr, tpr, _ = roc_curve(y_valid_fold, y_scores_fold)
        precision, recall, _ = precision_recall_curve(y_valid_fold, y_scores_fold)

        # Interpolate the ROC curve to have the same number of points
        tpr_interpolated = interp(mean_fpr, fpr, tpr)
        tpr_interpolated[0] = 0.0
        tprs.append(tpr_interpolated)

        # Calculate AUC for the current fold
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        # Store fold performances
        precisions.append(precision[:-1])  # Remove the last element to match shape

        recalls.append(recall[:-1])  # Remove the last element to match shape

        # Plot AUROC curve for the current fold
        axes[0].plot(fpr, tpr, alpha=0.3, color=colors[fold])

        # Plot AUPRC curve for the current fold
        axes[1].plot(recall, precision, alpha=0.3, color=colors[fold])

    # Calculate mean AUROC and AUPRC across folds
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = np.mean(aucs)
    mean_precision = np.mean(precisions, axis=0)
    mean_recall = np.mean(recalls, axis=0)

    # Plot mean AUROC curve
    axes[0].plot(mean_fpr, mean_tpr, color='blue', linewidth=2,
                 label="AUROC (%0.4f)" % mean_auc)
    axes[0].plot([0, 1], [0, 1], color="grey", linestyle="--",
                 label="Baseline (0.50)")
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.0])
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('Stacked AUROC Curve')
    axes[0].legend()

    # Plot mean AUPRC curve
    axes[1].plot(mean_recall, mean_precision, color='blue', linewidth=2,
                 label="AUPRC (%0.4f)" % mean_auc)
    axes[1].plot([0, 1], [np.mean(y_train), np.mean(y_train)], color="grey", linestyle="--",
                 label="Baseline (%0.4f)" % np.mean(y_train))
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.0])
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Stacked AUPRC Curve')
    axes[1].legend()

    plt.tight_layout()
    plt.show()

def model_evaluation(model, X_train, y_train, folds=5):
    """
    Perform k-fold cross-validation and plot stacked AUROC and AUPRC curves.

    Parameters
    ----------
    model : object
        The model to be evaluated.

    X_train : numpy array
        The training features.

    y_train : numpy array
        The training labels.

    folds : int, default=5
        The number of cross-validation folds.

    """

    # Create empty lists to store fold performances
    tprs = []
    aucs = []
    precisions = []
    recalls = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('Model Evaluation')

    # Define color map for fold lines
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(folds)]

    for fold in tqdm(range(folds)):
        # Split data into training and validation sets for the current fold
        X_train_fold, X_valid_fold, y_train_fold, y_valid_fold = train_test_split(
            X_train, y_train, test_size=0.2, random_state=fold)

        # Train the model on the training fold
        model.fit(X_train_fold, y_train_fold)

        # Perform predictions on the validation fold
        y_scores_fold = model.predict_proba(X_valid_fold)[:, 1]

        # Calculate AUROC and AUPRC for the current fold
        fpr, tpr, _ = roc_curve(y_valid_fold, y_scores_fold)
        precision, recall, _ = precision_recall_curve(y_valid_fold, y_scores_fold)

        # Interpolate the ROC curve to have the same number of points
        tpr_interpolated = interp(mean_fpr, fpr, tpr)
        tpr_interpolated[0] = 0.0
        tprs.append(tpr_interpolated)

        # Calculate AUC for the current fold
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        # Store fold performances
        precisions.append(interp(mean_recall, recall, precision))
        recalls.append(mean_recall)

        # Plot AUROC curve for the current fold
        axes[0].plot(fpr, tpr, alpha=0.3, color=colors[fold])

        # Plot AUPRC curve for the current fold
        axes[1].plot(recall, precision, alpha=0.3, color=colors[fold])

    # Calculate mean AUROC and AUPRC across folds
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = np.mean(aucs)
    mean_precision = np.mean(precisions, axis=0)
    mean_recall = np.mean(recalls, axis=0)

    # Plot mean AUROC curve
    axes[0].plot(mean_fpr, mean_tpr, color='blue', linewidth=2,
                 label="AUROC (%0.4f)" % mean_auc)
    axes[0].plot([0, 1], [0, 1], color="grey", linestyle="--",
                 label="Baseline (0.50)")
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.0])
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('Stacked AUROC Curve')
    axes[0].legend()

    # Plot mean AUPRC curve
    axes[1].plot(mean_recall, mean_precision, color='blue', linewidth=2,
                 label="AUPRC (%0.4f)" % mean_auc)
    axes[1].plot([0, 1], [np.mean(y_train), np.mean(y_train)], color="grey", linestyle="--",
                 label="Baseline (%0.4f)" % np.mean(y_train))
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.0])
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Stacked AUPRC Curve')
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def model_evaluation(model, X_train, y_train, folds=5):
    """
    Perform k-fold cross-validation and plot stacked AUROC and AUPRC curves.

    Parameters
    ----------
    model : object
        The model to be evaluated.

    X_train : numpy array
        The training features.

    y_train : numpy array
        The training labels.

    folds : int, default=5
        The number of cross-validation folds.

    """

    # Create empty lists to store fold performances
    tprs = []
    aucs = []
    precisions = []
    recalls = []
    accuracies = []
    f1_scores = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('Model Evaluation')

    # Define color map for fold lines
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(folds)]

    for fold in tqdm(range(folds)):
        # Split data into training and validation sets for the current fold
        X_train_fold, X_valid_fold, y_train_fold, y_valid_fold = train_test_split(
            X_train, y_train, test_size=0.2, random_state=fold)

        # Train the model on the training fold
        model.fit(X_train_fold, y_train_fold)

        # Perform predictions on the validation fold
        y_scores_fold = model.predict_proba(X_valid_fold)[:, 1]

        # Calculate AUROC and AUPRC for the current fold
        fpr, tpr, _ = roc_curve(y_valid_fold, y_scores_fold)
        precision, recall, _ = precision_recall_curve(y_valid_fold, y_scores_fold)

        # Interpolate the ROC curve to have the same number of points
        tpr_interpolated = interp(mean_fpr, fpr, tpr)
        tpr_interpolated[0] = 0.0
        tprs.append(tpr_interpolated)

        # Calculate AUC for the current fold
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        # Store fold performances
        precisions.append(interp(mean_recall, recall, precision))
        recalls.append(mean_recall)

        # Calculate accuracy and F1 score for the current fold
        y_pred_fold = model.predict(X_valid_fold)
        accuracy = accuracy_score(y_valid_fold, y_pred_fold)
        f1 = f1_score(y_valid_fold, y_pred_fold)

        accuracies.append(accuracy)
        f1_scores.append(f1)

        # Plot AUROC curve for the current fold
        axes[0].plot(fpr, tpr, alpha=0.3, color=colors[fold])

        # Plot AUPRC curve for the current fold
        axes[1].plot(recall, precision, alpha=0.3, color=colors[fold])

    # Calculate mean AUROC and AUPRC across folds
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = np.mean(aucs)
    mean_precision = np.mean(precisions, axis=0)
    mean_recall = np.mean

def model_evaluation(model, X_train, y_train, folds=5):
    """
    Perform k-fold cross-validation and plot stacked AUROC and AUPRC curves.

    Parameters
    ----------
    model : object
        The model to be evaluated.

    X_train : numpy array
        The training features.

    y_train : numpy array
        The training labels.

    folds : int, default=5
        The number of cross-validation folds.
    """

    cv = StratifiedKFold(n_splits=folds)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    precision_array = []
    recall_array = []
    pr_aucs = []
    mean_recall = np.linspace(0, 1, 100)

    f1_scores = []
    accuracy_scores = []

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    for i, (train, test) in enumerate(cv.split(X_train, y_train)):
        model.fit(X_train[train], y_train[train])
        y_pred = model.predict(X_train[test])
        y_score = model.predict_proba(X_train[test])[:, 1]
        
        # Compute ROC curve and ROC area
        fpr, tpr, thresholds = roc_curve(y_train[test], y_score)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        
        ax1.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        
        # Compute precision-recall curve and PR area
        precision, recall, _ = precision_recall_curve(y_train[test], y_score)
        precision_array.append(np.interp(mean_recall, recall[::-1], precision[::-1]))
        pr_auc = auc(recall, precision)
        pr_aucs.append(pr_auc)
        
        ax2.plot(recall, precision, lw=1, alpha=0.3, label='PR fold %d (AUC = %0.2f)' % (i, pr_auc))
        
        # Compute F1 score and accuracy
        f1_scores.append(f1_score(y_train[test], y_pred))
        accuracy_scores.append(accuracy_score(y_train[test], y_pred))

    ax1.plot([0, 1], [0, 1], color="grey", linestyle="--",
                 label="Baseline (%0.4f)" % 0.5)
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    ax1.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f)' % mean_auc, lw=2, alpha=.8)

    ax1.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="Receiver operating characteristic example")
    ax1.legend(loc="lower right")

    ax2.plot([0, 1], [np.mean(y_train), np.mean(y_train)], color="grey", linestyle="--",
                 label="Baseline (%0.4f)" % np.mean(y_train))
    mean_precision = np.mean(precision_array, axis=0)
    mean_pr_auc = auc(mean_recall, mean_precision)
    ax2.plot(mean_recall, mean_precision, color='b', label=r'Mean PR (AUC = %0.2f)' % mean_pr_auc, lw=2, alpha=.8)
    
    ax2.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="Precision-Recall curve")
    ax2.legend(loc="lower right")

    plt.show()

    print(f"Average F1 score: {np.mean(f1_scores):.2f}")
    print(f"Average Accuracy: {np.mean(accuracy_scores):.2f}")
    print(f"Average AUROC: {np.mean(aucs):.2f}")
    print(f"Average AUPRC: {np.mean(pr_aucs):.2f}")



def apply_dimensionality_reduction(X_train, X_test, method, K):
    """
    Apply dimensionality reduction technique to training and test data.

    Parameters
    ----------
    X_train : numpy array or pandas DataFrame
        The training features.

    X_test : numpy array or pandas DataFrame
        The test features.

    method : str
        The dimensionality reduction technique.
        Options: 'PCA', 'FA', 'LDA', 'SVD', 'KernelPCA', 'Isomap', 'ICA', 'NMF', 'SparsePCA'

    K : int
        The number of dimensions for the reduced feature space.

    Returns
    -------
    X_train_reduced : numpy array
        The training features after dimensionality reduction.

    X_test_reduced : numpy array
        The test features after dimensionality reduction.

    """

    if method == 'PCA':
        reducer = PCA(n_components=K)
    elif method == 'FA':
        reducer = FactorAnalysis(n_components=K)
    elif method == 'SVD':
        reducer = TruncatedSVD(n_components=K)
    elif method == 'KernelPCA':
        reducer = KernelPCA(n_components=K)
    elif method == 'Isomap':
        reducer = Isomap(n_components=K)
    elif method == 'ICA':
        #reducer = FastICA(n_components=K)
        reducer = FastICA(n_components=K, whiten='arbitrary-variance', max_iter=100000)
    elif method == 'NMF':
        reducer = NMF(n_components=K, max_iter=100000)
    elif method == 'SparsePCA':
        reducer = SparsePCA(n_components=K)
    else:
        raise ValueError("Invalid dimensionality reduction technique. Please choose from 'PCA', 'FA', 'SVD', 'KernelPCA', 'Isomap', 'ICA', 'NMF', or 'SparsePCA'.")

    X_train_reduced = reducer.fit_transform(X_train)
    X_test_reduced = reducer.transform(X_test)

    return X_train_reduced, X_test_reduced, reducer


def plot_dim_reduction_explained_variance(X_train, max_dim_reduced, step=10):
    """
    Plot explained variance for different dimension reduction techniques.

    Parameters
    ----------
    X_train : numpy array
        The training features.

    max_dim_reduced : int
        The maximum number of dimensions to which to reduce.

    step : int
        The step size for the number of components.
    """
    methods = [
        ('PCA', PCA()),
        ('FA', FactorAnalysis()),
        ('SVD', TruncatedSVD()),
    ]

    plt.figure(figsize=(10, 8))
    total_variance = np.var(X_train, axis=0).sum()

    for name, method in methods:
        explained_variance = []
        dims = list(range(1, max_dim_reduced + 1, step))
        for dim in tqdm(dims, desc=f'Processing {name}'):
            if name in ['PCA', 'SVD']:
                reducer = method.set_params(n_components=dim)
                reducer.fit(X_train)
                explained_variance.append(np.sum(reducer.explained_variance_ratio_) * 100)
            elif name == 'FA':
                reducer = method.set_params(n_components=dim)
                reducer.fit(X_train)
                explained_variance.append(100 - (np.sum(reducer.noise_variance_) / total_variance * 100))
        plt.plot(dims, explained_variance, label=name)

    plt.xlabel('Number of Dimensions')
    plt.ylabel('Explained Variance (%)')
    plt.title('Explained Variance vs Number of Dimensions')
    plt.legend()
    plt.show()


def get_feature_importance(x_test, y_test, reducer, model, location, N):
    # Convert DataFrame to NumPy array
    x_test_arr = x_test.values

    # Calculate accuracy and R-squared of the model on the original data
    x_test_transformed = reducer.transform(x_test_arr) if reducer is not None else x_test_arr
    y_pred_orig = model.predict(x_test_transformed)
    y_pred_prob_orig = model.predict_proba(x_test_transformed)[:, 1]
    acc_orig = accuracy_score(y_test, y_pred_orig)
    r2_orig = r2_score(y_test, y_pred_prob_orig)

    # Initialize arrays to store accuracy and R-squared differences
    acc_diffs = np.zeros(x_test.shape[1])
    r2_diffs = np.zeros(x_test.shape[1])

    # Define function for calculating differences for a single column
    def calculate_diffs(col):
        acc_diff_sum = 0.0
        r2_diff_sum = 0.0

        # Permute the current column N times and calculate accuracy and R-squared differences
        for _ in range(N):
            x_test_permuted = x_test_arr.copy()
            x_test_permuted[:, col] = shuffle(x_test_permuted[:, col])
            x_test_permuted_transformed = (
                reducer.transform(x_test_permuted) if reducer is not None else x_test_permuted
            )
            y_pred_permuted = model.predict(x_test_permuted_transformed)
            y_pred_prob_permuted = model.predict_proba(x_test_permuted_transformed)[:, 1]
            acc_permuted = accuracy_score(y_test, y_pred_permuted)
            r2_permuted = r2_score(y_test, y_pred_prob_permuted)
            acc_diff_sum += acc_orig - acc_permuted
            r2_diff_sum += r2_orig - r2_permuted

        # Calculate average accuracy and R-squared differences
        acc_diff_avg = acc_diff_sum / N
        r2_diff_avg = r2_diff_sum / N

        return acc_diff_avg, r2_diff_avg

    # Loop through each column of x_test in parallel with tqdm
    results = Parallel(n_jobs=-1)(delayed(calculate_diffs)(col) for col in tqdm(range(x_test.shape[1])))

    # Extract accuracy and R-squared differences from the results
    for col, (acc_diff, r2_diff) in enumerate(results):
        acc_diffs[col] = acc_diff
        r2_diffs[col] = r2_diff

    # Plot accuracy differences
    plt.figure(figsize=(8, 6))
    plt.plot(acc_diffs)
    plt.xlabel('Feature Index')
    plt.ylabel('Accuracy Difference')
    plt.title('Accuracy Difference by Feature Permutation')
    plt.savefig(f'{location}/accuracy_diff.png')
    plt.show()

    # Plot R-squared differences
    plt.figure(figsize=(8, 6))
    plt.plot(r2_diffs)
    plt.xlabel('Feature Index')
    plt.ylabel('R-squared Difference')
    plt.title('R-squared Difference by Feature Permutation')
    plt.savefig(f'{location}/r2_diff.png')
    plt.show()

    # Save accuracy and R-squared differences
    with open(f'{location}/accuracy_diff.pkl', 'wb') as fp:
        pickle.dump(acc_diffs, fp)
    with open(f'{location}/r2_diff.pkl', 'wb') as fp:
        pickle.dump(r2_diffs, fp)


def get_importances(data, importance_location):
    genes = [l[0] for l in data.columns.tolist()][1:]

    importances = pd.DataFrame()
    for filename in os.listdir(importance_location):
        # Load the joblib file
        file_path = os.path.join(importance_location, filename)
        try:
            pickle_data = joblib.load(file_path)
            importances[filename] = pickle_data
        except:
            print(f"Error loading file: {file_path}")

    importances.index = genes
    importances = importances.rank(method='average', ascending=False)
    importances['rank_total'] = importances.sum(axis=1)

    sorted_importances = importances.sort_values('rank_total')
    sorted_importances['rank_fin'] = 1 - (sorted_importances['rank_total'] - sorted_importances['rank_total'].min()) / (sorted_importances['rank_total'].max() - sorted_importances['rank_total'].min())

    return sorted_importances


def plot_metrics_vs_genes(mlp_int, x_train, y_train, importances, genes_for_consideration):
    def cv_evaluation(model, X, y, folds=5):
        skf = StratifiedKFold(n_splits=folds, shuffle=True)
        precision_sum = []
        recall_sum = []
        f1_score_sum = []
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        
        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            y_train = y_train.reshape(-1)
            y_test = y_test.reshape(-1)
            
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
            precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
            
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            
            roc_auc = metrics.auc(fpr, tpr)
            aucs.append(roc_auc)
            
            avg_precision = average_precision_score(y_test, y_pred_proba)
            precision_sum.append(avg_precision)
            recall_sum.append(np.mean(recall))
            f1_score_sum.append(metrics.f1_score(y_test, np.round(y_pred_proba)))
            
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        
        mean_auc = metrics.auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        
        precision_avg = np.mean(precision_sum)
        recall_avg = np.mean(recall_sum)
        f1_score_avg = np.mean(f1_score_sum)
        aucs_avg = np.mean(aucs)
        
        return precision_avg, recall_avg, f1_score_avg, aucs_avg
    
    precision_avg_list = []
    recall_avg_list = []
    f_score_avg_list = []
    auc_avg_list = []
    for i in tqdm(range(len(genes_for_consideration))):
        genes = list(importances['rank_total'].head(genes_for_consideration[i]).index)
        data_important_genes = x_train[genes]
        precision, recall, f1_score, auc = cv_evaluation(mlp_int, data_important_genes, np.array(y_train).reshape(-1, 1).ravel(), folds=5)
        precision_avg_list.append(precision)
        recall_avg_list.append(recall)
        f_score_avg_list.append(f1_score)
        auc_avg_list.append(auc)
    
    fig, ax = plt.subplots()
    show_lab = len(genes_for_consideration)
    size_p = 10
    ax.plot(genes_for_consideration[0:show_lab], precision_avg_list[0:show_lab])
    ax.scatter(genes_for_consideration[0:show_lab], precision_avg_list[0:show_lab], label='precision', s =size_p)
    ax.plot(genes_for_consideration[0:show_lab], recall_avg_list[0:show_lab])
    ax.scatter(genes_for_consideration[0:show_lab], recall_avg_list[0:show_lab], label='recall', s =size_p)
    ax.plot(genes_for_consideration[0:show_lab], auc_avg_list[0:show_lab])
    ax.scatter(genes_for_consideration[0:show_lab], auc_avg_list[0:show_lab], label='auroc', s =size_p)
    # plt.ylim(0.45, 0.9)
    ax.legend()
    ax.set_xlabel('number of top genes included')
    ax.set_title('Metric Scores Varying Number of Top Genes Included')
    # plt.axvline(x=20, color='grey', linestyle='--')
    plt.show()
    plt.plot(genes_for_consideration[0:show_lab], importances['rank_fin'][0:show_lab])
    plt.scatter(genes_for_consideration[0:show_lab], importances['rank_fin'][0:show_lab])
    # plt.axvline(x = 20, color = 'grey', linestyle = '--')
    # plt.ylim(0.70, 1)
    plt.show()

def plot_differencial_expression_analysis(data_imp):
    output_col = 'output'
    data_cols = [item[0] for item in data_imp.columns]
    data_cols = [col for col in data_cols if col != output_col]

    # define the number of rows and columns for the subplot grid
    num_cols = len(data_cols)
    num_rows = math.ceil(num_cols / 4)  # adjust as needed for your desired number of columns per row

    # create the subplot grid
    fig, axes = plt.subplots(nrows=num_rows, ncols=4, figsize=(20, 5*num_rows))

    significant_proportion = 0.0
    p_values_top = list()
    g1_median = list()
    g2_median = list()

    # iterate through each column in the dataframe
    for idx, col in enumerate(data_cols):
        row_idx = idx // 4
        col_idx = idx % 4

        ax = sns.violinplot(x=[item[0] for item in list(data_imp[[output_col]].values)], 
                            y=[item[0] for item in list(data_imp[[col]].values)], ax=axes[row_idx, col_idx])

        group1 = np.concatenate(data_imp[np.concatenate(list(data_imp[output_col].values == 0)).tolist()][col].values).tolist()
        group2 = np.concatenate(data_imp[np.concatenate(list(data_imp[output_col].values == 1)).tolist()][col].values).tolist()

        g1_median = np.median(group1)
        g2_median = np.median(group2)

        statistic, pvalue = mannwhitneyu(group1, group2)
        if pvalue <= 0.05:
            significant_proportion += 1
        p_values_top.append(pvalue)

        if g1_median < g2_median:
            expression_type = "overexpressed"
        else:
            expression_type = "underexpressed"

        ax.set_title("{} ({} - p={:.5f})\nG0 Median: {:.2f}\nG1 Median: {:.2f}".format(col, expression_type, pvalue, g1_median, g2_median))

    # adjust spacing between subplots and show the plot
    fig.tight_layout()
    plt.show()

def show_model_evaluation(model, x_test, x_train, y_test, y_train):
    y_pred = model.predict(x_test)
    print('Test Accuracy: {:.2f}'.format(accuracy_score(y_test, y_pred)))
    preds_prob = model.predict_proba(x_test)
    preds_prob = [item[1] for item in preds_prob]
    y_pred_r2 = le_me.r2_score(y_test, preds_prob) 

    y_pred_train = model.predict(x_train)
    print('Train Accuracy: {:.2f} \n\n'.format(accuracy_score(y_train, y_pred_train)))
    preds_prob_train = model.predict_proba(x_train)
    preds_prob_train = [item[1] for item in preds_prob_train]
    y_pred_r2_train = le_me.r2_score(y_train, preds_prob_train) 

    print('f1:', f1_score(y_test, y_pred))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print('precision:', precision)
    print('recall:', recall)

#     def show_accuracy(model, x_test, x_train, y_test_in, y_train_in):
#         y_pred = model.predict(x_test)
#         preds_prob = model.predict_proba(x_test)
#         preds_prob = [item[1] for item in preds_prob]
#         y_pred_r2 = le_me.r2_score(y_test, preds_prob) 

#         y_pred_train = model.predict(x_train)
#         preds_prob_train = model.predict_proba(x_train)
#         preds_prob_train = [item[1] for item in preds_prob_train]
#         y_pred_r2_train = le_me.r2_score(y_train, preds_prob_train) 

#         pred_1 = [preds_prob[i] for i in list(np.where(y_test == 1)[0])]
#         pred_0 = [preds_prob[i] for i in list(np.where(y_test == 0)[0])]
# #         plt.hist([pred_1, pred_0], bins=20, label=['1', '0'], color=['red', 'blue'])
# #         plt.ylabel('count', fontsize=10)
# #         plt.xlabel('predicted probability', fontsize=10)
# #         plt.legend(loc='upper right')
# #         plt.show()

#     show_accuracy(model, x_test, x_train, y_test, y_train)
    fig = plot_confusion_matrix(model, x_test, y_test, display_labels=model.classes_)
    fig.figure_.suptitle("Confusion Matrix for MLP Pred")
    plt.show()
    
    auc, prc = plot_performance(model, x_test, y_test)
    
    return(accuracy_score(y_test, y_pred), f1_score(y_test, y_pred), auc, prc, precision, recall)
    
def plot_performance(model, x_test, y_test):
    preds = model.predict_proba(x_test)[:, 1]

    fpr, tpr, thres = roc_curve(y_test, preds)
    roc_auc = roc_auc_score(y_test, preds)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    ax = axes.ravel()

    ax[0].plot(fpr, tpr, color="blue", label="MLP (%0.4f)" % roc_auc, alpha=1)
    ax[0].plot([0, 1], [0, 1], color="grey", linestyle="--", label="Baseline (0.50)")
    ax[0].set_xlim([0.0, 1.0])
    ax[0].set_ylim([0.0, 1.0])
    ax[0].set_xlabel("1-specificity")
    ax[0].set_ylabel("sensitivity")
    ax[0].set_title("ROC Curve")
    ax[0].legend()

    precision, recall, _ = precision_recall_curve(y_test, preds)
    pr_auc = auc(recall, precision)

    ax[1].plot(recall, precision, color="blue", label="MLP (%0.4f)" % pr_auc, alpha=1)
    ax[1].plot([0, 1], [np.mean(y_test), np.mean(y_test)], color="grey", linestyle="--", label="Baseline (%0.2f)" % np.mean(y_test))
    ax[1].set_xlim([0.0, 1.0])
    ax[1].set_ylim([0.0, 1.0])
    ax[1].set_xlabel("recall")
    ax[1].set_ylabel("precision")
    ax[1].set_title("PR Curve")
    ax[1].legend()

    plt.show()
    return(roc_auc, pr_auc)


def random_feature_eval(model, train_x, test_x, train_y, test_y, numb, x):
    def evaluate_iteration(selected_features, train_x, test_x, train_y, test_y):
        train_x_selected = train_x[selected_features]
        test_x_selected = test_x[selected_features]
        model.fit(train_x_selected, train_y)
        
        y_pred = model.predict(test_x_selected)

        f1 = f1_score(test_y, y_pred)
        accuracy = accuracy_score(test_y, y_pred)
        auroc = roc_auc_score(test_y, y_pred)
        aupr = average_precision_score(test_y, y_pred)
        recall = recall_score(test_y, y_pred)
        precision = precision_score(test_y, y_pred)

        return f1, accuracy, auroc, aupr, recall, precision

    features = train_x.columns.tolist()
    results = Parallel(n_jobs=-1)(delayed(evaluate_iteration)(random.sample(features, x), train_x, test_x, train_y, test_y) for _ in tqdm(range(numb), position=0, leave=True))

    f1_scores, accuracies, aurocs, auprs, recalls, precisions = zip(*results)

    avg_f1 = np.mean(f1_scores)
    avg_accuracy = np.mean(accuracies)
    avg_auroc = np.mean(aurocs)
    avg_aupr = np.mean(auprs)
    avg_recall = np.mean(recalls)
    avg_precision = np.mean(precisions)

    print("Average F1 Score:", avg_f1)
    print("Average Accuracy:", avg_accuracy)
    print("Average AUROC:", avg_auroc)
    print("Average AUPR:", avg_aupr)
    print("Average Recall:", avg_recall)
    print("Average Precision:", avg_precision)
    
    return f1_scores, accuracies, aurocs, auprs, recalls, precisions
