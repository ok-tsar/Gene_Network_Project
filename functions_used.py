
# File loading
import os
import math
import keras
import random
import pickle
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from itertools import permutations
from tqdm import tqdm
from umap import UMAP
from scipy import interp
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
from tensorflow.keras.models import Sequential, load_model
from joblib import Parallel, delayed
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import silhouette_score
import networkx as nx
from tensorflow.keras.models import load_model


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
        Options: 'PCA', 'FA', 'LDA', 'SVD', 'KernelPCA', 'Isomap', 'ICA', 'NMF', 'SparsePCA', 'UMAP'

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
        reducer = FastICA(n_components=K, whiten='arbitrary-variance', max_iter=100000)
    elif method == 'NMF':
        reducer = NMF(n_components=K, max_iter=100000)
    elif method == 'SparsePCA':
        reducer = SparsePCA(n_components=K)
    elif method == 'UMAP':
        reducer = UMAP(n_neighbors=5, n_components=K, min_dist=0.3, metric='correlation')
    else:
        raise ValueError("Invalid dimensionality reduction technique. Please choose from 'PCA', 'FA', 'SVD', 'KernelPCA', 'Isomap', 'ICA', 'NMF', 'SparsePCA', or 'UMAP'.")

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

def plot_hist_and_value(data, value):
    # Calculate the number of bins using the square-root rule
    n_bins = np.round(math.ceil(math.sqrt(len(data))))

    # Plot histogram
    plt.hist(data, bins=n_bins, alpha=0.5, color='g')

    # Plot vertical dotted line for single value
    plt.axvline(value, color='r', linestyle='dotted', linewidth=2)
    
    # Compute percentile of the single value within the list
    percentile = stats.percentileofscore(data, value)
    print('Percentile of single value:', percentile)
    
    # Perform one-sample t-test to check if the single value is significantly different from the mean of the list
    t_stat, p_value = stats.ttest_1samp(data, value)
    print('One-sample t-test p-value:', p_value)

    plt.show()


# def get_feature_importance(x_test, y_test, reducer, model, location, N):
#     # Convert DataFrame to NumPy array
#     x_test_arr = x_test.values

#     # Calculate accuracy and R-squared of the model on the original data
#     x_test_transformed = reducer.transform(x_test_arr) if reducer is not None else x_test_arr
#     y_pred_orig = model.predict(x_test_transformed)
#     y_pred_prob_orig = model.predict_proba(x_test_transformed)[:, 1]
#     acc_orig = accuracy_score(y_test, y_pred_orig)
#     r2_orig = r2_score(y_test, y_pred_prob_orig)

#     # Initialize arrays to store accuracy and R-squared differences
#     acc_diffs = np.zeros(x_test.shape[1])
#     r2_diffs = np.zeros(x_test.shape[1])

#     # Define function for calculating differences for a single column
#     def calculate_diffs(col):
#         acc_diff_sum = 0.0
#         r2_diff_sum = 0.0

#         # Permute the current column N times and calculate accuracy and R-squared differences
#         for _ in range(N):
#             x_test_permuted = x_test_arr.copy()
#             x_test_permuted[:, col] = shuffle(x_test_permuted[:, col])
#             x_test_permuted_transformed = (
#                 reducer.transform(x_test_permuted) if reducer is not None else x_test_permuted
#             )
#             y_pred_permuted = model.predict(x_test_permuted_transformed)
#             y_pred_prob_permuted = model.predict_proba(x_test_permuted_transformed)[:, 1]
#             acc_permuted = accuracy_score(y_test, y_pred_permuted)
#             r2_permuted = r2_score(y_test, y_pred_prob_permuted)
#             acc_diff_sum += acc_orig - acc_permuted
#             r2_diff_sum += r2_orig - r2_permuted

#         # Calculate average accuracy and R-squared differences
#         acc_diff_avg = acc_diff_sum / N
#         r2_diff_avg = r2_diff_sum / N

#         return acc_diff_avg, r2_diff_avg

#     # Loop through each column of x_test in parallel with tqdm
#     results = Parallel(n_jobs=-1)(delayed(calculate_diffs)(col) for col in tqdm(range(x_test.shape[1])))

#     # Extract accuracy and R-squared differences from the results
#     for col, (acc_diff, r2_diff) in enumerate(results):
#         acc_diffs[col] = acc_diff
#         r2_diffs[col] = r2_diff

#     # Plot accuracy differences
#     plt.figure(figsize=(8, 6))
#     plt.plot(acc_diffs)
#     plt.xlabel('Feature Index')
#     plt.ylabel('Accuracy Difference')
#     plt.title('Accuracy Difference by Feature Permutation')
#     plt.savefig(f'{location}/accuracy_diff.png')
#     plt.show()

#     # Plot R-squared differences
#     plt.figure(figsize=(8, 6))
#     plt.plot(r2_diffs)
#     plt.xlabel('Feature Index')
#     plt.ylabel('R-squared Difference')
#     plt.title('R-squared Difference by Feature Permutation')
#     plt.savefig(f'{location}/r2_diff.png')
#     plt.show()

#     # Save accuracy and R-squared differences
#     with open(f'{location}/accuracy_diff.pkl', 'wb') as fp:
#         pickle.dump(acc_diffs, fp)
#     with open(f'{location}/r2_diff.pkl', 'wb') as fp:
#         pickle.dump(r2_diffs, fp)

# def get_feature_importance(x_test, y_test, reducer, model_imp, model_loc, location, N): 
#     """
#     Calculates and visualizes the importance of each feature based on the degradation of model performance 
#     when a feature is permuted.
    
#     Parameters
#     ----------
#     x_test : pandas DataFrame or numpy array
#         The test data features.

#     y_test : pandas Series or numpy array
#         The test data labels.

#     reducer : sklearn's dimensionality reduction transformer or None
#         If provided, the reducer will be used to transform the data before prediction.
    
#     model : Sequential or sklearn's model
#         The trained model to evaluate feature importance.

#     location : str
#         The location to save the generated plots and pickled results.

#     N : int
#         The number of times to permute each feature.

#     Returns
#     -------
#     None
#     """

#     # Function to save files with incrementing filenames
#     def save_with_increment(filename):
#         # Initialize an index to append to filenames if necessary
#         i = 0
#         new_filename = filename
#         # Check if a file exists with the current filename
#         while os.path.exists(new_filename):
#             # If a file exists, split the filename into the base and extension
#             base, ext = os.path.splitext(filename)
#             # Add the index to the end of the base (before the extension)
#             new_filename = f"{base}_{i}{ext}"
#             # Increment the index
#             i += 1
#         # Return the new filename
#         return new_filename
    
#     # Convert DataFrame to NumPy array
#     x_test_arr = x_test.values

#     # Calculate accuracy and R-squared of the model on the original data
#     # If a reducer is provided, transform the test data before prediction
#     x_test_transformed = reducer.transform(x_test_arr) if reducer is not None else x_test_arr
    
#     # If model is Sequential (Keras), use .predict method, else use .predict for sklearn
#     if isinstance(model_imp, Sequential):
#         model = load_model(model_loc)
#         y_pred_prob_orig = model.predict(x_test_transformed)
#         y_pred_orig = np.round(y_pred_prob_orig)
#     else:
#         model = joblib.load(model_loc)
#         y_pred_orig = model.predict(x_test_transformed)
#         y_pred_prob_orig = model.predict_proba(x_test_transformed)[:, 1]

#     # Calculate the accuracy and R-squared of the original predictions
#     acc_orig = accuracy_score(y_test, y_pred_orig)
#     r2_orig = r2_score(y_test, y_pred_prob_orig)

#     # Initialize arrays to store accuracy and R-squared differences
#     acc_diffs = np.zeros(x_test.shape[1])
#     r2_diffs = np.zeros(x_test.shape[1])

#     # Define function for calculating differences for a single column
#     def calculate_diffs(col):
# #         if isinstance(model_imp, Sequential):
# #             int_model = load_model(model_loc)
# #         else:
# #             int_model = joblib.load(model_loc)
#         # Initialize sum of differences for accuracy and R-squared
#         acc_diff_sum = 0.0
#         r2_diff_sum = 0.0

#         # Permute the current column N times and calculate accuracy and R-squared differences
#         for _ in range(N):
#             # Copy the test data and shuffle one column for permutation
#             x_test_permuted = x_test_arr.copy()
#             x_test_permuted[:, col] = shuffle(x_test_permuted[:, col])
#             x_test_permuted_transformed = (
#                 reducer.transform(x_test_permuted) if reducer is not None else x_test_permuted)
            
#             # Predict on the permuted data
#             if isinstance(model, Sequential):
#                 y_pred_prob_permuted = int_model.predict(x_test_permuted_transformed)
#                 y_pred_permuted = np.round(y_pred_prob_permuted)
#             else:
#                 y_pred_permuted = int_model.predict(x_test_permuted_transformed)
#                 y_pred_prob_permuted = int_model.predict_proba(x_test_permuted_transformed)[:, 1]
                
#             # Calculate the accuracy and R-squared of the permuted data
#             acc_permuted = accuracy_score(y_test, y_pred_permuted)
#             r2_permuted = r2_score(y_test, y_pred_prob_permuted)
            
#             # Add the differences of accuracy and R-squared to the sum
#             acc_diff_sum += acc_orig - acc_permuted
#             r2_diff_sum += r2_orig - r2_permuted

#         # Calculate average accuracy and R-squared differences
#         acc_diff_avg = acc_diff_sum / N
#         r2_diff_avg = r2_diff_sum / N
#         del int_model
#         return acc_diff_avg, r2_diff_avg

#     # Loop through each column of x_test in parallel with tqdm
#     # Each job calculates the differences for one column
#     results = Parallel(n_jobs=-1)(delayed(calculate_diffs)(col) for col in tqdm(range(x_test.shape[1]), position=0, leave=True))

#     # Extract accuracy and R-squared differences from the results
#     for col, (acc_diff, r2_diff) in enumerate(results):
#         acc_diffs[col] = acc_diff
#         r2_diffs[col] = r2_diff

#     # Plot accuracy differences
#     plt.figure(figsize=(8, 6))
#     plt.plot(acc_diffs)
#     plt.xlabel('Feature Index')
#     plt.ylabel('Accuracy Difference')
#     plt.title('Accuracy Difference by Feature Permutation')
#     plt.savefig(save_with_increment(f'{location}/accuracy_diff.png'))
#     plt.show()

#     # Plot R-squared differences
#     plt.figure(figsize=(8, 6))
#     plt.plot(r2_diffs)
#     plt.xlabel('Feature Index')
#     plt.ylabel('R-squared Difference')
#     plt.title('R-squared Difference by Feature Permutation')
#     plt.savefig(save_with_increment(f'{location}/r2_diff.png'))
#     plt.show()

#     # Save accuracy and R-squared differences as .pkl files
#     with open(save_with_increment(f'{location}/accuracy_diff.pkl'), 'wb') as fp:
#         pickle.dump(acc_diffs, fp)
#     with open(save_with_increment(f'{location}/r2_diff.pkl'), 'wb') as fp:
#         pickle.dump(r2_diffs, fp)
        
# def get_feature_importance_seq(x_test, y_test, reducer, model, model_loc, location, N):
#     def save_with_increment(filename):
#         # Initialize an index to append to filenames if necessary
#         i = 0
#         new_filename = filename
#         # Check if a file exists with the current filename
#         while os.path.exists(new_filename):
#             # If a file exists, split the filename into the base and extension
#             base, ext = os.path.splitext(filename)
#             # Add the index to the end of the base (before the extension)
#             new_filename = f"{base}_{i}{ext}"
#             # Increment the index
#             i += 1
#         # Return the new filename
#         return new_filename
    
#     # Convert DataFrame to NumPy array
#     x_test_arr = x_test.values

#     # Calculate accuracy and R-squared of the model on the original data
#     x_test_transformed = reducer.transform(x_test_arr) if reducer is not None else x_test_arr
#     y_pred_orig = np.round(model.predict(x_test_transformed))
#     y_pred_prob_orig = model.predict(x_test_transformed)
#     acc_orig = accuracy_score(y_test, y_pred_orig)
#     r2_orig = r2_score(y_test, y_pred_prob_orig)

#     # Initialize arrays to store accuracy and R-squared differences
#     acc_diffs = np.zeros(x_test.shape[1])
#     r2_diffs = np.zeros(x_test.shape[1])

#     # Define function for calculating differences for a single column
#     def calculate_diffs(col):
#         # Load model here
#         #model = keras.models.load_model('./model_final/m2_umap100_MLP') # model_path is the path where your trained model is saved
#         model = load_model(model_loc)
#         acc_diff_sum = 0.0
#         r2_diff_sum = 0.0

#         # Permute the current column N times and calculate accuracy and R-squared differences
#         for _ in range(N):
#             x_test_permuted = x_test_arr.copy()
#             x_test_permuted[:, col] = shuffle(x_test_permuted[:, col])
#             x_test_permuted_transformed = (
#                 reducer.transform(x_test_permuted) if reducer is not None else x_test_permuted
#             )
#             y_pred_permuted = np.round(model.predict(x_test_permuted_transformed))
#             y_pred_prob_permuted = model.predict(x_test_permuted_transformed)
#             acc_permuted = accuracy_score(y_test, y_pred_permuted)
#             r2_permuted = r2_score(y_test, y_pred_prob_permuted)
#             acc_diff_sum += acc_orig - acc_permuted
#             r2_diff_sum += r2_orig - r2_permuted

#         # Calculate average accuracy and R-squared differences
#         acc_diff_avg = acc_diff_sum / N
#         r2_diff_avg = r2_diff_sum / N

#         return acc_diff_avg, r2_diff_avg

#     # Loop through each column of x_test in parallel with tqdm
#     results = Parallel(n_jobs=-1)(delayed(calculate_diffs)(col) for col in tqdm(range(x_test.shape[1])))

#     # Extract accuracy and R-squared differences from the results
#     for col, (acc_diff, r2_diff) in enumerate(results):
#         acc_diffs[col] = acc_diff
#         r2_diffs[col] = r2_diff

#     # Plot accuracy differences
#     plt.figure(figsize=(8, 6))
#     plt.plot(acc_diffs)
#     plt.xlabel('Feature Index')
#     plt.ylabel('Accuracy Difference')
#     plt.title('Accuracy Difference by Feature Permutation')
#     plt.savefig(save_with_increment(f'{location}/accuracy_diff.png'))
#     plt.show()

#     # Plot R-squared differences
#     plt.figure(figsize=(8, 6))
#     plt.plot(r2_diffs)
#     plt.xlabel('Feature Index')
#     plt.ylabel('R-squared Difference')
#     plt.title('R-squared Difference by Feature Permutation')
#     plt.savefig(save_with_increment(f'{location}/r2_diff.png'))
#     plt.show()

#     # Save accuracy and R-squared differences
#     with open(save_with_increment(f'{location}/accuracy_diff.pkl'), 'wb') as fp:
#         pickle.dump(acc_diffs, fp)
#     with open(save_with_increment(f'{location}/r2_diff.pkl'), 'wb') as fp:
#         pickle.dump(r2_diffs, fp)
        


# def get_feature_importance(x_test, y_test, reducer, model_imp, location, N):
#     # Function to save files with incrementing filenames
#     def save_with_increment(filename):
#         # Initialize an index to append to filenames if necessary
#         i = 0
#         new_filename = filename
#         # Check if a file exists with the current filename
#         while os.path.exists(new_filename):
#             # If a file exists, split the filename into the base and extension
#             base, ext = os.path.splitext(filename)
#             # Add the index to the end of the base (before the extension)
#             new_filename = f"{base}_{i}{ext}"
#             # Increment the index
#             i += 1
#         # Return the new filename
#         return new_filename
    
#     # Convert DataFrame to NumPy array
#     x_test_arr = x_test.values

#     # Calculate accuracy and R-squared of the model on the original data
#     # If a reducer is provided, transform the test data before prediction
#     x_test_transformed = reducer.transform(x_test_arr) if reducer is not None else x_test_arr
    
#     # If model is Sequential (Keras), use .predict method, else use .predict for sklearn
#     if isinstance(model_imp, Sequential):
#         model = load_model(model_loc)
#         y_pred_prob_orig = model.predict(x_test_transformed)
#         y_pred_orig = np.round(y_pred_prob_orig)
#     else:
#         model = joblib.load(model_loc)
#         y_pred_orig = model.predict(x_test_transformed)
#         y_pred_prob_orig = model.predict_proba(x_test_transformed)[:, 1]

#     # Calculate the accuracy and R-squared of the original predictions
#     acc_orig = accuracy_score(y_test, y_pred_orig)
#     r2_orig = r2_score(y_test, y_pred_prob_orig)

#     # Initialize arrays to store accuracy and R-squared differences
#     acc_diffs = np.zeros(x_test.shape[1])
#     r2_diffs = np.zeros(x_test.shape[1])
    
#     def calculate_diffs(col):
#         # Do not load model here anymore
#         # Initialize sum of differences for accuracy and R-squared
#         acc_diff_sum = 0.0
#         r2_diff_sum = 0.0

#         # Permute the current column N times and calculate accuracy and R-squared differences
#         for _ in range(N):
#             # Copy the test data and shuffle one column for permutation
#             x_test_permuted = x_test_arr.copy()
#             x_test_permuted[:, col] = shuffle(x_test_permuted[:, col])
#             x_test_permuted_transformed = (
#                 reducer.transform(x_test_permuted) if reducer is not None else x_test_permuted)
            
#             # Return the permuted data for prediction outside of this function
#             return x_test_permuted_transformed


#     # Loop through each column of x_test in parallel with tqdm
#     # Each job returns the permuted data for one column
#     results = Parallel(n_jobs=-1)(delayed(calculate_diffs)(col) for col in tqdm(range(x_test.shape[1]), position=0, leave=True))

#     # Predict on the permuted data outside of the parallel jobs
#     for col, x_test_permuted_transformed in tqdm(enumerate(results), position=0, leave=True):
#         if isinstance(model_imp, Sequential):
#             y_pred_prob_permuted = model_imp.predict(x_test_permuted_transformed)
#             y_pred_permuted = np.round(y_pred_prob_permuted)

#         else:
#             y_pred_permuted = model_imp.predict(x_test_permuted_transformed)
#             y_pred_prob_permuted = model_imp.predict_proba(x_test_permuted_transformed)[:, 1]

        
#         # Calculate the accuracy and R-squared of the permuted data
#         acc_permuted = accuracy_score(y_test, y_pred_permuted)
#         r2_permuted = r2_score(y_test, y_pred_prob_permuted)
        
#         # Add the differences of accuracy and R-squared to the sum
#         acc_diffs[col] = acc_orig - acc_permuted
#         r2_diffs[col] = r2_orig - r2_permuted
#      # Plot accuracy differences
#     plt.figure(figsize=(8, 6))
#     plt.plot(acc_diffs)
#     plt.xlabel('Feature Index')
#     plt.ylabel('Accuracy Difference')
#     plt.title('Accuracy Difference by Feature Permutation')
#     plt.savefig(save_with_increment(f'{location}/accuracy_diff.png'))
#     plt.show()

#     # Plot R-squared differences
#     plt.figure(figsize=(8, 6))
#     plt.plot(r2_diffs)
#     plt.xlabel('Feature Index')
#     plt.ylabel('R-squared Difference')
#     plt.title('R-squared Difference by Feature Permutation')
#     plt.savefig(save_with_increment(f'{location}/r2_diff.png'))
#     plt.show()

#     # Save accuracy and R-squared differences as .pkl files
#     with open(save_with_increment(f'{location}/accuracy_diff.pkl'), 'wb') as fp:
#         pickle.dump(acc_diffs, fp)
#     with open(save_with_increment(f'{location}/r2_diff.pkl'), 'wb') as fp:
#         pickle.dump(r2_diffs, fp)


def get_feature_importance(x_test, y_test, reducer, model_loc, location, N):
    """
    This function evaluates feature importance by permuting the features of a test dataset and
    comparing the model's performance with the permuted data to its performance with the original data.
    The difference in performance serves as an indication of the feature's importance.

    Parameters:
    x_test (DataFrame): The test data features
    y_test (Series): The test data target
    reducer (PCA or other dimensionality reduction object, optional): If provided, this is used to transform the data before prediction
    model_loc (str): The location of the model file to be loaded for prediction
    location (str): The directory to save the output files
    N (int): The number of times to permute each feature

    Outputs:
    - Plots of accuracy and R-squared differences by feature
    - Pickle files of accuracy and R-squared differences
    """

    # This function checks if a file already exists with the given filename, and if so, appends an index to the filename
    def save_with_increment(filename):
        i = 0
        new_filename = filename
        while os.path.exists(new_filename):
            base, ext = os.path.splitext(filename)
            new_filename = f"{base}_{i}{ext}"
            i += 1
        return new_filename

    # Convert DataFrame to NumPy array
    x_test_arr = x_test.values
    # Transform test data if reducer is provided
    x_test_transformed = reducer.transform(x_test_arr) if reducer is not None else x_test_arr

    # Load the model only once
    # The method of loading depends on whether the model was trained with Keras (.h5 file) or sklearn (.pkl file)
    if os.path.splitext(model_loc)[1] == '.h5':
        model = load_model(model_loc)
        y_pred_prob_orig = model.predict(x_test_transformed)
        y_pred_orig = np.round(y_pred_prob_orig)
    else:
        model = joblib.load(model_loc)
        y_pred_orig = model.predict(x_test_transformed)
        y_pred_prob_orig = model.predict_proba(x_test_transformed)[:, 1]

    # Calculate the model's original performance
    acc_orig = accuracy_score(y_test, y_pred_orig)
    r2_orig = r2_score(y_test, y_pred_prob_orig)

    # Initialize arrays to store performance differences
    acc_diffs = np.zeros(x_test.shape[1])
    r2_diffs = np.zeros(x_test.shape[1])

    # This function shuffles a single feature (column) of the test data
    def calculate_diffs(col):
        x_test_permuted = x_test_arr.copy()
        np.random.shuffle(x_test_permuted[:, col])
        x_test_permuted_transformed = (
            reducer.transform(x_test_permuted) if reducer is not None else x_test_permuted)
        return x_test_permuted_transformed

    # Generate permuted test data in parallel
    results = Parallel(n_jobs=-1)(delayed(calculate_diffs)(col) for col in tqdm(range(x_test.shape[1]), position=0, leave=True))

    # Use the model to predict on the permuted data
    for col, x_test_permuted_transformed in tqdm(enumerate(results), position=0, leave=True):
        if os.path.splitext(model_loc)[1] == '.h5':
            y_pred_prob_permuted = model.predict(x_test_permuted_transformed)
            y_pred_permuted = np.round(y_pred_prob_permuted)
        else:
            y_pred_permuted = model.predict(x_test_permuted_transformed)
            y_pred_prob_permuted = model.predict_proba(x_test_permuted_transformed)[:, 1]

        # Calculate the model's performance with the permuted data
        acc_permuted = accuracy_score(y_test, y_pred_permuted)
        r2_permuted = r2_score(y_test, y_pred_prob_permuted)

        # Calculate the difference in performance
        acc_diffs[col] = acc_orig - acc_permuted
        r2_diffs[col] = r2_orig - r2_permuted

    # Save and plot the results
    with open(save_with_increment(f'{location}/accuracy_diff.pkl'), 'wb') as fp:
        pickle.dump(acc_diffs, fp)
    with open(save_with_increment(f'{location}/r2_diff.pkl'), 'wb') as fp:
        pickle.dump(r2_diffs, fp)
    plt.figure(figsize=(8, 6))
    plt.plot(acc_diffs)
    plt.xlabel('Feature Index')
    plt.ylabel('Accuracy Difference')
    plt.title('Accuracy Difference by Feature Permutation')
    plt.savefig(save_with_increment(f'{location}/accuracy_diff.png'))
    plt.show()
    plt.figure(figsize=(8, 6))
    plt.plot(r2_diffs)
    plt.xlabel('Feature Index')
    plt.ylabel('R-squared Difference')
    plt.title('R-squared Difference by Feature Permutation')
    plt.savefig(save_with_increment(f'{location}/r2_diff.png'))
    plt.show()


# def fetch_importances(data, importance_location):
#     genes = [l[0] for l in data.columns.tolist()][1:]

#     importances = pd.DataFrame()
#     for filename in os.listdir(importance_location):
#         # Load the joblib file
#         file_path = os.path.join(importance_location, filename)
#         try:
#             pickle_data = joblib.load(file_path)
#             importances[filename] = pickle_data
#         except:
#             print(f"Error loading file: {file_path}")

#     importances.index = genes
#     importances = importances.rank(method='average', ascending=False)
#     importances['rank_total'] = importances.sum(axis=1)

#     sorted_importances = importances.sort_values('rank_total')
#     sorted_importances['rank_fin'] = 1 - (sorted_importances['rank_total'] - sorted_importances['rank_total'].min()) / (sorted_importances['rank_total'].max() - sorted_importances['rank_total'].min())

#     return sorted_importances

def fetch_importances(data, importance_location):
    """
    Loads, ranks, and normalizes feature importances saved as joblib (pickle) files in a specific location.

    Parameters
    ----------
    data : pandas DataFrame
        The input data where each column corresponds to a feature. The first column is assumed to be non-informative 
        and is ignored. The order and labels of the remaining columns should correspond to the features in the 
        loaded joblib files.

    importance_location : str
        The directory where the joblib (pickle) files of feature importances are stored.

    Returns
    -------
    sorted_importances : pandas DataFrame
        A DataFrame where each row represents a feature and each column corresponds to the rank of that feature's 
        importance as determined by a specific model. Additional columns "rank_total" and "rank_fin" represent 
        the sum of ranks across all models and the normalized rank respectively. The DataFrame is sorted by "rank_total".
    """
    # Generate a list of features, excluding the first non-informative column
    features = [l[0] for l in data.columns.tolist()][1:]

    # Initialize DataFrame to store importances
    importances = pd.DataFrame()

    # Loop through each file in the specified directory
    for filename in os.listdir(importance_location):
        # Check if the file is a pickle file before loading
        if filename.endswith(".pkl"):
            # Construct the full file path
            file_path = os.path.join(importance_location, filename)

            # Attempt to load the pickle file and add its content to the importances DataFrame
            try:
                pickle_data = joblib.load(file_path)
                importances[filename] = pickle_data
            except:
                print(f"Error loading file: {file_path}")

    # Assign features as the index of the DataFrame
    importances.index = features

    # Rank importances for each feature across all models
    importances = importances.rank(method='average', ascending=False)

    # Calculate the sum of ranks for each feature across all models
    importances['rank_total'] = importances.sum(axis=1)

    # Sort the DataFrame by the total rank
    sorted_importances = importances.sort_values('rank_total')

    # Normalize the total rank for each feature
    sorted_importances['rank_fin'] = 1 - (sorted_importances['rank_total'] - sorted_importances['rank_total'].min()) / (sorted_importances['rank_total'].max() - sorted_importances['rank_total'].min())

    return sorted_importances


# def plot_metrics_vs_genes(mlp_int, x_train, y_train, importances, genes_for_consideration):
#     def cv_evaluation(model, X, y, folds=5):
#         skf = StratifiedKFold(n_splits=folds, shuffle=True)
#         precision_sum = []
#         recall_sum = []
#         f1_score_sum = []
#         tprs = []
#         aucs = []
#         mean_fpr = np.linspace(0, 1, 100)
        
#         for i, (train_index, test_index) in enumerate(skf.split(X, y)):
#             X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#             y_train, y_test = y[train_index], y[test_index]
#             y_train = y_train.reshape(-1)
#             y_test = y_test.reshape(-1)
            
#             model.fit(X_train, y_train)
#             y_pred_proba = model.predict_proba(X_test)[:, 1]
            
#             fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
#             precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
            
#             tprs.append(np.interp(mean_fpr, fpr, tpr))
#             tprs[-1][0] = 0.0
            
#             roc_auc = metrics.auc(fpr, tpr)
#             aucs.append(roc_auc)
            
#             avg_precision = average_precision_score(y_test, y_pred_proba)
#             precision_sum.append(avg_precision)
#             recall_sum.append(np.mean(recall))
#             f1_score_sum.append(metrics.f1_score(y_test, np.round(y_pred_proba)))
            
#         mean_tpr = np.mean(tprs, axis=0)
#         mean_tpr[-1] = 1.0
        
#         mean_auc = metrics.auc(mean_fpr, mean_tpr)
#         std_auc = np.std(aucs)
#         std_tpr = np.std(tprs, axis=0)
#         tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
#         tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        
#         precision_avg = np.mean(precision_sum)
#         recall_avg = np.mean(recall_sum)
#         f1_score_avg = np.mean(f1_score_sum)
#         aucs_avg = np.mean(aucs)
        
#         return precision_avg, recall_avg, f1_score_avg, aucs_avg
    
#     precision_avg_list = []
#     recall_avg_list = []
#     f_score_avg_list = []
#     auc_avg_list = []
#     for i in tqdm(range(len(genes_for_consideration))):
#         genes = list(importances['rank_total'].head(genes_for_consideration[i]).index)
#         data_important_genes = x_train[genes]
#         precision, recall, f1_score, auc = cv_evaluation(mlp_int, data_important_genes, np.array(y_train).reshape(-1, 1).ravel(), folds=5)
#         precision_avg_list.append(precision)
#         recall_avg_list.append(recall)
#         f_score_avg_list.append(f1_score)
#         auc_avg_list.append(auc)
    
#     fig, ax = plt.subplots()
#     show_lab = len(genes_for_consideration)
#     size_p = 10
#     ax.plot(genes_for_consideration[0:show_lab], precision_avg_list[0:show_lab])
#     ax.scatter(genes_for_consideration[0:show_lab], precision_avg_list[0:show_lab], label='precision', s =size_p)
#     ax.plot(genes_for_consideration[0:show_lab], recall_avg_list[0:show_lab])
#     ax.scatter(genes_for_consideration[0:show_lab], recall_avg_list[0:show_lab], label='recall', s =size_p)
#     ax.plot(genes_for_consideration[0:show_lab], auc_avg_list[0:show_lab])
#     ax.scatter(genes_for_consideration[0:show_lab], auc_avg_list[0:show_lab], label='auroc', s =size_p)
#     # plt.ylim(0.45, 0.9)
#     ax.legend()
#     ax.set_xlabel('number of top genes included')
#     ax.set_title('Metric Scores Varying Number of Top Genes Included')
#     # plt.axvline(x=20, color='grey', linestyle='--')
#     plt.show()
#     plt.plot(genes_for_consideration[0:show_lab], importances['rank_fin'][0:show_lab])
#     plt.scatter(genes_for_consideration[0:show_lab], importances['rank_fin'][0:show_lab])
#     # plt.axvline(x = 20, color = 'grey', linestyle = '--')
#     # plt.ylim(0.70, 1)
#     plt.show()

# def plot_metrics_vs_genes(mlp_int, x_train, y_train, importances, genes_for_consideration, threshold):
#     def cv_evaluation(model, X, y, folds=5):
#         skf = StratifiedKFold(n_splits=folds, shuffle=True)
#         precision_sum = []
#         recall_sum = []
#         f1_score_sum = []
#         tprs = []
#         aucs = []
#         mean_fpr = np.linspace(0, 1, 100)
        
#         for i, (train_index, test_index) in enumerate(skf.split(X, y)):
#             X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#             y_train, y_test = y[train_index], y[test_index]
#             y_train = y_train.reshape(-1)
#             y_test = y_test.reshape(-1)
            
#             model.fit(X_train, y_train)
#             y_pred_proba = model.predict_proba(X_test)[:, 1]
            
#             fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
#             precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
            
#             tprs.append(np.interp(mean_fpr, fpr, tpr))
#             tprs[-1][0] = 0.0
            
#             roc_auc = metrics.auc(fpr, tpr)
#             aucs.append(roc_auc)
            
#             avg_precision = average_precision_score(y_test, y_pred_proba)
#             precision_sum.append(avg_precision)
#             recall_sum.append(np.mean(recall))
#             f1_score_sum.append(metrics.f1_score(y_test, np.round(y_pred_proba)))
            
#         mean_tpr = np.mean(tprs, axis=0)
#         mean_tpr[-1] = 1.0
        
#         mean_auc = metrics.auc(mean_fpr, mean_tpr)
#         std_auc = np.std(aucs)
#         std_tpr = np.std(tprs, axis=0)
#         tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
#         tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        
#         precision_avg = np.mean(precision_sum)
#         recall_avg = np.mean(recall_sum)
#         f1_score_avg = np.mean(f1_score_sum)
#         aucs_avg = np.mean(aucs)
        
#         return precision_avg, recall_avg, f1_score_avg, aucs_avg

#     precision_avg_list = []
#     recall_avg_list = []
#     f_score_avg_list = []
#     auc_avg_list = []
    
#     for i in tqdm(range(len(genes_for_consideration))):
#         genes = list(importances['rank_total'].head(genes_for_consideration[i]).index)
#         data_important_genes = x_train[genes]
#         precision, recall, f1_score, auc = cv_evaluation(mlp_int, data_important_genes, np.array(y_train).reshape(-1, 1).ravel(), folds=5)
#         precision_avg_list.append(precision)
#         recall_avg_list.append(recall)
#         f_score_avg_list.append(f1_score)
#         auc_avg_list.append(auc)
    
#     # Fit a polynomial regression to the AUROC values
#     poly = np.polyfit(genes_for_consideration, auc_avg_list, deg=3)
#     polyder = np.polyder(poly)  # Derivative of the polynomial

#     # Calculate the value of the derivative at each point
#     polyder_values = np.polyval(polyder, genes_for_consideration)
    
#     # Find the first point where the derivative falls below the threshold
#     cutoff_index = np.where(polyder_values < threshold)[0]
#     if cutoff_index.size > 0:
#         cutoff = genes_for_consideration[cutoff_index[0]]
#     else:
#         cutoff = genes_for_consideration[-1]
    
#     fig, ax = plt.subplots()
#     show_lab = len(genes_for_consideration)
#     size_p = 10
#     ax.plot(genes_for_consideration[0:show_lab], precision_avg_list[0:show_lab])
#     ax.scatter(genes_for_consideration[0:show_lab], precision_avg_list[0:show_lab], label='precision', s =size_p)
#     ax.plot(genes_for_consideration[0:show_lab], recall_avg_list[0:show_lab])
#     ax.scatter(genes_for_consideration[0:show_lab], recall_avg_list[0:show_lab], label='recall', s =size_p)
#     ax.plot(genes_for_consideration[0:show_lab], auc_avg_list[0:show_lab])
#     ax.scatter(genes_for_consideration[0:show_lab], auc_avg_list[0:show_lab], label='auroc', s =size_p)
    
#     # Add the smoothed fitted line to the plot
#     smoothed_fitted_line = np.polyval(poly, genes_for_consideration[0:show_lab])
#     ax.plot(genes_for_consideration[0:show_lab], smoothed_fitted_line, color='green',alpha = 0.3, label='auroc fitted line')
    
#     # Add vertical line at the cutoff
#     ax.axvline(x=cutoff, color='r', linestyle='--', label='proposed cutoff')
    
#     ax.legend()
#     ax.set_xlabel('number of top genes included')
#     ax.set_title('Metric Scores Varying Number of Top Genes Included\n Proposed Cutoff: {:.0f}'.format(cutoff))
#     plt.show()

#     return cutoff

def plot_metrics_vs_genes(mlp_int, x_train, y_train, importances, genes_for_consideration, threshold):
    """
    Evaluates a model's performance for varying numbers of the most important features and plots 
    the resulting metrics. Additionally, identifies an optimal cutoff for the number of features 
    to include based on a threshold for the derivative of the AUROC curve.

    Parameters
    ----------
    mlp_int : sklearn estimator instance
        The machine learning model to evaluate (untrained).

    x_train : pandas DataFrame
        The training data, with samples as rows and genes as columns.

    y_train : pandas Series or numpy array
        The training labels.

    importances : pandas DataFrame
        The gene importance data, where each row represents a gene and each column represents the 
        importance of that gene from different files.

    genes_for_consideration : list
        List of integers indicating the number of top genes to consider in each iteration.

    threshold : float
        Threshold value for the derivative of the polynomial regression fit to the AUROC values. 
        This is used to determine the cutoff point for the number of genes to consider.

    Returns
    -------
    cutoff : int
        The suggested cutoff for the number of genes to consider, based on where the derivative 
        of the fitted polynomial drops below the threshold.

    """
    
    # Function to perform cross-validation evaluation
    def cv_evaluation(model, X, y, folds=5):
        """
        This function performs stratified k-fold cross-validation, computes precision, recall, 
        f1 score, and AUC for each fold, and averages these values across all folds.
        """
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

    # Lists to store average metric values across all folds for each number of top genes
    precision_avg_list = []
    recall_avg_list = []
    f_score_avg_list = []
    auc_avg_list = []
    
    # Iterate over all numbers of top genes to consider
    for i in tqdm(range(len(genes_for_consideration)), position=0, leave=True)):
        # Get top genes
        genes = list(importances['rank_total'].head(genes_for_consideration[i]).index)
        # Subset the data to these genes
        data_important_genes = x_train[genes]
        # Perform CV and get metrics
        precision, recall, f1_score, auc = cv_evaluation(mlp_int, data_important_genes, np.array(y_train).reshape(-1, 1).ravel(), folds=5)
        # Store the metrics
        precision_avg_list.append(precision)
        recall_avg_list.append(recall)
        f_score_avg_list.append(f1_score)
        auc_avg_list.append(auc)
    
    # Fit a polynomial regression to the AUROC values
    poly = np.polyfit(genes_for_consideration, auc_avg_list, deg=3)
    # Take derivative of the polynomial
    polyder = np.polyder(poly)
    # Calculate the value of the derivative at each point
    polyder_values = np.polyval(polyder, genes_for_consideration)
    
    # Find the first point where the derivative falls below the threshold
    cutoff_index = np.where(polyder_values < threshold)[0]
    if cutoff_index.size > 0:
        cutoff = genes_for_consideration[cutoff_index[0]]
    else:
        cutoff = genes_for_consideration[-1]
    
    # Plotting the results
    fig, ax = plt.subplots()
    show_lab = len(genes_for_consideration)
    size_p = 10
    ax.plot(genes_for_consideration[0:show_lab], precision_avg_list[0:show_lab])
    ax.scatter(genes_for_consideration[0:show_lab], precision_avg_list[0:show_lab], label='precision', s =size_p)
    ax.plot(genes_for_consideration[0:show_lab], recall_avg_list[0:show_lab])
    ax.scatter(genes_for_consideration[0:show_lab], recall_avg_list[0:show_lab], label='recall', s =size_p)
    ax.plot(genes_for_consideration[0:show_lab], auc_avg_list[0:show_lab])
    ax.scatter(genes_for_consideration[0:show_lab], auc_avg_list[0:show_lab], label='auroc', s =size_p)
    
    # Add the smoothed fitted line to the plot
    smoothed_fitted_line = np.polyval(poly, genes_for_consideration[0:show_lab])
    ax.plot(genes_for_consideration[0:show_lab], smoothed_fitted_line, color='green',alpha = 0.3, label='auroc fitted line')
    
    # Add vertical line at the cutoff
    ax.axvline(x=cutoff, color='r', linestyle='--', label='proposed cutoff')
    
    ax.legend()
    ax.set_xlabel('number of top genes included')
    ax.set_title('Metric Scores Varying Number of Top Genes Included\n Proposed Cutoff: {:.0f}'.format(cutoff))
    plt.show()

    return cutoff


# def plot_metrics_vs_genes(mlp_int, x_train, y_train, importances, genes_for_consideration, threshold):
#     """
#     Evaluates a model's performance for varying numbers of the most important features and plots 
#     the resulting metrics. Additionally, identifies an optimal cutoff for the number of features 
#     to include based on a threshold for the derivative of the AUROC curve.

#     Parameters
#     ----------
#     mlp_int : sklearn estimator instance
#         The machine learning model to evaluate.

#     x_train : pandas DataFrame
#         The training features.

#     y_train : pandas Series or numpy array
#         The training labels.

#     importances : pandas DataFrame
#         A DataFrame where each row represents a feature and each column corresponds to the rank of that feature's 
#         importance as determined by a specific model. Additional columns "rank_total" and "rank_fin" represent 
#         the sum of ranks across all models and the normalized rank respectively. The DataFrame should be sorted by "rank_total".

#     genes_for_consideration : list of int
#         A list of integers representing the varying numbers of the top-ranked features to consider.

#     threshold : float
#         The derivative value of the smoothed AUROC curve below which the optimal number of features is identified.

#     Returns
#     -------
#     cutoff : int
#         The proposed optimal number of features to include based on the specified threshold for the derivative 
#         of the smoothed AUROC curve.
#     """
#     # Define helper function for cross-validated model evaluation
#     def cv_evaluation(model, X, y, folds=5):
#         # Code for cross-validated model evaluation goes here

#     # Lists to store average precision, recall, F1 score, and AUROC for each number of top features
#     precision_avg_list = []
#     recall_avg_list = []
#     f_score_avg_list = []
#     auc_avg_list = []
    
#     # Evaluate model performance for each number of top features
#     for i in tqdm(range(len(genes_for_consideration))):
#         # Code for model evaluation goes here

#     # Fit a polynomial to the AUROC values and determine the first point where its derivative falls below the threshold
#     # Code for polynomial fit and finding the cutoff goes here

#     # Plot average precision, recall, AUROC, smoothed AUROC curve, and cutoff
#     # Code for plotting goes here

#     return cutoff


# def plot_differencial_expression_analysis(data_imp):
#     output_col = 'output'
#     data_cols = [item[0] for item in data_imp.columns]
#     data_cols = [col for col in data_cols if col != output_col]

#     # define the number of rows and columns for the subplot grid
#     num_cols = len(data_cols)
#     num_rows = math.ceil(num_cols / 4)  # adjust as needed for your desired number of columns per row

#     # create the subplot grid
#     fig, axes = plt.subplots(nrows=num_rows, ncols=4, figsize=(20, 5*num_rows))

#     significant_proportion = 0.0
#     p_values_top = list()
#     g1_median = list()
#     g2_median = list()

#     # iterate through each column in the dataframe
#     for idx, col in enumerate(data_cols):
#         row_idx = idx // 4
#         col_idx = idx % 4

#         ax = sns.violinplot(x=[item[0] for item in list(data_imp[[output_col]].values)], 
#                             y=[item[0] for item in list(data_imp[[col]].values)], ax=axes[row_idx, col_idx])

#         group1 = np.concatenate(data_imp[np.concatenate(list(data_imp[output_col].values == 0)).tolist()][col].values).tolist()
#         group2 = np.concatenate(data_imp[np.concatenate(list(data_imp[output_col].values == 1)).tolist()][col].values).tolist()

#         g1_median = np.median(group1)
#         g2_median = np.median(group2)

#         statistic, pvalue = mannwhitneyu(group1, group2)
#         if pvalue <= 0.05:
#             significant_proportion += 1
#         p_values_top.append(pvalue)

#         if g1_median < g2_median:
#             expression_type = "overexpressed"
#         else:
#             expression_type = "underexpressed"

#         ax.set_title("{} ({} - p={:.5f})\nG0 Median: {:.2f}\nG1 Median: {:.2f}".format(col, expression_type, pvalue, g1_median, g2_median))

#     # adjust spacing between subplots and show the plot
#     fig.tight_layout()
#     plt.show()

                                                                                       
def plot_differencial_expression_analysis(data_imp):
    """
    Perform differential expression analysis for each gene based on output and plots the results.
    Statistical test used: Mann-Whitney U test between the two groups at 0.05 significance

    Parameters
    ----------
    data_imp : pandas DataFrame
        The input data. Each column should represent a gene and contain expression 
        values for that gene. One column should be the 'output' column, 
        indicating the class label for each sample.

    Returns
    -------
    None : 
        This function doesn't return any value. Its purpose is to perform 
        statistical tests on each gene and generate plots.
    """
    
    # Define the output column name
    output_col = 'output'
    
    # Get the list of column names (genes), excluding the output column
    data_cols = [item[0] for item in data_imp.columns]
    data_cols = [col for col in data_cols if col != output_col]

    # Define the number of rows and columns for the subplot grid
    num_cols = len(data_cols)
    num_rows = math.ceil(num_cols / 4)  # 4 columns per row

    # Create the subplot grid
    fig, axes = plt.subplots(nrows=num_rows, ncols=4, figsize=(20, 5*num_rows))

    # Define variables to store information about the tests
    significant_proportion = 0.0
    p_values_top = list()
    g1_median = list()
    g2_median = list()

    # Iterate through each column (gene) in the dataframe
    for idx, col in enumerate(data_cols):
        row_idx = idx // 4
        col_idx = idx % 4

        # Generate a violin plot for this gene, showing the distribution of 
        # expression values for the two classes
        ax = sns.violinplot(x=[item[0] for item in list(data_imp[[output_col]].values)], 
                            y=[item[0] for item in list(data_imp[[col]].values)], ax=axes[row_idx, col_idx])

        # Split the expression values for this gene into two groups, 
        # corresponding to the two classes
        group1 = np.concatenate(data_imp[np.concatenate(list(data_imp[output_col].values == 0)).tolist()][col].values).tolist()
        group2 = np.concatenate(data_imp[np.concatenate(list(data_imp[output_col].values == 1)).tolist()][col].values).tolist()

        # Calculate the median of each group
        g1_median = np.median(group1)
        g2_median = np.median(group2)

        # Perform a Mann-Whitney U test between the two groups
        statistic, pvalue = mannwhitneyu(group1, group2)
        
        # If the p-value is less than 0.05, increment the count of significant tests
        if pvalue <= 0.05:
            significant_proportion += 1
        p_values_top.append(pvalue)

        # Determine if the gene is overexpressed or underexpressed in class 1 compared to class 0
        if g1_median < g2_median:
            expression_type = "overexpressed"
        else:
            expression_type = "underexpressed"

        # Set the title for this subplot, showing the gene name, 
        # whether it is over- or under-expressed, the p-value from the test, 
        # and the median expression values for the two classes
        ax.set_title("{} ({} - p={:.5f})\nG0 Median: {:.2f}\nG1 Median: {:.2f}".format(col, expression_type, pvalue, g1_median, g2_median))

    # Adjust spacing between subplots and show the plot
    fig.tight_layout()
    plt.show()


# def show_model_evaluation(model, x_test, x_train, y_test, y_train):
#     if isinstance(model, Sequential):
#         y_pred_prob = model.predict(x_test)
#         y_pred = (y_pred_prob > 0.5).astype(int)
#         y_pred_train_prob = model.predict(x_train)
#         y_pred_train = (y_pred_train_prob > 0.5).astype(int)
#     else:
#         y_pred = model.predict(x_test)
#         y_pred_prob = model.predict_proba(x_test)[:, 1]
#         y_pred_train = model.predict(x_train)
#         y_pred_train_prob = model.predict_proba(x_train)[:, 1]

#     print('Test Accuracy: {:.2f}'.format(accuracy_score(y_test, y_pred)))
#     print('Train Accuracy: {:.2f} \n\n'.format(accuracy_score(y_train, y_pred_train)))

#     print('f1:', f1_score(y_test, y_pred))
#     tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
#     precision = tp / (tp + fp)
#     recall = tp / (tp + fn)
#     print('precision:', precision)
#     print('recall:', recall)
    
#     conf_matrix = confusion_matrix(y_test, y_pred)
    
#     plt.figure(figsize=(5,5))
#     sns.heatmap(conf_matrix, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'viridis');
#     plt.ylabel('Actual label');
#     plt.xlabel('Predicted label');
#     plt.suptitle("Confusion Matrix")
#     #all_sample_title = 'Accuracy Score: {0}'.format(accuracy_score(y_test, y_pred))
#     #plt.title(all_sample_title, size = 15);

#     precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
#     prc = auc(recall, precision)
#     print('AUC-PRC:', prc)
    
#     # Calculate the AUROC
#     auroc = roc_auc_score(y_test, y_pred_prob)
#     print('AUC-ROC:', auroc)


#     return(accuracy_score(y_test, y_pred), f1_score(y_test, y_pred), auroc, prc, precision, recall)

def show_model_evaluation(model, x_test, x_train, y_test, y_train):
    """
    Evaluate the performance of a trained model.

    Parameters
    ----------
    model : sklearn model or keras Sequential model
        The pre-trained machine learning model.
    x_test : pandas DataFrame or numpy array
        The test dataset.
    x_train : pandas DataFrame or numpy array
        The train dataset.
    y_test : pandas Series or numpy array
        The true labels for the test dataset.
    y_train : pandas Series or numpy array
        The true labels for the train dataset.

    Returns
    -------
    tuple
        A tuple containing the accuracy score, F1 score, area under the ROC 
        curve (AUROC), area under the precision-recall curve (AUPRC), precision, 
        and recall for the model on the test set.
    """

    # Check the type of the model
    # If the model is a Keras Sequential model, use the predict method 
    # to get the predicted probabilities and convert to binary predictions
    if isinstance(model, Sequential):
        y_pred_prob = model.predict(x_test)
        y_pred = (y_pred_prob > 0.5).astype(int)
        y_pred_train_prob = model.predict(x_train)
        y_pred_train = (y_pred_train_prob > 0.5).astype(int)
    # If the model is not a Sequential model (i.e., it's a sklearn model), 
    # use the predict and predict_proba methods
    else:
        y_pred = model.predict(x_test)
        y_pred_prob = model.predict_proba(x_test)[:, 1]
        y_pred_train = model.predict(x_train)
        y_pred_train_prob = model.predict_proba(x_train)[:, 1]

    # Print the accuracy score for the model on the test and train sets
    print('Test Accuracy: {:.2f}'.format(accuracy_score(y_test, y_pred)))
    print('Train Accuracy: {:.2f} \n\n'.format(accuracy_score(y_train, y_pred_train)))

    # Print the F1 score for the model on the test set
    print('f1:', f1_score(y_test, y_pred))
    
    # Calculate and print the precision and recall for the model on the test set
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print('precision:', precision)
    print('recall:', recall)
    
    # Plot the confusion matrix for the model predictions on the test set
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,5))
    sns.heatmap(conf_matrix, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'viridis');
    plt.ylabel('Actual label');
    plt.xlabel('Predicted label');
    plt.suptitle("Confusion Matrix")

    # Calculate and print the area under the precision-recall curve (AUPRC)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    prc = auc(recall, precision)
    print('AUC-PRC:', prc)
    
    # Calculate and print the area under the ROC curve (AUROC)
    auroc = roc_auc_score(y_test, y_pred_prob)
    print('AUC-ROC:', auroc)

    # Return the accuracy score, F1 score, AUROC, AUPRC, precision, and recall
    return(accuracy_score(y_test, y_pred), f1_score(y_test, y_pred), auroc, prc, precision, recall)

def test_evaluation(model, x_test, x_train, y_test, y_train):
    """
    Evaluate the performance of a trained model.

    Parameters
    ----------
    model : sklearn model or keras Sequential model
        The pre-trained machine learning model.
    x_test : pandas DataFrame or numpy array
        The test dataset.
    x_train : pandas DataFrame or numpy array
        The train dataset.
    y_test : pandas Series or numpy array
        The true labels for the test dataset.
    y_train : pandas Series or numpy array
        The true labels for the train dataset.

    Returns
    -------
    tuple
        A tuple containing the accuracy score, F1 score, area under the ROC 
        curve (AUROC), area under the precision-recall curve (AUPRC), precision, 
        and recall for the model on the test set.
    """

    # Check the type of the model
    # If the model is a Keras Sequential model, use the predict method 
    # to get the predicted probabilities and convert to binary predictions
    if isinstance(model, Sequential):
        y_pred_prob = model.predict(x_test)
        y_pred = (y_pred_prob > 0.5).astype(int)
        y_pred_train_prob = model.predict(x_train)
        y_pred_train = (y_pred_train_prob > 0.5).astype(int)
    # If the model is not a Sequential model (i.e., it's a sklearn model), 
    # use the predict and predict_proba methods
    else:
        y_pred = model.predict(x_test)
        y_pred_prob = model.predict_proba(x_test)[:, 1]
        y_pred_train = model.predict(x_train)
        y_pred_train_prob = model.predict_proba(x_train)[:, 1]

    # Print the accuracy score for the model on the test and train sets
    print('Test Accuracy: {:.2f}'.format(accuracy_score(y_test, y_pred)))
    print('Train Accuracy: {:.2f} \n\n'.format(accuracy_score(y_train, y_pred_train)))

    # Print the F1 score for the model on the test set
    print('f1:', f1_score(y_test, y_pred))
    
    # Calculate and print the precision and recall for the model on the test set
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print('precision:', precision)
    print('recall:', recall)
    
    # Plot the confusion matrix for the model predictions on the test set
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,5))
    sns.heatmap(conf_matrix, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'viridis');
    plt.ylabel('Actual label');
    plt.xlabel('Predicted label');
    plt.suptitle("Confusion Matrix")

    # Calculate and print the area under the precision-recall curve (AUPRC)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    prc = auc(recall, precision)
    print('AUC-PRC:', prc)
    
    # Calculate and print the area under the ROC curve (AUROC)
    auroc = roc_auc_score(y_test, y_pred_prob)
    print('AUC-ROC:', auroc)
    
    # Plot the AUROC curve and AUPRC curve side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # AUROC curve
    fpr, tpr, thres = roc_curve(y_test, y_pred_prob)
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    ax1.plot(fpr, tpr, color="blue", label="Model (%0.4f)" % roc_auc, alpha=1)
    ax1.plot([0, 1], [0, 1], color="grey", linestyle="--", label="Baseline (0.50)")
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.0])
    ax1.set_xlabel("1-specificity")
    ax1.set_ylabel("sensitivity")
    ax1.set_title("ROC Curve")
    ax1.legend()
    
    # AUPRC curve
    ax2.plot(recall, precision, color="blue", label="Model (%0.4f)" % prc, alpha=1)
    ax2.plot([0, 1], [np.mean(y_test), np.mean(y_test)], color="grey", linestyle="--", label="Baseline (%0.2f)" % np.mean(y_test))
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.0])
    ax2.set_xlabel("recall")
    ax2.set_ylabel("precision")
    ax2.set_title("PR Curve")
    ax2.legend()
    
    plt.show()
    # Return the accuracy score, F1 score, AUROC, AUPRC, precision, and recall
    return accuracy_score(y_test, y_pred), f1_score(y_test, y_pred), auroc, prc, precision, recall

# import matplotlib.pyplot as plt
# import seaborn as sns

# def show_model_evaluation_2(model, x_test, x_train, y_test, y_train):
#     y_pred = model.predict(x_test)
#     y_pred = np.round(y_pred)
#     preds_prob = model.predict(x_test)
#     print('Test Accuracy: {:.2f}'.format(accuracy_score(y_test, y_pred)))
#     y_pred_r2 = le_me.r2_score(y_test, preds_prob) 

#     y_pred_train = model.predict(x_train)
#     y_pred_train = np.round(y_pred_train)
#     preds_prob_train = model.predict(x_train)
#     print('Train Accuracy: {:.2f} \n\n'.format(accuracy_score(y_train, y_pred_train)))
#     y_pred_r2_train = le_me.r2_score(y_train, preds_prob_train) 

#     print('f1:', f1_score(y_test, y_pred))
#     tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
#     precision = tp / (tp + fp)
#     recall = tp / (tp + fn)
#     print('precision:', precision)
#     print('recall:', recall)

#     confusion_matrix = tf.math.confusion_matrix(y_test, y_pred)

#     plt.figure(figsize=(10, 7))
#     sns.heatmap(confusion_matrix, annot=True, fmt='d')
#     plt.title("Confusion Matrix for MLP Pred")
#     plt.xlabel('Predicted')
#     plt.ylabel('Truth')
#     plt.show()

#     auc, prc = plot_performance(model, x_test, y_test)
#     return(accuracy_score(y_test, y_pred), f1_score(y_test, y_pred), auc, prc, precision, recall)

# def show_model_evaluation_2(model, x_test, x_train, y_test, y_train):
#     y_pred = model.predict(x_test)
#     y_pred = np.round(y_pred)
#     preds_prob = model.predict(x_test)
#     print('Test Accuracy: {:.2f}'.format(accuracy_score(y_test, y_pred)))
#     # preds_prob = model.predict_proba(x_test)
#     # preds_prob = [item[1] for item in preds_prob]
#     y_pred_r2 = le_me.r2_score(y_test, preds_prob) 

#     y_pred_train = model.predict(x_train)
#     y_pred_train = np.round(y_pred_train)
#     preds_prob_train = model.predict(x_train)
#     print('Train Accuracy: {:.2f} \n\n'.format(accuracy_score(y_train, y_pred_train)))
# #     preds_prob_train = model.predict_proba(x_train)
# #     preds_prob_train = [item[1] for item in preds_prob_train]
#     y_pred_r2_train = le_me.r2_score(y_train, preds_prob_train) 

#     print('f1:', f1_score(y_test, y_pred))
#     tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
#     precision = tp / (tp + fp)
#     recall = tp / (tp + fn)
#     print('precision:', precision)
#     print('recall:', recall)

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
    #fig = plot_confusion_matrix(model, x_test, y_test, display_labels=model.classes_)
#     fig = tf.math.confusion_matrix(y_test, y_pred)
#     fig.figure_.suptitle("Confusion Matrix for MLP Pred")
#     plt.show()
    
#     cm = tf.math.confusion_matrix(y_test, y_pred).numpy()
#     plt.figure(figsize=(10, 7))
#     sns.heatmap(cm, annot=True, fmt='d')
#     plt.title("Confusion Matrix for MLP Pred")
#     plt.xlabel('Predicted')
#     plt.ylabel('Truth')
#     plt.show()
    
#     auc, prc = plot_performance_2(model, x_test, y_test)
#     return(accuracy_score(y_test, y_pred), f1_score(y_test, y_pred), auc, prc, precision, recall)

    
# def plot_performance(model, x_test, y_test):
#     preds = model.predict_proba(x_test)[:, 1]

#     fpr, tpr, thres = roc_curve(y_test, preds)
#     roc_auc = roc_auc_score(y_test, preds)

#     fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
#     ax = axes.ravel()

#     ax[0].plot(fpr, tpr, color="blue", label="MLP (%0.4f)" % roc_auc, alpha=1)
#     ax[0].plot([0, 1], [0, 1], color="grey", linestyle="--", label="Baseline (0.50)")
#     ax[0].set_xlim([0.0, 1.0])
#     ax[0].set_ylim([0.0, 1.0])
#     ax[0].set_xlabel("1-specificity")
#     ax[0].set_ylabel("sensitivity")
#     ax[0].set_title("ROC Curve")
#     ax[0].legend()

#     precision, recall, _ = precision_recall_curve(y_test, preds)
#     pr_auc = auc(recall, precision)

#     ax[1].plot(recall, precision, color="blue", label="MLP (%0.4f)" % pr_auc, alpha=1)
#     ax[1].plot([0, 1], [np.mean(y_test), np.mean(y_test)], color="grey", linestyle="--", label="Baseline (%0.2f)" % np.mean(y_test))
#     ax[1].set_xlim([0.0, 1.0])
#     ax[1].set_ylim([0.0, 1.0])
#     ax[1].set_xlabel("recall")
#     ax[1].set_ylabel("precision")
#     ax[1].set_title("PR Curve")
#     ax[1].legend()

#     plt.show()
#     return(roc_auc, pr_auc)

# def plot_performance_2(model, x_test, y_test):
#     preds = model.predict(x_test)

#     fpr, tpr, thres = roc_curve(y_test, preds)
#     roc_auc = roc_auc_score(y_test, preds)

#     fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
#     ax = axes.ravel()

#     ax[0].plot(fpr, tpr, color="blue", label="MLP (%0.4f)" % roc_auc, alpha=1)
#     ax[0].plot([0, 1], [0, 1], color="grey", linestyle="--", label="Baseline (0.50)")
#     ax[0].set_xlim([0.0, 1.0])
#     ax[0].set_ylim([0.0, 1.0])
#     ax[0].set_xlabel("1-specificity")
#     ax[0].set_ylabel("sensitivity")
#     ax[0].set_title("ROC Curve")
#     ax[0].legend()

#     precision, recall, _ = precision_recall_curve(y_test, preds)
#     pr_auc = auc(recall, precision)

#     ax[1].plot(recall, precision, color="blue", label="MLP (%0.4f)" % pr_auc, alpha=1)
#     ax[1].plot([0, 1], [np.mean(y_test), np.mean(y_test)], color="grey", linestyle="--", label="Baseline (%0.2f)" % np.mean(y_test))
#     ax[1].set_xlim([0.0, 1.0])
#     ax[1].set_ylim([0.0, 1.0])
#     ax[1].set_xlabel("recall")
#     ax[1].set_ylabel("precision")
#     ax[1].set_title("PR Curve")
#     ax[1].legend()

#     plt.show()
#     return(roc_auc, pr_auc)


# def random_feature_eval(model, train_x, test_x, train_y, test_y, numb, x):
#     def evaluate_iteration(selected_features, train_x, test_x, train_y, test_y):
#         train_x_selected = train_x[selected_features]
#         test_x_selected = test_x[selected_features]
#         model.fit(train_x_selected, train_y)
        
#         y_pred = model.predict(test_x_selected)

#         f1 = f1_score(test_y, y_pred)
#         accuracy = accuracy_score(test_y, y_pred)
#         auroc = roc_auc_score(test_y, y_pred)
#         aupr = average_precision_score(test_y, y_pred)
#         recall = recall_score(test_y, y_pred)
#         precision = precision_score(test_y, y_pred)

#         return f1, accuracy, auroc, aupr, recall, precision

#     features = train_x.columns.tolist()
#     results = Parallel(n_jobs=-1)(delayed(evaluate_iteration)(random.sample(features, x), train_x, test_x, train_y, test_y) for _ in tqdm(range(numb), position=0, leave=True))

#     f1_scores, accuracies, aurocs, auprs, recalls, precisions = zip(*results)

#     avg_f1 = np.mean(f1_scores)
#     avg_accuracy = np.mean(accuracies)
#     avg_auroc = np.mean(aurocs)
#     avg_aupr = np.mean(auprs)
#     avg_recall = np.mean(recalls)
#     avg_precision = np.mean(precisions)

#     print("Average F1 Score:", avg_f1)
#     print("Average Accuracy:", avg_accuracy)
#     print("Average AUROC:", avg_auroc)
#     print("Average AUPR:", avg_aupr)
#     print("Average Recall:", avg_recall)
#     print("Average Precision:", avg_precision)
    
#     return f1_scores, accuracies, aurocs, auprs, recalls, precisions

def random_feature_eval(model, train_x, test_x, train_y, test_y, numb, x):
    """
    Evaluate the performance of a model with a random subset of features many times.

    Parameters
    ----------
    model : sklearn model
        The pre-trained machine learning model.
    train_x : pandas DataFrame
        The training dataset.
    test_x : pandas DataFrame
        The test dataset.
    train_y : pandas Series or numpy array
        The true labels for the training dataset.
    test_y : pandas Series or numpy array
        The true labels for the test dataset.
    numb : int
        The number of iterations to perform.
    x : int
        The number of features to sample in each iteration.

    Returns
    -------
    tuple of lists
        A tuple containing lists of F1 scores, accuracies, AUROCs, AUPRs, 
        recall scores, and precision scores for each iteration.
    """

    # Define the inner function for a single evaluation iteration
    def evaluate_iteration(selected_features, train_x, test_x, train_y, test_y):
        # Select the features for this iteration
        train_x_selected = train_x[selected_features]
        test_x_selected = test_x[selected_features]
        
        # Fit the model and make predictions
        model.fit(train_x_selected, train_y)
        y_pred = model.predict(test_x_selected)

        # Calculate performance metrics
        f1 = f1_score(test_y, y_pred)
        accuracy = accuracy_score(test_y, y_pred)
        auroc = roc_auc_score(test_y, y_pred)
        aupr = average_precision_score(test_y, y_pred)
        recall = recall_score(test_y, y_pred)
        precision = precision_score(test_y, y_pred)

        return f1, accuracy, auroc, aupr, recall, precision

    # Get the list of all features
    features = train_x.columns.tolist()

    # Evaluate the model numb times with x randomly selected features each time
    # Note: Parallel and delayed are functions from the joblib library that are used to parallelize the process
    results = Parallel(n_jobs=-1)(delayed(evaluate_iteration)(random.sample(features, x), train_x, test_x, train_y, test_y) for _ in tqdm(range(numb), position=0, leave=True))

    # Unzip the results into separate lists
    f1_scores, accuracies, aurocs, auprs, recalls, precisions = zip(*results)

    # Calculate and print the average performance metrics
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
    
    # Return the performance metrics for each iteration
    return f1_scores, accuracies, aurocs, auprs, recalls, precisions

# def calculate_biomarker(df, significance_level):
#     # Convert dataframe to matrix (numpy array)
#     matrix = df.values
#     columns = [t[0] for t in list(df.columns)]
#     output_index = [t[0] for t in list(df.columns)].index('output')

#     # Create a new dataframe
#     result_df = pd.DataFrame()
    
#     # Create an ID column
#     result_df['id'] = df.index

#     # Initialize biomarker column with zeros
#     result_df['biomarker'] = 0.0
#     total_sig = 0
#     overexp = 0
#     underexp = 0

#     # Loop through the columns in the original dataframe (now matrix)
#     for i in range(matrix.shape[1]):
#         if i != output_index:
#             # Divide the dataframe into two groups based on 'output'
#             group0 = matrix[matrix[:, output_index] == 0, i]
#             group1 = matrix[matrix[:, output_index] == 1, i]

#             # Perform Mann-Whitney U test
#             stat, p_value = mannwhitneyu(group0, group1)
#             print(p_value)

#             # If p_value is less than significance level, adjust 'biomarker' value
#             if p_value < significance_level:
#                 total_sig += 1
#                 if np.mean(group1) > np.mean(group0):
#                     # Over expressed in group 1
#                     overexp += 1
#                     result_df['biomarker'] += df.iloc[:, i]
#                 else:
#                     # Under expressed in group 1
#                     underexp += 1
#                     result_df['biomarker'] -= df.iloc[:, i]

#     result_df['output'] = df['output']
#     print('Significantly Differencially Expressed: ',total_sig)
#     print('Overexpressed (in AD) : ',overexp)
#     print('Underexpressed (in AD): ',underexp)
#     return result_df

# def calculate_biomarker(df, significance_level):
#     # Convert dataframe to matrix (numpy array)
#     matrix = df.values
#     columns = df.columns
#     output_index = list(columns).index(('output',))  # assuming 'output' is the top level of the multiindex

#     # Create a new dataframe
#     result_df = pd.DataFrame()
    
#     # Create an ID column
#     result_df['id'] = df.index

#     # Initialize biomarker column with zeros
#     result_df['biomarker'] = 0.0
#     total_sig = 0
#     overexp = 0
#     underexp = 0

#     # Loop through the columns in the original dataframe (now matrix)
#     for i in range(matrix.shape[1]):
#         if i != output_index:
#             # Divide the dataframe into two groups based on 'output'
#             group0 = matrix[matrix[:, output_index] == 0, i]
#             group1 = matrix[matrix[:, output_index] == 1, i]

#             # Perform Mann-Whitney U test
#             stat, p_value = mannwhitneyu(group0, group1)

#             # If p_value is less than significance level, adjust 'biomarker' value
#             if p_value < significance_level:
#                 total_sig += 1
#                 if np.mean(group1) > np.mean(group0):
#                     # Over expressed in group 1
#                     overexp += 1
#                     #result_df['biomarker'] += df.iloc[:, i]
#                     result_df['biomarker'] += pd.Series(list(df.iloc[:, i]))
#                 else:
#                     # Under expressed in group 1
#                     underexp += 1
#                     result_df['biomarker'] -= pd.Series(list(df.iloc[:, i]))

#     result_df['output'] = pd.Series(list(df[('output',)]))    

#     print('Significantly Differentially Expressed: ', total_sig)
#     print('Overexpressed (in AD) : ', overexp)
#     print('Underexpressed (in AD): ', underexp)
    
#     return result_df

def calculate_biomarker(df, significance_level):
    """
    Calculates a biomarker value based on the significant differential expression of features.

    Parameters
    ----------
    df : pandas DataFrame
        The input DataFrame. Each column corresponds to a feature and each row corresponds to a sample. 
        The DataFrame includes a column named 'output' which contains the labels for the two groups (0 and 1).
    significance_level : float
        The significance level for the Mann-Whitney U test.

    Returns
    -------
    result_df : pandas DataFrame
        The output DataFrame with an 'id' column (original index), a 'biomarker' column (biomarker values for each sample),
        and an 'output' column (original output labels).
    """

    # Convert dataframe to matrix (numpy array)
    matrix = df.values

    # Store column names and find the index of the 'output' column
    columns = df.columns
    output_index = list(columns).index(('output',))  # assuming 'output' is the top level of the multiindex

    # Create a new dataframe to store results
    result_df = pd.DataFrame()
    
    # Create an 'id' column with original DataFrame's index
    result_df['id'] = df.index

    # Initialize 'biomarker' column with zeros
    result_df['biomarker'] = 0.0

    # Initialize variables to keep track of significant, overexpressed, and underexpressed features
    total_sig = 0
    overexp = 0
    underexp = 0

    # Loop through the columns (features) in the original dataframe (now a numpy array)
    for i in range(matrix.shape[1]):
        if i != output_index:
            # Divide the samples into two groups based on 'output'
            group0 = matrix[matrix[:, output_index] == 0, i]
            group1 = matrix[matrix[:, output_index] == 1, i]

            # Perform Mann-Whitney U test to check if distributions of the two groups are significantly different
            stat, p_value = mannwhitneyu(group0, group1)

            # If p_value is less than the chosen significance level, adjust the 'biomarker' value
            if p_value < significance_level:
                total_sig += 1
                if np.mean(group1) > np.mean(group0):
                    # If the feature is overexpressed in group 1
                    overexp += 1
                    result_df['biomarker'] += pd.Series(list(df.iloc[:, i]))
                else:
                    # If the feature is underexpressed in group 1
                    underexp += 1
                    result_df['biomarker'] -= pd.Series(list(df.iloc[:, i]))

    # Add original 'output' column to the result DataFrame
    result_df['output'] = pd.Series(list(df[('output',)]))    

    # Print the number of significantly differentially expressed features, as well as number of overexpressed and underexpressed ones
    print('Significantly Differentially Expressed: ', total_sig)
    print('Overexpressed (in AD) : ', overexp)
    print('Underexpressed (in AD): ', underexp)
    
    # Return the result DataFrame with calculated biomarker values
    return result_df


# def plot_differential_violin(df, column, group_by):
#     # Split the data into two groups based on 'group_by' column
#     group0 = df[df[group_by] == 0][column].dropna()
#     group1 = df[df[group_by] == 1][column].dropna()
    
#     # Perform Mann-Whitney U test
#     _, p_value = mannwhitneyu(group0, group1)
    
#     # Create a new DataFrame for plotting
#     plot_df = pd.DataFrame({0: group0, 1: group1}).reset_index().melt(id_vars='index', var_name=group_by, value_name=column)
    
#     # Create the violin plot
#     plt.figure(figsize=(8, 6))
#     sns.violinplot(x=group_by, y=column, data=plot_df, inner='quartile')
#     plt.title(f'Violin Plot of {column} Grouped by {group_by}\nMann-Whitney U Test p-value: {p_value:.4f}')
#     plt.show()

def plot_differential_violin(df, column, group_by):
    """
    Plots a violin plot showing the distribution of values in a given column, grouped by a categorical column.

    Parameters
    ----------
    df : pandas DataFrame
        The input DataFrame. 
    column : str
        The name of the column to plot.
    group_by : str
        The name of the column to group data by. This column should contain two groups, denoted by 0 and 1.

    Returns
    -------
    None. This function only plots a figure and does not return anything.
    """
    
    # Split the data into two groups based on the 'group_by' column
    group0 = df[df[group_by] == 0][column].dropna()
    group1 = df[df[group_by] == 1][column].dropna()
    
    # Perform Mann-Whitney U test to check if distributions of the two groups are significantly different
    _, p_value = mannwhitneyu(group0, group1)
    
    # Create a new DataFrame in a format suitable for seaborn's violinplot function
    plot_df = pd.DataFrame({0: group0, 1: group1}).reset_index().melt(id_vars='index', var_name=group_by, value_name=column)
    
    # Create the violin plot with inner quartiles
    plt.figure(figsize=(8, 6))
    sns.violinplot(x=group_by, y=column, data=plot_df, inner='quartile')

    # Set the title of the plot to include the column name, group_by column, and p-value of the Mann-Whitney U test
    plt.title(f'Violin Plot of {column} Grouped by {group_by}\nMann-Whitney U Test p-value: {p_value:.4f}')
    plt.show()
    

# def plot_survival_curves(df, column):
#     # Instantiate the KaplanMeierFitter object
#     kmf = KaplanMeierFitter()
    
#     # If the column is binary (0s and 1s only), use those values directly
#     if set(df[column].unique()) == {0, 1}:
#         group1 = df[df[column] == 0]
#         group2 = df[df[column] == 1]
#     # Otherwise, split the dataframe into two groups based on the median of the column
#     else:
#         median_val = df[column].median()
#         group1 = df[df[column] <= median_val]
#         group2 = df[df[column] > median_val]
    
#     # Generate the Kaplan-Meier survival estimate for each group
#     kmf.fit(group1['survival_day'], event_observed=group1['survival_event'], label=f'{column} low/0')
#     ax = kmf.plot()
    
#     kmf.fit(group2['survival_day'], event_observed=group2['survival_event'], label=f'{column} high/1')
#     kmf.plot(ax=ax)
    
#     # Perform the log-rank test
#     results = logrank_test(group1['survival_day'], group2['survival_day'], event_observed_A=group1['survival_event'], event_observed_B=group2['survival_event'])
#     print('Log-rank Test p-value:', results.p_value)


def plot_survival_curves(df, column):
    """
    This function plots Kaplan-Meier survival curves for two groups based on an input column in the DataFrame.
    If the column contains only binary values (0s and 1s), the function directly uses those values to split the data into two groups.
    Otherwise, the function splits the data into two groups based on the median value of the column.
    The log-rank test p-value is printed to provide information about the statistical significance of the difference in survival.

    Parameters
    ----------
    df : pandas DataFrame
        The input DataFrame containing survival data.
    column : str
        The name of the column used for grouping the survival data.

    Returns
    -------
    None. This function only plots the survival curves and prints the log-rank test p-value.

    """
    # Instantiate the KaplanMeierFitter object
    kmf = KaplanMeierFitter()

    # If the column is binary (0s and 1s only), use those values directly
    if set(df[column].unique()) == {0, 1}:
        group1 = df[df[column] == 0]
        group2 = df[df[column] == 1]
    # Otherwise, split the dataframe into two groups based on the median of the column
    else:
        median_val = df[column].median()
        group1 = df[df[column] <= median_val]
        group2 = df[df[column] > median_val]

    # Generate the Kaplan-Meier survival estimate for the first group
    kmf.fit(group1['survival_day'], event_observed=group1['survival_event'], label=f'{column} low/0')
    ax = kmf.plot()

    # Generate the Kaplan-Meier survival estimate for the second group and plot on the same axes
    kmf.fit(group2['survival_day'], event_observed=group2['survival_event'], label=f'{column} high/1')
    kmf.plot(ax=ax)

    # Perform the log-rank test to compare the survival distributions of the two groups
    results = logrank_test(group1['survival_day'], group2['survival_day'], event_observed_A=group1['survival_event'], event_observed_B=group2['survival_event'])
    print('Log-rank Test p-value:', results.p_value)

    
# def permute_and_predict(x_test, important_genes, model, reducer, n):
#     # Initialize dataframes for storing the average changes
#     class_diff_df = pd.DataFrame(index=x_test.index, columns=important_genes)
#     prob_diff_df = pd.DataFrame(index=x_test.index, columns=important_genes)
    
#     # Compute the base predictions
#     if isinstance(model, Sequential):
#         base_preds_prob = model.predict(reducer.transform(x_test))
#         base_preds_class = (base_preds_prob > 0.5).astype(int).flatten()
#     else:
#         base_preds_class = model.predict(reducer.transform(x_test))
#         base_preds_prob = model.predict_proba(reducer.transform(x_test))[:, 1]

#     # Loop over the important genes
#     for gene in tqdm(important_genes):
#         # Initialize lists for storing the differences
#         class_diff_list = []
#         prob_diff_list = []

#         # Loop over the number of permutations
#         for i in range(n):
#             # Permute the column
#             x_test_permuted = x_test.copy()
#             x_test_permuted[gene] = np.random.permutation(x_test_permuted[gene].values)

#             # Compute the predictions for the permuted data
#             if isinstance(model, Sequential):
#                 perm_preds_prob = model.predict(reducer.transform(x_test_permuted))
#                 perm_preds_class = (perm_preds_prob > 0.5).astype(int).flatten()
#             else:
#                 perm_preds_class = model.predict(reducer.transform(x_test_permuted))
#                 perm_preds_prob = model.predict_proba(reducer.transform(x_test_permuted))[:, 1]

#             # Compute the differences in predictions
#             class_diff = perm_preds_class - base_preds_class
#             prob_diff = perm_preds_prob - base_preds_prob

#             # Append the differences to the lists
#             class_diff_list.append(class_diff)
#             prob_diff_list.append(prob_diff)

#         # Compute the average differences and add them to the dataframes
#         class_diff_df[gene] = np.mean(np.array(class_diff_list), axis=0)
#         prob_diff_df[gene] = np.mean(np.array(prob_diff_list), axis=0)
        
#     def rescale(df, min_val, max_val):
#         result = df.copy()
#         for feature_name in df.columns:
#             max_value = df[feature_name].max()
#             min_value = df[feature_name].min()
#             if max_value == min_value:
#                 # Skip this column, or assign a default value.
#                 continue
#             result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
#             result[feature_name] = result[feature_name] * (max_val - min_val) + min_val
#         return result

#     class_diff_df.fillna(0, inplace=True)
#     prob_diff_df.fillna(0, inplace=True)

#     # find the global min and max
#     global_min = min(class_diff_df.min().min(), class_diff_df.min().min())
#     global_max = max(prob_diff_df.max().max(), prob_diff_df.max().max())

#     # rescale both dataframes
#     class_diff_df = rescale(class_diff_df, global_min, global_max)
#     prob_diff_df = rescale(prob_diff_df, global_min, global_max)

#     average_difference = class_diff_df.add(prob_diff_df)
#     similarity_matrix = average_difference.corr()
#     similarity_array = similarity_matrix.values
#     np.fill_diagonal(similarity_array, 0)
#     similarity_matrix = pd.DataFrame(similarity_array, columns=similarity_matrix.columns, index=similarity_matrix.index)

#     return similarity_matrix

def permute_and_predict(x_test, important_genes, model, reducer, n):
    """
    Performs permutation test by permuting the values of each important gene in the input data,
    computing predictions for permuted data, and comparing them with base predictions.
    Returns a similarity matrix based on the average differences in predictions.The similarity 
    matrix reflects the degree of similarity between important genes in terms of their impact 
    on predictions.

    Parameters
    ----------
    x_test : pandas DataFrame
        The input test data.
    important_genes : list
        List of genes considered important for prediction.
    model : object
        The prediction model.
    reducer : object
        Dimensionality reduction model.
    n : int
        The number of permutations to perform for each gene.

    Returns
    -------
    similarity_matrix : pandas DataFrame
        The similarity matrix based on average differences in predictions.
    """

    # Initialize dataframes for storing the average changes
    class_diff_df = pd.DataFrame(index=x_test.index, columns=important_genes)
    prob_diff_df = pd.DataFrame(index=x_test.index, columns=important_genes)

    # Compute the base predictions
    if isinstance(model, Sequential):
        base_preds_prob = model.predict(reducer.transform(x_test))
        base_preds_class = (base_preds_prob > 0.5).astype(int).flatten()
    else:
        base_preds_class = model.predict(reducer.transform(x_test))
        base_preds_prob = model.predict_proba(reducer.transform(x_test))[:, 1]

    # Loop over the important genes
    for gene in tqdm(important_genes, position=0, leave=True):
        # Initialize lists for storing the differences
        class_diff_list = []
        prob_diff_list = []

        # Loop over the number of permutations
        for i in range(n):
            # Permute the column
            x_test_permuted = x_test.copy()
            x_test_permuted[gene] = np.random.permutation(x_test_permuted[gene].values)

            # Compute the predictions for the permuted data
            if isinstance(model, Sequential):
                perm_preds_prob = model.predict(reducer.transform(x_test_permuted))
                perm_preds_class = (perm_preds_prob > 0.5).astype(int).flatten()
            else:
                perm_preds_class = model.predict(reducer.transform(x_test_permuted))
                perm_preds_prob = model.predict_proba(reducer.transform(x_test_permuted))[:, 1]

            # Compute the differences in predictions
            class_diff = perm_preds_class - base_preds_class
            prob_diff = perm_preds_prob - base_preds_prob

            # Append the differences to the lists
            class_diff_list.append(class_diff)
            prob_diff_list.append(prob_diff)

        # Compute the average differences and add them to the dataframes
        class_diff_df[gene] = np.mean(np.array(class_diff_list), axis=0)
        prob_diff_df[gene] = np.mean(np.array(prob_diff_list), axis=0)

    def rescale(df, min_val, max_val):
        # Function to rescale dataframe values between min_val and max_val
        result = df.copy()
        for feature_name in df.columns:
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
            if max_value == min_value:
                # Skip this column, or assign a default value.
                continue
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
            result[feature_name] = result[feature_name] * (max_val - min_val) + min_val
        return result

    class_diff_df.fillna(0, inplace=True)
    prob_diff_df.fillna(0, inplace=True)

    # find the global min and max
    global_min = min(class_diff_df.min().min(), class_diff_df.min().min())
    global_max = max(prob_diff_df.max().max(), prob_diff_df.max().max())

    # rescale both dataframes
    class_diff_df = rescale(class_diff_df, global_min, global_max)
    prob_diff_df = rescale(prob_diff_df, global_min, global_max)

    # Compute the average difference and construct the similarity matrix
    average_difference = class_diff_df.add(prob_diff_df)
    similarity_matrix = average_difference.corr()
    similarity_array = similarity_matrix.values
    np.fill_diagonal(similarity_array, 0)
    similarity_matrix = pd.DataFrame(similarity_array, columns=similarity_matrix.columns, index=similarity_matrix.index)

    return similarity_matrix


# def create_dendrogram_and_clustering_graph(similarity_matrix, max_clusters=10, clus_sel = 0):
#     # If there are negative numbers in the matrix, shift all values
#     min_val = similarity_matrix.min().min()
#     if min_val < 0:
#         similarity_matrix = similarity_matrix + abs(min_val)

#     # create a distance matrix
#     distances = 1 - similarity_matrix
    
#     # Convert DataFrame to NumPy array
#     arr = distances.to_numpy()

#     # Set diagonal values to 0
#     np.fill_diagonal(arr, 0)

#     # Convert back to DataFrame
#     distances = pd.DataFrame(arr, columns=distances.columns, index=distances.index)
    
#     # convert the redundant n*n square matrix form into a condensed nC2 array
#     distArray = squareform(distances)

#     # Calculate the linkage: mergings
#     mergings = linkage(distArray, method='complete')

#     silhouette_scores = []
#     for num_clusters in range(2, max_clusters+1):
#         labels = fcluster(mergings, num_clusters, criterion='maxclust')
#         score = silhouette_score(distances, labels, metric='euclidean')
#         silhouette_scores.append(score)
    
#     # Elbow method
#     optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
#     if clus_sel != 0:
#         optimal_clusters = clus_sel

#     plt.figure(figsize=(10, 5))
#     plt.plot(range(2, max_clusters+1), silhouette_scores, marker='o')
#     plt.title('Clustering success by number of clusters')
#     plt.xlabel('Number of clusters')
#     plt.ylabel('Silhouette score')
#     plt.vlines(optimal_clusters, min(silhouette_scores)-0.01, max(silhouette_scores)+0.01, colors='r', linestyles='dotted')
#     plt.grid()
#     plt.show()
    
#     # Determine the color threshold
#     labels = fcluster(mergings, optimal_clusters, criterion='maxclust')
#     color_threshold = mergings[-optimal_clusters, 2]

#     # Plot the dendrogram, using varieties as labels
#     plt.figure(figsize=(10, 5))
#     dendrogram(mergings,
#                labels = distances.columns,
#                leaf_rotation=90,
#                leaf_font_size=6,
#                color_threshold=color_threshold
#                )

#     plt.show()
    
#     # Draw the clustering graph (heatmap with dendrograms)
#     g = sns.clustermap(similarity_matrix, method='complete', cmap='viridis', row_linkage=mergings, col_linkage=mergings)
#     plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
#     plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90)
    
#     plt.show()



# def create_dendrogram_and_clustering_graph(similarity_matrix, max_clusters=10, clus_sel = 0):
#     # If there are negative numbers in the matrix, shift all values
#     min_val = similarity_matrix.min().min()
#     if min_val < 0:
#         similarity_matrix = similarity_matrix + abs(min_val)

#     # create a distance matrix
#     distances = 1 - similarity_matrix
    
#     # Convert DataFrame to NumPy array
#     arr = distances.to_numpy()

#     # Set diagonal values to 0
#     np.fill_diagonal(arr, 0)

#     # Convert back to DataFrame
#     distances = pd.DataFrame(arr, columns=distances.columns, index=distances.index)
    
#     # convert the redundant n*n square matrix form into a condensed nC2 array
#     distArray = squareform(distances)

#     # Calculate the linkage: mergings
#     mergings = linkage(distArray, method='complete')

#     silhouette_scores = []
#     for num_clusters in range(2, max_clusters+1):
#         labels = fcluster(mergings, num_clusters, criterion='maxclust')
#         score = silhouette_score(distances, labels, metric='euclidean')
#         silhouette_scores.append(score)
    
#     # Elbow method
#     optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
#     if clus_sel != 0:
#         optimal_clusters = clus_sel

#     plt.figure(figsize=(10, 5))
#     plt.plot(range(2, max_clusters+1), silhouette_scores, marker='o')
#     plt.title('Clustering success by number of clusters')
#     plt.xlabel('Number of clusters')
#     plt.ylabel('Silhouette score')
#     plt.vlines(optimal_clusters, min(silhouette_scores)-0.01, max(silhouette_scores)+0.01, colors='r', linestyles='dotted')
#     plt.grid()
#     plt.show()
    
#     # Determine the color threshold
#     labels = fcluster(mergings, optimal_clusters, criterion='maxclust')
#     color_threshold = mergings[-optimal_clusters, 2]

#     # Plot the dendrogram, using varieties as labels
#     plt.figure(figsize=(10, 5))
#     dendrogram(mergings,
#                labels = distances.columns,
#                leaf_rotation=90,
#                leaf_font_size=6,
#                color_threshold=color_threshold
#                )

#     plt.show()
    
#     # Draw the clustering graph (heatmap with dendrograms)
#     g = sns.clustermap(similarity_matrix, method='complete', cmap='viridis', row_linkage=mergings, col_linkage=mergings)
#     plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
#     plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90)
    
#     plt.show()

#     # Create a networkx graph object
#     G = nx.from_pandas_adjacency(similarity_matrix)
#     subgraphs = list(nx.connected_component_subgraphs(G))
    
#     return G, subgraphs


# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from scipy.spatial.distance import squareform
# from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
# from sklearn.metrics import silhouette_score
# import networkx as nx
# import community as community_louvain

# def create_dendrogram_and_clustering_graph(similarity_matrix, max_clusters=10, clus_sel = 0):
#     # If there are negative numbers in the matrix, shift all values
#     min_val = similarity_matrix.min().min()
#     if min_val < 0:
#         similarity_matrix = similarity_matrix + abs(min_val)

#     # Rescale values to range [0, 1]
#     max_val = similarity_matrix.max().max()
#     similarity_matrix = similarity_matrix / max_val
    
#     # create a distance matrix
#     distances = 1 - similarity_matrix
    
#     # Convert DataFrame to NumPy array
#     arr = distances.to_numpy()

#     # Set diagonal values to 0
#     np.fill_diagonal(arr, 0)

#     # Convert back to DataFrame
#     distances = pd.DataFrame(arr, columns=distances.columns, index=distances.index)
    
#     # convert the redundant n*n square matrix form into a condensed nC2 array
#     distArray = squareform(distances)

#     # Calculate the linkage: mergings
#     mergings = linkage(distArray, method='complete')

#     silhouette_scores = []
#     for num_clusters in range(2, max_clusters+1):
#         labels = fcluster(mergings, num_clusters, criterion='maxclust')
#         score = silhouette_score(distances, labels, metric='euclidean')
#         silhouette_scores.append(score)
    
#     # Elbow method
#     optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
#     if clus_sel != 0:
#         optimal_clusters = clus_sel

#     plt.figure(figsize=(10, 5))
#     plt.plot(range(2, max_clusters+1), silhouette_scores, marker='o')
#     plt.title('Clustering success by number of clusters')
#     plt.xlabel('Number of clusters')
#     plt.ylabel('Silhouette score')
#     plt.vlines(optimal_clusters, min(silhouette_scores)-0.01, max(silhouette_scores)+0.01, colors='r', linestyles='dotted')
#     plt.grid()
#     plt.show()
    
#     # Determine the color threshold
#     labels = fcluster(mergings, optimal_clusters, criterion='maxclust')
#     color_threshold = mergings[-optimal_clusters, 2]

#     # Plot the dendrogram, using varieties as labels
#     plt.figure(figsize=(10, 5))
#     dendrogram(mergings,
#                labels = distances.columns,
#                leaf_rotation=90,
#                leaf_font_size=6,
#                color_threshold=color_threshold
#                )

#     plt.show()
    
#     # Draw the clustering graph (heatmap with dendrograms)
#     g = sns.clustermap(similarity_matrix, method='complete', cmap='viridis', row_linkage=mergings, col_linkage=mergings)
#     plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
#     plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90)
    
#     plt.show()

#     # Create a networkx graph object
#     G = nx.from_pandas_adjacency(similarity_matrix)
    
#     # Create the subgraphs based on optimal clusters
#     for i, nodes in enumerate(nx.connected_components(G)):
#         subgraph = G.subgraph(nodes)
#         nx.set_node_attributes(subgraph, i, 'cluster')
        
#     # List of subgraphs
#     subgraphs = [G.subgraph(c) for c in nx.connected_components(G)]
    
#     return G, subgraphs

# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from scipy.spatial.distance import squareform
# from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
# from sklearn.metrics import silhouette_score
# import networkx as nx

# def create_dendrogram_and_clustering_graph(similarity_matrix, max_clusters=10, clus_sel = 0):
#     # If there are negative numbers in the matrix, shift all values
#     min_val = similarity_matrix.min().min()
#     if min_val < 0:
#         similarity_matrix = similarity_matrix + abs(min_val)
    
#     # Rescale values to range [0, 1]
#     max_val = similarity_matrix.max().max()
#     similarity_matrix = similarity_matrix / max_val
    
#     # create a distance matrix
#     distances = 1 - similarity_matrix
    
#     # Convert DataFrame to NumPy array
#     arr = distances.to_numpy()

#     # Set diagonal values to 0
#     np.fill_diagonal(arr, 0)

#     # Convert back to DataFrame
#     distances = pd.DataFrame(arr, columns=distances.columns, index=distances.index)
    
#     # convert the redundant n*n square matrix form into a condensed nC2 array
#     distArray = squareform(distances)

#     # Calculate the linkage: mergings
#     mergings = linkage(distArray, method='complete')

#     silhouette_scores = []
#     for num_clusters in range(2, max_clusters+1):
#         labels = fcluster(mergings, num_clusters, criterion='maxclust')
#         score = silhouette_score(distances, labels, metric='euclidean')
#         silhouette_scores.append(score)
    
#     # Elbow method
#     optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
#     if clus_sel != 0:
#         optimal_clusters = clus_sel

#     plt.figure(figsize=(10, 5))
#     plt.plot(range(2, max_clusters+1), silhouette_scores, marker='o')
#     plt.title('Clustering success by number of clusters')
#     plt.xlabel('Number of clusters')
#     plt.ylabel('Silhouette score')
#     plt.vlines(optimal_clusters, min(silhouette_scores)-0.01, max(silhouette_scores)+0.01, colors='r', linestyles='dotted')
#     plt.grid()
#     plt.show()
    
#     # Determine the color threshold
#     labels = fcluster(mergings, optimal_clusters, criterion='maxclust')
#     color_threshold = mergings[-optimal_clusters, 2]

#     # Plot the dendrogram, using varieties as labels
#     plt.figure(figsize=(10, 5))
#     dendrogram(mergings,
#                labels = distances.columns,
#                leaf_rotation=90,
#                leaf_font_size=6,
#                color_threshold=color_threshold
#                )

#     plt.show()
    
#     # Draw the clustering graph (heatmap with dendrograms)
#     g = sns.clustermap(similarity_matrix, method='complete', cmap='viridis', row_linkage=mergings, col_linkage=mergings)
#     plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
#     plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90)
    
#     plt.show()

#     # Create a networkx graph object
#     G = nx.from_pandas_adjacency(similarity_matrix)
#     G.remove_edges_from(nx.selfloop_edges(G))

#     # Assign cluster label to each node
#     for i, node in enumerate(G.nodes()):
#         G.nodes[node]['cluster'] = labels[i]
    
#     # Create subgraphs based on the cluster labels
#     subgraphs = []
#     for i in range(1, optimal_clusters+1):
#         nodes = [node for node, data in G.nodes(data=True) if data['cluster'] == i]
#         subgraphs.append(G.subgraph(nodes))
    
#     return G, subgraphs

def create_dendrogram_and_clustering_graph(similarity_matrix, max_clusters=10, clus_sel=0):
    """
    This function creates a dendrogram and clustering graph based on the similarity matrix.
    It performs hierarchical clustering on the distances derived from the similarity matrix,
    and determines the optimal number of clusters using the silhouette score.
    The function plots the silhouette scores and the dendrogram, and creates a clustering graph
    with subgraphs representing the clusters.
    
    The function returns the networkx Graph object `G` representing the similarity matrix
    and a list of subgraphs `subgraphs` representing the clusters in the graph.

    Parameters
    ----------
    similarity_matrix : pandas DataFrame
        The similarity matrix based on average differences in predictions.
    max_clusters : int, optional
        The maximum number of clusters to consider for clustering, by default 10.
    clus_sel : int, optional
        The number of clusters to select instead of using the optimal value, by default 0.

    Returns
    -------
    G : networkx Graph object
        The graph object representing the similarity matrix.
    subgraphs : list
        A list of subgraphs representing the clusters in the graph.
    """

    # If there are negative numbers in the matrix, shift all values
    min_val = similarity_matrix.min().min()
    if min_val < 0:
        similarity_matrix = similarity_matrix + abs(min_val)

    # Rescale values to range [0, 1]
    max_val = similarity_matrix.max().max()
    similarity_matrix = similarity_matrix / max_val

    # Create a distance matrix
    distances = 1 - similarity_matrix

    # Convert DataFrame to NumPy array
    arr = distances.to_numpy()

    # Set diagonal values to 0
    np.fill_diagonal(arr, 0)

    # Convert back to DataFrame
    distances = pd.DataFrame(arr, columns=distances.columns, index=distances.index)

    # Convert the redundant n*n square matrix form into a condensed nC2 array
    distArray = squareform(distances)

    # Calculate the linkage: mergings
    mergings = linkage(distArray, method='complete')

    silhouette_scores = []
    for num_clusters in range(2, max_clusters + 1):
        labels = fcluster(mergings, num_clusters, criterion='maxclust')
        score = silhouette_score(distances, labels, metric='euclidean')
        silhouette_scores.append(score)

    # Elbow method
    optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
    if clus_sel != 0:
        optimal_clusters = clus_sel

    # Plot the silhouette scores
    plt.figure(figsize=(10, 5))
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
    plt.title('Clustering success by number of clusters')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette score')
    plt.vlines(optimal_clusters, min(silhouette_scores) - 0.01, max(silhouette_scores) + 0.01, colors='r', linestyles='dotted')
    plt.grid()
    plt.show()

    # Determine the color threshold
    labels = fcluster(mergings, optimal_clusters, criterion='maxclust')
    color_threshold = mergings[-optimal_clusters, 2]

    # Plot the dendrogram
    plt.figure(figsize=(10, 5))
    dendrogram(mergings,
               labels=distances.columns,
               leaf_rotation=90,
               leaf_font_size=6,
               color_threshold=color_threshold
               )
    plt.show()

    # Plot the clustering graph
    g = sns.clustermap(similarity_matrix, method='complete', cmap='viridis', row_linkage=mergings, col_linkage=mergings)
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90)
    plt.show()

    # Create a networkx graph object
    G = nx.from_pandas_adjacency(similarity_matrix)
    G.remove_edges_from(nx.selfloop_edges(G))

    # Assign cluster label to each node
    for i, node in enumerate(G.nodes()):
        G.nodes[node]['cluster'] = labels[i]

    # Create subgraphs based on the cluster labels
    subgraphs = []
    for i in range(1, optimal_clusters + 1):
        nodes = [node for node, data in G.nodes(data=True) if data['cluster'] == i]
        subgraphs.append(G.subgraph(nodes))

    return G, subgraphs

def plot_subgraphs(G, subgraphs):
    """
    Plots the subgraphs representing clusters.

    Parameters
    ----------
    G : networkx Graph object
        The graph object representing the similarity matrix.
    subgraphs : list
        A list of subgraphs representing the clusters in the graph.

    Returns
    -------
    None. This function only plots the subgraphs and does not return anything.

    """
    for i, sg in enumerate(subgraphs):
        # Create a new figure for each subgraph
        plt.figure()
        
        # Get the cluster labels for each node in the subgraph
        colors = [nx.get_node_attributes(G, 'cluster')[node] for node in sg.nodes()]
        
        # Draw the subgraph with labels and specified node color and color map
        nx.draw(sg, with_labels=True, node_color='lightblue', cmap='viridis')
        
        # Set the title of the plot to include the subgraph index
        plt.title(f'Subgraph {i+1}')
        
        # Show the plot
        plt.show()

