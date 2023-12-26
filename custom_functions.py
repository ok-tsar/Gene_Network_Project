
# Basic Libraries
import os
import csv
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import pickle
import joblib
from lifelines.statistics import logrank_test

# Machine Learning and Preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA, FactorAnalysis, TruncatedSVD, KernelPCA, FastICA, NMF, SparsePCA
from sklearn.manifold import TSNE, MDS, Isomap
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (classification_report, accuracy_score, plot_confusion_matrix, 
                             roc_curve, auc, precision_recall_curve, average_precision_score, 
                             f1_score, recall_score, precision_score, r2_score, confusion_matrix, 
                             silhouette_score, roc_auc_score)
from sklearn.utils import shuffle
from sklearn import decomposition, manifold, metrics

# Keras and TensorFlow
import keras
from tensorflow.keras.models import Sequential, load_model, clone_model

# Statistical Analysis
from scipy import interp
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.stats import mannwhitneyu

# Network Analysis
import networkx as nx

# Survival Analysis
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

# Others
from itertools import permutations
from tqdm import tqdm
from umap import UMAP
from joblib import Parallel, delayed
import math
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
from pandas import DataFrame
import numpy as np
from scipy.stats import binom

import os
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import load_model
import os
import pickle
import shap
from sklearn.linear_model import LinearRegression, LogisticRegression
from tensorflow.keras.models import load_model
import os
import pickle
import pandas as pd
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from keras.models import load_model
from tqdm import tqdm  # Import tqdm
from alibi.explainers import ALE
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tqdm import tqdm
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import AUC
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from scipy.stats import mannwhitneyu
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
import math

import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers
from matplotlib.patches import Patch

from itertools import product
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import (KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering, 
                             OPTICS, AffinityPropagation)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from tqdm import tqdm

# ... [other imports]
from scipy.spatial.distance import euclidean

import numpy as np
from scipy.spatial.distance import euclidean
import numpy as np
from scipy.spatial.distance import euclidean, cdist

# ------------------------------------------------------------------------
# ---------------- 1-modelTraining_geneRanking FUNCTIONS -----------------
# ------------------------------------------------------------------------

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

def split_csv(file_path, output_folder, max_size=15):
    """
    Split a CSV file into multiple smaller CSV files.

    Parameters
    ----------
    file_path : str
        The file path of the large CSV file.
    output_folder : str
        The folder where the split CSV files will be stored.
    max_size : int
        Maximum size of each split file in MB.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    chunk_size = max_size * (1024**2)  # Convert MB to bytes
    chunk_number = 1
    current_size = 0
    header = True

    with open(file_path, 'r', newline='') as f:
        reader = csv.reader(f)
        header_row = next(reader)

        for row in reader:
            if current_size == 0:
                output_file = open(os.path.join(output_folder, f'chunk_{chunk_number}.csv'), 'w', newline='')
                writer = csv.writer(output_file)
                if header:
                    writer.writerow(header_row)
                    header = False

            writer.writerow(row)
            current_size += sum(len(str(field)) for field in row)

            if current_size >= chunk_size:
                output_file.close()
                chunk_number += 1
                current_size = 0
                header = True

    if not output_file.closed:
        output_file.close()

    print(f"Data split into {chunk_number} chunks in the folder '{output_folder}'.")

def merge_load_csv(folder_path):
    """
    Merge multiple CSV files into a single DataFrame.

    Parameters
    ----------
    folder_path : str
        The folder path containing the CSV files.

    Returns
    -------
    df : pandas DataFrame
        The merged DataFrame.
    """
    df = pd.DataFrame()
    file_paths = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')])

    for file_path in file_paths:
        temp_df = pd.read_csv(file_path, index_col=0)
        df = pd.concat([df, temp_df])

    df.columns = pd.MultiIndex.from_tuples([(col,) for col in df.columns])

    print("data shape:", df.shape)
    return df

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
    
def plot_feature_density(dataframe, feature):
    """
    Plots the density of a specified feature within a dataframe for two different output classes.

    Parameters
    ----------
    dataframe : pandas DataFrame
        The dataframe containing the feature and 'output' columns.

    feature : str
        The name of the feature to plot.

    Returns
    -------
    None. Plots the densities.
    """
    # Normalize feature name by removing '.rescaled' if present
    normalized_feature = feature.replace('.rescaled', '')
    feature_with_suffix = normalized_feature + '.rescaled'
    
    # Check if the feature and 'output' are present in the dataframe columns
    if feature_with_suffix in dataframe.columns and 'output' in dataframe.columns:
        plt.figure(figsize=(10, 6))

        # Plot histogram for output=0
        sns.histplot(dataframe[dataframe[('output',)] == 0][(feature_with_suffix,)], color='blue', label='Output=0', kde=False, stat='density')
        
        # Plot histogram for output=1
        sns.histplot(dataframe[dataframe[('output',)] == 1][(feature_with_suffix,)], color='red', label='Output=1', kde=False, stat='density')

        # Plot labels and title
        plt.title(f'Density of {normalized_feature} by Output')
        plt.xlabel(f'{normalized_feature} Expression')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        plt.tight_layout()
        plt.show()
    else:
        # If the required columns are not present, print a debug statement
        print("The required columns are not present in the dataframe.")
        if feature_with_suffix not in dataframe.columns:
            print(f"Missing feature column: {feature_with_suffix}")
        if 'output' not in dataframe.columns:
            print("Missing 'output' column.")

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

def print_model_summary(model_path):
    """
    Prints model from model path.

    Parameters
    ----------
    model_path : path to model

    Returns
    -------
    None. prints model summary.
    """
    
    # Load the model from the given path
    model = load_model(model_path)
    
    # Print the model summary
    model.summary()

def get_feature_importance(x_test, y_test, reducer, model_path, location, N):
    """
    This function evaluates feature importance by permuting the features of a test dataset and
    comparing the model's performance with the permuted data to its performance with the original data.
    The difference in performance serves as an indication of the feature's importance.

    Parameters
    ----------
    x_test : DataFrame
        The test data features.

    y_test : Series
        The test data target.

    reducer : PCA or other dimensionality reduction object, optional
        Dimensionality reducer to transform data before prediction.

    model_path : str
        Path to the model file for prediction.

    location : str
        Directory to save output files.

    N : int
        Number of permutations for each feature.

    Outputs
    -------
    Plots and Pickle Files:
        - Plots of accuracy and R-squared differences by feature.
        - Pickle files of accuracy and R-squared differences.
    """

    # Load the model
    model = keras.models.load_model(model_path)

    # Convert DataFrame to NumPy array
    x_test_arr = x_test.values

    # Calculate accuracy and R-squared of the model on the original data
    x_test_transformed = reducer.transform(x_test_arr) if reducer is not None else x_test_arr
    y_pred_orig = np.round(model.predict(x_test_transformed))
    y_pred_prob_orig = model.predict(x_test_transformed)
    acc_orig = accuracy_score(y_test, y_pred_orig)
    r2_orig = r2_score(y_test, y_pred_prob_orig)

    # Initialize arrays to store accuracy and R-squared differences
    acc_diffs = np.zeros(x_test.shape[1])
    r2_diffs = np.zeros(x_test.shape[1])

    # Define function for calculating differences for a single column
    def calculate_diffs(col):
        # Load model here
        model = keras.models.load_model(model_path)

        acc_diff_sum = 0.0
        r2_diff_sum = 0.0

        # Permute the current column N times and calculate accuracy and R-squared differences
        for _ in range(N):
            x_test_permuted = x_test_arr.copy()
            x_test_permuted[:, col] = shuffle(x_test_permuted[:, col])
            x_test_permuted_transformed = (
                reducer.transform(x_test_permuted) if reducer is not None else x_test_permuted
            )
            y_pred_permuted = np.round(model.predict(x_test_permuted_transformed))
            y_pred_prob_permuted = model.predict(x_test_permuted_transformed)
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
        
    # Ensure the directory exists
    if not os.path.exists(location):
        os.makedirs(location)

    # Function to get unique file name
    def get_unique_filename(base_filename):
        counter = 1
        while os.path.exists(os.path.join(location, f"{base_filename}_{counter}.png")):
            counter += 1
        return f"{base_filename}_{counter}"

    # Plot and save accuracy differences
    unique_acc_diff_name = get_unique_filename("accuracy_diff")
    unique_r2_diff_name = get_unique_filename("r2_diff")

    plt.figure(figsize=(8, 6))
    plt.plot(acc_diffs)
    plt.xlabel('Feature Index')
    plt.ylabel('Accuracy Difference')
    plt.title('Accuracy Difference by Feature Permutation')
    plt.savefig(os.path.join(location, f"{unique_acc_diff_name}.png"))
    plt.show()

    # Plot and save R-squared differences
    plt.figure(figsize=(8, 6))
    plt.plot(r2_diffs)
    plt.xlabel('Feature Index')
    plt.ylabel('R-squared Difference')
    plt.title('R-squared Difference by Feature Permutation')
    plt.savefig(os.path.join(location, f"{unique_r2_diff_name}.png"))
    plt.show()

    # Save accuracy and R-squared differences
    with open(os.path.join(location, f"{unique_acc_diff_name}.pkl"), 'wb') as fp:
        pickle.dump(acc_diffs, fp)
    with open(os.path.join(location, f"{unique_r2_diff_name}.pkl"), 'wb') as fp:
        pickle.dump(r2_diffs, fp)

def fetch_importances(data, importance_location):
    """
    Loads, ranks, and normalizes feature importances saved as joblib files in a specific location.

    Parameters
    ----------
    data : pandas DataFrame
        The input data where each column corresponds to a feature.

    importance_location : str
        The directory where the joblib files of feature importances are stored.

    Returns
    -------
    sorted_importances : pandas DataFrame
        A DataFrame where each row represents a feature and each column corresponds to the rank of that feature's importance.
    """
    
    # Generate a list of features, excluding the first non-informative column
    features = [l[0] for l in data.columns.tolist()][1:]

    # Initialize DataFrame to store importances
    importances = pd.DataFrame()

    # Loop through each file in the specified directory
    for filename in os.listdir(importance_location):
        # Check if the file is a pickle file before loading
        if filename.endswith(".pkl") and os.path.isfile(os.path.join(importance_location, filename)):
            file_path = os.path.join(importance_location, filename)
            
            # Use pickle to load the data
            try:
                with open(file_path, 'rb') as f:
                    pickle_data = pickle.load(f)
                importances[filename] = pickle_data
            except Exception as e:
                print(f"Error loading file: {file_path}. Reason: {e}")

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

def log_factorial(n):
    return sum(math.log(i) for i in range(1, n+1))

def log_binomial_coeff(n, k):
    return log_factorial(n) - log_factorial(k) - log_factorial(n-k)

def log_probability_of_overlap_two_groups(k, top_n, total_features):
    p_single = top_n / total_features
    return log_binomial_coeff(total_features, k) + (2 * k * math.log(p_single)) + (2 * (total_features - k) * math.log(1 - p_single))

def prob_of_at_least_k(total_features, p_overlap, k):
    # Calculate the probability of getting i overlaps for each i in 0 to total_features
    probs = [binom.pmf(i, total_features, p_overlap) for i in range(total_features+1)]
    
    # Convolve the probabilities to account for three sets
    for _ in range(2):
        probs = np.convolve(probs, probs)[:total_features+1]
    
    # Sum probabilities from k onward
    return sum(probs[k:])

def plot_top_features_venn(importance_list, top_n=100, total_features=None, show_annotations=True):
    """
    Plots a Venn diagram for the top features from multiple importance dataframes.

    Parameters
    ----------
    importance_list : list of DataFrames
        The list of DataFrames with feature importances.

    top_n : int, default=100
        Number of top features to consider.

    total_features : int, optional
        Total number of features, used for calculating expected overlap.

    show_annotations : bool, default=True
        Whether to show feature names in annotations.

    Returns
    -------
    None. Plots the Venn diagram.
    """
    
    # Extract top feature names, and remove '.rescaled' from names
    cnn_features = set(importance_list[0].head(top_n).index.str.replace('.rescaled', '', regex=False))
    mlp_features = set(importance_list[1].head(top_n).index.str.replace('.rescaled', '', regex=False))
    log_features = set(importance_list[2].head(top_n).index.str.replace('.rescaled', '', regex=False))

    # Identify shared features
    shared_all = cnn_features.intersection(mlp_features, log_features)
    print(shared_all)
    shared_mlp_cnn = mlp_features.intersection(cnn_features) - shared_all
    shared_mlp_log = mlp_features.intersection(log_features) - shared_all
    shared_cnn_log = cnn_features.intersection(log_features) - shared_all

    total_overlap = len(shared_mlp_cnn) + len(shared_mlp_log) + len(shared_cnn_log) + len(shared_all)

    if total_features:
        # Expected overlap for any two groups
        p_single = top_n / total_features
        expected_overlap_two_groups = (p_single ** 2) * total_features
        expected_total_overlap = 3 * expected_overlap_two_groups

        p_overlap = p_single ** 2
        probability = prob_of_at_least_k(total_features, p_overlap, total_overlap)

        print(f"Expected total overlap (any two of three groups): {expected_total_overlap:.2f}")
        print(f"Probability of observing at least {total_overlap} overlaps: {probability:.7f}")

    # Create Venn diagram
    plt.figure(figsize=(12, 10))
    v = venn3(subsets=(cnn_features, mlp_features, log_features), 
              set_labels=('CNN', 'MLP', 'LOG'))

    if show_annotations:
        # Annotate with shared features in specified positions
        plt.annotate('\n'.join([f'* {f} *' if f in shared_all else f for f in shared_mlp_cnn]), xy=(-0.01, 0.55), ha='center', va='top',
                     fontsize=10, bbox=dict(boxstyle='round,pad=0.5', fc='gray', alpha=0.1))
        
        plt.annotate('\n'.join([f'* {f} *' if f in shared_all else f for f in shared_cnn_log]), xy=(-0.4, -0.32), ha='center', va='bottom',
                     fontsize=10, bbox=dict(boxstyle='round,pad=0.5', fc='gray', alpha=0.1))
        
        plt.annotate('\n'.join([f'* {f} *' if f in shared_all else f for f in shared_mlp_log]), xy=(0.5, -0.45), ha='center', va='bottom',
                     fontsize=10, bbox=dict(boxstyle='round,pad=0.5', fc='gray', alpha=0.1))

    plt.title(f"Top {top_n} Features Venn Diagram with Shared Features", fontsize=14)
    plt.show()


# ------------------------------------------------------------------------
# ---------------- 2-cuttoffModeling_biologicalRelevence -----------------
# ------------------------------------------------------------------------

def surrogate_model_importance(x_test, pca_reducer, model_location,  save_location=None):
    """
    Extract feature importances using a surrogate model based on predictions of a primary model.
    
    Parameters
    ----------
    x_test : DataFrame
        Test features.
    pca_reducer : PCA object
        Fitted PCA object for transforming the data.
    model_location : str
        Path to the trained model.
    save_location : str, optional
        Directory to save the feature importances.
    
    Returns
    -------
    feature_importances_df : DataFrame
        DataFrame of feature importances with gene names as the index.
    """
    
    # Transform the data using PCA
    x_test_transformed = pca_reducer.transform(x_test)
    
    # Load the trained model
    model = load_model(model_location)
    
    # Predict using the loaded model
    y_pred = model.predict(x_test_transformed).flatten()
    
    # Convert predictions to binary (assuming binary classification)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Fit a surrogate model (Logistic Regression) on the original features using binary predictions as target
    surrogate = LogisticRegression(max_iter=1000)
    surrogate.fit(x_test, y_pred_binary)
    
    # Extract feature importances from the surrogate model
    feature_importances = surrogate.coef_[0]
    
    # Create a dataframe with gene names as index and importances as the column
    feature_importances_df = pd.DataFrame({
        'feature_importance': feature_importances
    }, index=x_test.columns)
    
    # Sort by absolute value but preserve the sign
    feature_importances_df['absolute_importance'] = feature_importances_df['feature_importance'].abs()
    feature_importances_df = feature_importances_df.sort_values(by='absolute_importance', ascending=False).drop(columns='absolute_importance')

    # Save the dataframe using pickle only if save_location is provided
    if save_location:
        if not os.path.exists(save_location):
            os.makedirs(save_location)
        unique_file_name = get_unique_filename("feature_importances")
        with open(os.path.join(save_location, f"{unique_file_name}.pkl"), 'wb') as fp:
            pickle.dump(feature_importances_df, fp)
    
    return feature_importances_df

def surrogate_shap_importance(x_test, pca_reducer, model_location,  save_location=None):
    """
    Extract feature importances using SHAP values on a surrogate model's predictions.
    
    Parameters
    ----------
    x_test : DataFrame
        Test features.
    pca_reducer : PCA object
        Fitted PCA object for transforming the data.
    model_location : str
        Path to the trained model.
    save_location : str, optional
        Directory to save the feature importances.
    
    Returns
    -------
    shap_values_df : DataFrame
        DataFrame of SHAP values with gene names as the index.
    """
    
    # Transform the data using PCA
    x_test_transformed = pca_reducer.transform(x_test)
    
    # Load the primary model
    model = load_model(model_location)
    
    # Predict using the primary model
    y_pred = model.predict(x_test_transformed).flatten()
    
    # Convert predictions to binary (assuming binary classification)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Train a surrogate model on the original features
    surrogate = LogisticRegression(max_iter=1000)
    surrogate.fit(x_test, y_pred_binary)
    
    # Compute SHAP values using the surrogate model
    explainer = shap.Explainer(surrogate, x_test)
    shap_values = explainer(x_test)
    
    # Create a dataframe with gene names as index and mean absolute SHAP values as the column
    shap_values_df = pd.DataFrame({
        'shap_value': shap_values.values.mean(axis=0)
    }, index=x_test.columns)
    
    # Sort by absolute SHAP value
    shap_values_df = shap_values_df.sort_values(by='shap_value', key=abs, ascending=False)

    # Save the dataframe using pickle only if save_location is provided
    if save_location:
        if not os.path.exists(save_location):
            os.makedirs(save_location)
        unique_file_name = get_unique_filename("feature_importances")
        with open(os.path.join(save_location, f"{unique_file_name}.pkl"), 'wb') as fp:
            pickle.dump(shap_values_df, fp)
    
    return shap_values_df


def inverse_mapping_shap(x_test, kpca_reducer, model_location, save_location=None):
    """
    Compute SHAP values for the original features by mapping from the KernelPCA reduced space.
    
    Parameters
    ----------
    x_test : DataFrame
        Test features.
    kpca_reducer : KernelPCA object
        Fitted KernelPCA object for transforming the data.
    model_location : str
        Path to the trained model.
    save_location : str, optional
        Directory to save the SHAP values.
    
    Returns
    -------
    shap_values_df : DataFrame
        DataFrame of SHAP values with gene names as the index.
    """

    # Load the saved model
    model = load_model(model_location)

    # Transform the data using KernelPCA
    x_test_transformed = kpca_reducer.transform(x_test)

    # Compute SHAP values using the model on KernelPCA-reduced data
    background_data = shap.sample(x_test_transformed, 100)  # Taking a random sample as the background
    explainer = shap.Explainer(model, background_data)

    shap_values_reduced = explainer(x_test_transformed).values

    # Map SHAP values from reduced space to original space using surrogate models
    num_features_original = x_test.shape[1]
    shap_values_original = np.zeros((x_test.shape[0], num_features_original))

    for i in range(x_test_transformed.shape[1]):
        regressor = LinearRegression().fit(x_test, x_test_transformed[:, i])
        coeffs = regressor.coef_
        for j in range(num_features_original):
            shap_values_original[:, j] += shap_values_reduced[:, i] * coeffs[j]

    # Convert SHAP values to DataFrame
    shap_values_df = pd.DataFrame(shap_values_original, columns=x_test.columns)

    # Calculate the mean SHAP value for each feature across all samples and sort by absolute value
    avg_shap_values = shap_values_df.mean()
    avg_shap_values_sorted_by_abs = avg_shap_values.abs().sort_values(ascending=False)
    avg_shap_values = avg_shap_values.loc[avg_shap_values_sorted_by_abs.index]

    # Reshape to column format
    avg_shap_values_df = avg_shap_values.to_frame(name="SHAP Value")

    # Save the dataframe using pickle only if save_location is provided
    if save_location:
        if not os.path.exists(save_location):
            os.makedirs(save_location)
        unique_file_name = get_unique_filename("feature_importances")
        with open(os.path.join(save_location, f"{unique_file_name}.pkl"), 'wb') as fp:
            pickle.dump(avg_shap_values_df, fp)

    return avg_shap_values_df

def lime_importance(x_test, x_train, y_train, pca_reducer, model_location, is_regression, save_location=None):
    """
    Compute LIME importances for the original features by mapping from the PCA reduced space.
    
    Parameters
    ----------
    x_test : DataFrame
        Test features.
    x_train : DataFrame
        Train features.
    y_train : Array-like
        Train labels.
    pca_reducer : PCA object
        Fitted PCA object for transforming the data.
    model_location : str
        Path to the trained model.
    is_regression : bool
        Specifies whether the model is a regression model.
    save_location : str, optional
        Directory to save the LIME importances.
    
    Returns
    -------
    aggregated_importances_df : DataFrame
        DataFrame of aggregated LIME importances with gene names as the index.
    """

    model = load_model(model_location)
    
    def predict_fn(x):
        x_transformed = pca_reducer.transform(x)
        predictions = model.predict(x_transformed)
        if is_regression:
            return predictions
        else:
            original_predictions = predictions[:, 0]
            predictions_b = 1 - original_predictions
            final_predictions = np.column_stack((original_predictions, predictions_b))
            return final_predictions

    explainer = LimeTabularExplainer(
        training_data=x_train.to_numpy(),
        training_labels=y_train,
        feature_names=x_train.columns.tolist(),
        class_names=['0', '1'],
        mode='regression' if is_regression else 'classification',
        discretize_continuous=False
    )

    lime_values_matrix = np.zeros(x_test.shape)
    
    # Use tqdm to add a progress bar
    for i, instance in tqdm(enumerate(x_test.to_numpy()), total=x_test.shape[0], desc="Explaining"):
        exp = explainer.explain_instance(instance, predict_fn)
        instance_importances = dict(exp.as_list())
        for feature, importance in instance_importances.items():
            col_index = x_test.columns.get_loc(feature)
            lime_values_matrix[i, col_index] = importance
    
    avg_lime_importances = np.mean(lime_values_matrix, axis=0)
    aggregated_importances_df = pd.DataFrame({
        'LIME Importance': avg_lime_importances
    }, index=x_test.columns)
    
    # Order by absolute value but keep the sign of the importance
    aggregated_importances_df['abs_importance'] = aggregated_importances_df['LIME Importance'].abs()
    aggregated_importances_df = aggregated_importances_df.sort_values(by='abs_importance', ascending=False).drop('abs_importance', axis=1)
    
    if save_location:
        aggregated_importances_df.to_csv(save_location)
    
    return aggregated_importances_df

def ale_feature_importance(x_test, pca_reducer, model_path):
    """
    Estimate ALE on original features by perturbing one feature at a time.
    
    Parameters
    ----------
    x_test : DataFrame
        Original test features.
    pca_reducer : PCA object
        PCA transformation model.
    model_path : str
        Path to the saved model.
    
    Returns
    -------
    ale_df : DataFrame
        A DataFrame with ALE values for each original feature.
    """

    # Load the model from the provided path
    model = load_model(model_path)
    
    # Transform the original dataset with PCA once
    x_test_transformed = pca_reducer.transform(x_test)
    # Predict on the original dataset once
    preds_original = model.predict(x_test_transformed)
    
    ale_values = {}
    
    for feature in tqdm(x_test.columns, total=len(x_test.columns), desc="Explaining"):
        # Define a small perturbation value
        perturbation = 0.01 * (x_test[feature].max() - x_test[feature].min())
        
        # Create a perturbed version of the dataset
        x_test_perturbed = x_test.copy()
        x_test_perturbed[feature] += perturbation
        
        # Transform the perturbed dataset
        x_test_perturbed_transformed = pca_reducer.transform(x_test_perturbed)
        
        # Make predictions for the perturbed dataset
        preds_perturbed = model.predict(x_test_perturbed_transformed)
        
        # Compute the ALE as the mean squared error between original and perturbed predictions
        ale_value = mean_squared_error(preds_original, preds_perturbed)
        
        ale_values[feature] = ale_value

    # Convert dictionary to DataFrame for returning
    ale_df = pd.DataFrame.from_dict(ale_values, orient='index', columns=['ALE Value']).sort_values(by='ALE Value', ascending=False)
    ale_df.index = ale_df.index.str[0]

    return ale_df

def save_dataframe_to_txt(df, filename):
    """
    Save the provided DataFrame to a text file.
    
    Parameters
    ----------
    df : DataFrame
        The DataFrame to be saved.
    filename : str
        The name (including location) of the file where the DataFrame should be saved.
    """
    df.to_csv(filename, sep='\t', header=True, index=True)


def load_dataframe_from_txt(filename):
    """
    Load a DataFrame from a text file.
    
    Parameters
    ----------
    filename : str
        The name (including location) of the file from which the DataFrame should be loaded.
    
    Returns
    -------
    DataFrame
        The loaded DataFrame.
    """
    df = pd.read_csv(filename, delimiter="\t", index_col=0)
    return df


def model_performance_analysis(x_train, y_train, x_test, y_test, model_type, 
                               importance_dataframe, min_features, max_features, 
                               sim_number=10):
    """
    Analyze the performance of a model based on varying number of features.
    
    Parameters
    ----------
    x_train : DataFrame or numpy array
        Training features.
    y_train : DataFrame or numpy array
        Training labels.
    x_test : DataFrame or numpy array
        Testing features.
    y_test : DataFrame or numpy array
        Testing labels.
    model_type : str
        Type of the model ('cnn' or others).
    importance_dataframe : DataFrame
        DataFrame containing feature importances.
    min_features : int
        Minimum number of features to evaluate.
    max_features : int
        Maximum number of features to evaluate.
    sim_number : int, optional
        Number of simulations to run.
    
    Returns
    -------
    mean_aurocs : numpy array
        Mean AUROC scores across simulations.
    mean_auprcs : numpy array
        Mean AUPRC scores across simulations.
    """
    all_aurocs = []
    all_auprcs = []

    # Run multiple simulations
    for _ in tqdm(range(sim_number), desc="Simulation Progress"):
        avg_aurocs = []
        avg_auprcs = []

        # Loop over different numbers of features
        for num_features in range(min_features, max_features+1):
            # Select top features based on their importance
            selected_features = importance_dataframe.index.tolist()[:num_features]
            x_train_selected = x_train[selected_features]
            x_test_selected = x_test[selected_features]

            # Adjust input shape for CNN if specified
            if model_type == 'cnn':
                sqrt_p = math.ceil(math.sqrt(num_features))
                padded_dim = sqrt_p**2
                padding_size = padded_dim - num_features
                pad_values_train = np.ones((x_train_selected.shape[0], padding_size))
                pad_values_test = np.ones((x_test_selected.shape[0], padding_size))
                x_train_selected = np.hstack([x_train_selected, pad_values_train])
                x_test_selected = np.hstack([x_test_selected, pad_values_test])

            # Create, compile and train the model
            model = create_model(model_type, x_train_selected.shape[1])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[AUC(name='auc')])
            model.fit(x_train_selected, y_train, epochs=10, verbose=0)
            y_pred = model.predict(x_test_selected)

            # Store performance metrics
            avg_aurocs.append(roc_auc_score(y_test, y_pred))
            avg_auprcs.append(average_precision_score(y_test, y_pred))
            
        all_aurocs.append(avg_aurocs)
        all_auprcs.append(avg_auprcs)

     # Calculate mean performance metrics across all simulations
    mean_aurocs = np.mean(all_aurocs, axis=0)
    mean_auprcs = np.mean(all_auprcs, axis=0)

    # Plotting the Main Metrics
    plt.figure(figsize=(14, 7))
    plt.plot(range(min_features, max_features+1), mean_aurocs, label='Avg AUROC', marker='o')
    plt.plot(range(min_features, max_features+1), mean_auprcs, label='Avg AUPRC', marker='o')
    plt.xlabel('Number of Features', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.title('Model Performance Metrics', fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return mean_aurocs, mean_auprcs

def save_arrays(aurocs, auprcs, name, folder_location):
    """
    Saves arrays of AUROC and AUPRC to the specified folder with the specified name.
    
    Parameters
    ----------
    aurocs : numpy array
        Array of AUROC scores to save.
    auprcs : numpy array
        Array of AUPRC scores to save.
    name : str
        Name for the saved files.
    folder_location : str
        Path of the folder where the files will be saved.
    """    
    # Ensure the directory exists
    if not os.path.exists(folder_location):
        os.makedirs(folder_location)
    
    # Generate filenames
    auroc_filename = os.path.join(folder_location, f"{name}_auroc.npy")
    auprc_filename = os.path.join(folder_location, f"{name}_auprc.npy")
    
    # Save the arrays
    np.save(auroc_filename, aurocs)
    np.save(auprc_filename, auprcs)

def load_arrays(name, folder_location):
    """
    Loads and returns arrays of AUROC and AUPRC from the specified folder with the specified name.
    
    Parameters
    ----------
    name : str
        Name of the files to load.
    folder_location : str
        Path of the folder where the files are located.
    
    Returns
    -------
    aurocs : numpy array
        Loaded array of AUROC scores.
    auprcs : numpy array
        Loaded array of AUPRC scores.
    """
    
    # Generate filenames
    auroc_filename = os.path.join(folder_location, f"{name}_auroc.npy")
    auprc_filename = os.path.join(folder_location, f"{name}_auprc.npy")
    
    # Load the arrays
    aurocs = np.load(auroc_filename)
    auprcs = np.load(auprc_filename)
    
    return aurocs, auprcs

def detect_cuttoff_gene(data, highlight_value):
    """
    Detects the optimal point based on the first point within the shaded region.
    
    Parameters
    ----------
    data : list or numpy array
        Array of data points to consider.
    highlight_value : float
        Value used to determine the shaded region around the optimal point.
    
    Returns
    -------
    int
        Index of the optimal point.
    """
    top_score = max(data)
    lower_bound = top_score - highlight_value
    upper_bound = top_score + highlight_value
    
    # Find the first point that is within the shaded region
    for i, value in enumerate(data):
        if lower_bound <= value <= upper_bound:
            return i
    return None

def plot_with_cuttoff_gene(aurocs, auprcs, optimal_point, highlight_value):
    """
    Plots AUROCs and AUPRCs vs. Number of Features with optimal point and shaded region.
    
    Parameters
    ----------
    aurocs : list or numpy array
        List of AUROC values.
    auprcs : list or numpy array
        List of AUPRC values.
    optimal_point : int
        Index of the optimal point to highlight.
    highlight_value : float
        Value to determine the size of the shaded region around the top score.
    """
    plt.figure(figsize=(20, 6))
    
    top_score = max(aurocs)  # considering both aurocs and auprcs
    plt.axhspan(top_score - highlight_value, top_score + highlight_value, alpha=0.5, color='lightgrey', zorder=0)
    
    # Plotting AUROCs and AUPRCs after the shaded region to keep them on top
    plt.plot(aurocs, label='Avg AUROC', marker='o', linewidth=1, zorder=1)
    plt.plot(auprcs, label='Avg AUPRC', marker='o', linewidth=1, zorder=1)
    
    # Highlighting the optimal point
    plt.axvline(x=optimal_point, color='red', linestyle='--', label=f'Optimal Point at {optimal_point}', zorder=2)
    
    plt.xlabel('Number of Features')
    plt.ylabel('AUROC Score')
    plt.title('Model Performance Metrics')
    plt.legend()
    
    plt.show()

def print_first_n_indices(importances, n):
    """
    Prints the first n indices from the provided importances object.
    
    Parameters
    ----------
    importances : pandas DataFrame or Series
        Object containing feature importances.
    n : int
        Number of indices to print.
    """    
    # Get the first n indices
    first_n_indices = importances.index[:n]
    
    # Print each index without the '.rescaled' suffix
    for idx in first_n_indices:
        print(idx.replace('.rescaled', ''))

def simulate_model_performance(x_train, y_train, x_test, y_test, model_type, 
                               importance_dataframes, recommended_features, num_simulations,
                               labels=None):
    """
    Simulates model performance using specified feature importance dataframes.
    
    Parameters
    ----------
    x_train : DataFrame or numpy array
        Training features.
    y_train : DataFrame or numpy array
        Training labels.
    x_test : DataFrame or numpy array
        Testing features.
    y_test : DataFrame or numpy array
        Testing labels.
    model_type : str
        Type of the model to use for simulations.
    importance_dataframes : list of DataFrames
        List of DataFrames with feature importances.
    recommended_features : int
        Number of top features to use from the importance dataframes.
    num_simulations : int
        Number of simulations to run.
    labels : list of str, optional
        Labels for each importance dataframe. Defaults to None.
    
    Returns
    -------
    auroc_means : list
        List of mean AUROC scores for each dataframe.
    auprc_means : list
        List of mean AUPRC scores for each dataframe.
    """
    
    # Lists to store results
    aurocs_results = []
    auprcs_results = []

    # Loop over each importance dataframe provided
    for importance_dataframe in importance_dataframes:
        aurocs = []
        auprcs = []

        # Get the top 'recommended_features' number of features from the dataframe
        selected_features = importance_dataframe.index.tolist()[:recommended_features]
        x_train_selected = x_train[selected_features]
        x_test_selected = x_test[selected_features]

        # Padding for CNN. Ensure input is square in shape for CNNs
        if model_type == 'cnn':
            sqrt_p = math.ceil(math.sqrt(x_train_selected.shape[1]))
            padded_dim = sqrt_p**2
            padding_size = padded_dim - x_train_selected.shape[1]
            
            pad_values_train = np.ones((x_train_selected.shape[0], padding_size))
            x_train_selected = np.hstack([x_train_selected, pad_values_train])

            pad_values_test = np.ones((x_test_selected.shape[0], padding_size))
            x_test_selected = np.hstack([x_test_selected, pad_values_test])
        
        # Loop for each simulation
        for _ in tqdm(range(num_simulations)):
            # Create and compile the model
            model = create_model(model_type, x_train_selected.shape[1])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[AUC(name='auc')])

            # Train the model (10 epochs without verbosity)
            model.fit(x_train_selected, y_train, epochs=10, verbose=0)

            # Get predictions for test data
            y_pred = model.predict(x_test_selected)

            # Compute performance metrics and store them
            aurocs.append(roc_auc_score(y_test, y_pred))
            auprcs.append(average_precision_score(y_test, y_pred))

        # Store average performance metrics for all simulations
        aurocs_results.append((np.mean(aurocs), np.std(aurocs)))
        auprcs_results.append((np.mean(auprcs), np.std(auprcs)))

    # Plotting
    if not labels:
        labels = [f'DataFrame {i+1}' for i in range(len(importance_dataframes))]
    auroc_means = [f"{result[0]:.4f} ± {result[1]:.4f}" for result in aurocs_results]
    auprc_means = [f"{result[0]:.4f} ± {result[1]:.4f}" for result in auprcs_results]

    # Define pastel colors
    colors = ['#AEC6CF', '#FFCC99', '#99E6E6', '#FFB3E6', '#FF6666','#FFD1DC','#FFD700', '#FF99E6']
    
    x = np.arange(len(labels))
    width = 0.9  # Increased bar thickness

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    # AUROC plot
    ax[0].bar(x, [float(mean.split(" ± ")[0]) for mean in auroc_means], width, 
              yerr=[float(mean.split(" ± ")[1]) for mean in auroc_means], 
              label='AUROC', capsize=10, color=colors[:len(labels)])
    ax[0].set_ylabel('Scores')
    ax[0].set_title('Mean AUROC by DataFrame')
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(labels)

    # AUPRC plot
    ax[1].bar(x, [float(mean.split(" ± ")[0]) for mean in auprc_means], width, 
              yerr=[float(mean.split(" ± ")[1]) for mean in auprc_means], 
              label='AUPRC', capsize=10, color=colors[:len(labels)])
    ax[1].set_ylabel('Scores')
    ax[1].set_title('Mean AUPRC by DataFrame')
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(labels)

    fig.tight_layout()
    plt.show()
    
    return auroc_means, auprc_means


def extract_mean_std(performance_strings):
    """
    Extracts mean and standard deviation values from a list of performance metric strings.

    Parameters
    ----------
    performance_strings : list of str
        A list of strings, each containing a mean and standard deviation value separated by ' ± '.

    Returns
    -------
    means : numpy array
        Array of mean values extracted from the input strings.
    stds : numpy array
        Array of standard deviation values extracted from the input strings.
    """
    means = []
    stds = []
    for perf_str in performance_strings:
        mean, std = perf_str.split(' ± ')
        means.append(float(mean))
        stds.append(float(std))
    return np.array(means), np.array(stds)

def rank_and_plot(methods, *comparison_groups):
    """
    Creates bar plots to rank and compare methods based on performance metrics across different 
    comparison groups.

    Parameters
    ----------
    methods : list of str
        List of method names corresponding to each comparison group.
    comparison_groups : variable number of arguments
        Each argument is a list of performance metric strings for a specific comparison group.

    """
    # Extract means and standard deviations for each comparison group
    all_means = [extract_mean_std(group)[0] for group in comparison_groups]
    all_stds = [extract_mean_std(group)[1] for group in comparison_groups]

    # Compute the average of means for each method across all comparison groups
    method_avgs = np.array(all_means).mean(axis=0)
    method_stds = np.array(all_stds).mean(axis=0)

    # Sort methods by their average of means
    sorted_indices = np.argsort(-method_avgs)
    sorted_avgs = method_avgs[sorted_indices]
    sorted_stds = method_stds[sorted_indices]
    sorted_methods = [methods[i] for i in sorted_indices]

    # Create the plot with subplots for each method
    fig, axs = plt.subplots(1, len(methods), figsize=(20, 5), sharey=True) # (10, 5)
    legend_handles = [Patch(facecolor='lightblue', label='LOG'),
                  Patch(facecolor='lightgreen', label='MLP'),
                  Patch(facecolor='lightsalmon', label='CNN')]

    # Define colors for the comparisons
    colors = ['lightblue', 'lightgreen', 'lightsalmon', 'lightblue', 'lightgreen', 'lightsalmon']

    # Adjust the gap between test and validation bars
    gap = 0.5
    bar_width = 0.9

    # Plotting the bars
    for idx, ax in enumerate(axs):
        group_count = len(comparison_groups)
        mean_totals = []
        temp_mean_sum = 0
        for j in range(group_count):
            # Adjust the position to create a gap between the 3rd and 4th bar
            pos_adjustment = gap if j >= group_count / 2 else 0
            position = j + pos_adjustment
            color = colors[j]
            mean = all_means[j][sorted_indices[idx]]
            temp_mean_sum += mean
            if((j+1)%3 == 0):
                mean_totals.append(temp_mean_sum/3)
                temp_mean_sum = 0
            
            std = all_stds[j][sorted_indices[idx]]
            # Plot the bar
            bar = ax.bar(position, mean, yerr=std, width=bar_width, color=color, capsize=5,
                         error_kw={'capthick': 2, 'elinewidth': 2, 'ecolor': 'black'})
            

        # Set the title with method name and average
        ax.set_title(f"{sorted_methods[idx]}\n(Average: {sorted_avgs[idx]:.3f})", fontsize=20)

        # Set x-ticks and labels
        ax.set_xticks([1, 4 + gap])
        ax.set_xticklabels([f'Test\n({mean_totals[0]:.3f})', f'Validation\n({mean_totals[1]:.3f})'], fontsize=15)

        # Remove the spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(idx == 0)  

    # Global settings
    axs[0].set_ylabel('AUROC')
    axs[0].set_ylim(0, 1.0) 
    # Place the legend on the figure
    fig.legend(handles=legend_handles, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05))


    plt.tight_layout()
    plt.show()


def rank_and_plot_std(methods, *comparison_groups):
    """
    Creates bar plots to rank and compare methods based on standard deviation of performance metrics 
    across different comparison groups.

    Parameters
    ----------
    methods : list of str
        List of method names corresponding to each comparison group.
    comparison_groups : variable number of arguments
        Each argument is a list of performance metric strings for a specific comparison group.

    """
    # Extract means and standard deviations for each comparison group
    all_means = [extract_mean_std(group)[0] for group in comparison_groups]
    all_stds = [extract_mean_std(group)[1] for group in comparison_groups]

    # Compute the average of means for each method across all comparison groups
    method_avgs = np.array(all_means).mean(axis=0)
    method_stds = np.array(all_stds).mean(axis=0)

    # Sort methods by their average of means
    sorted_indices = np.argsort(-method_avgs)
    sorted_avgs = method_avgs[sorted_indices]
    sorted_stds = method_stds[sorted_indices]
    sorted_methods = [methods[i] for i in sorted_indices]

    # Create the plot with subplots for each method
    fig, axs = plt.subplots(1, len(methods), figsize=(20, 5), sharey=True) # (10, 5)
    legend_handles = [Patch(facecolor='lightblue', label='LOG'),
                  Patch(facecolor='lightgreen', label='MLP'),
                  Patch(facecolor='lightsalmon', label='CNN')]

    # Define colors for the comparisons
    colors = ['lightblue', 'lightgreen', 'lightsalmon', 'lightblue', 'lightgreen', 'lightsalmon']

    # Adjust the gap between test and validation bars
    gap = 0.5
    bar_width = 0.9

    # Plotting the bars
    for idx, ax in enumerate(axs):
        group_count = len(comparison_groups)
        std_totals = []
        temp_std_sum = 0
        for j in range(group_count):
            # Adjust the position to create a gap between the 3rd and 4th bar
            pos_adjustment = gap if j >= group_count / 2 else 0
            position = j + pos_adjustment
            color = colors[j]
            mean = all_means[j][sorted_indices[idx]]            
            std = all_stds[j][sorted_indices[idx]]
            temp_std_sum += std
            if((j+1)%3 == 0):
                std_totals.append(temp_std_sum/3)
                temp_std_sum = 0
            # Plot the bar
            bar = ax.bar(position, std, width=bar_width, color=color)
            

        # Set the title with method name and average
        ax.set_title(f"{sorted_methods[idx]}\n(Average: {sorted_stds[idx]:.4f})", fontsize=20)

        # Set x-ticks and labels
        ax.set_xticks([1, 4 + gap])
        ax.set_xticklabels([f'Test\n({std_totals[0]:.4f})', f'Validation\n({std_totals[1]:.4f})'], fontsize=15)

        # Remove the spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(idx == 0)  # Only show for the first subplot

    # Global settings
    axs[0].set_ylabel('Standard Deviation of AUROC')
    # Place the legend on the figure
    fig.legend(handles=legend_handles, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05))
    plt.tight_layout()
    plt.show()

def calculate_biomarker(df, significance_level, print_feats = False):
    """
    Calculates a biomarker value based on the significant differential expression of features.

    Parameters
    ----------
    df : pandas DataFrame
        The input DataFrame. Each column corresponds to a feature, and each row corresponds to a sample.
        The DataFrame includes a column named 'output' which contains the labels for the two groups (0 and 1).
    significance_level : float
        The significance level for the Mann-Whitney U test.
    print_feats : bool, optional
        If set to True, prints the number of significantly differentially expressed features.

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

    if (print_feats == True):
    # Print the number of significantly differentially expressed features, as well as number of overexpressed and underexpressed ones
        print('Significantly Differentially Expressed: ', total_sig)
        print('Overexpressed: ', overexp)
        print('Underexpressed: ', underexp)
    
    # Return the result DataFrame with calculated biomarker values
    return result_df

def plot_differential_violin(df, column, group_by, ax=None):
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
    sns.violinplot(x=group_by, y=column, data=plot_df, inner='quartile', ax=ax)

    return(p_value)

def plot_survival_curves(df, column, ax=None):
    """
    Plots survival curves for two groups within a DataFrame, split based on the median value of a specified column.

    Parameters
    ----------
    df : pandas DataFrame
        The DataFrame containing survival data.
    column : str
        The column name in the DataFrame based on which the two groups are split.
    ax : matplotlib.axes.Axes, optional
        The axes on which to plot the survival curves. If None, a new figure and axes are created.

    Returns
    -------
    p_value : float
        The p-value from the log-rank test comparing the survival distributions of the two groups.
    """
    kmf = KaplanMeierFitter()
    
    median_value = df[column].median()
    low_group = df[df[column] <= median_value]
    high_group = df[df[column] > median_value]
    
    kmf.fit(low_group['survival_day'], event_observed=low_group['survival_event'], label=f"{column} <= median")
    kmf.plot(ax=ax)
    kmf.fit(high_group['survival_day'], event_observed=high_group['survival_event'], label=f"{column} > median")
    kmf.plot(ax=ax)
    
    # Perform the log-rank test
    results = logrank_test(
        low_group['survival_day'], high_group['survival_day'], 
        event_observed_A=low_group['survival_event'], event_observed_B=high_group['survival_event']
    )
    

    return results.p_value

def plot_combined_analysis(X, y, surv_data, optimal_points, importance_dfs, names, significance_level):
    """
    Performs a combined analysis of survival data using different feature selection methodologies.

    Parameters
    ----------
    X : pandas DataFrame
        Features data.
    y : pandas Series or array-like
        Survival event data.
    surv_data : pandas DataFrame
        Additional survival data.
    optimal_points : list
        List of optimal points for each feature selection method.
    importance_dfs : list of pandas DataFrames
        List of DataFrames containing feature importances for each method.
    names : list of str
        Names of the feature selection methods.
    significance_level : float
        Significance level for biomarker calculation.

    Returns
    -------
    mannwhitney_p_values : list
        List of Mann-Whitney p-values for differential expression analysis.
    logrank_p_values : list
        List of log-rank p-values for survival analysis.
    """
    
    # Merge X, y, and survival data into a single DataFrame
    combined_data = X.copy()
    combined_data['survival_event'] = y
    combined_data = combined_data.merge(surv_data, left_index=True, right_on='sampleName', how='inner')
    
    # Set up the plotting area with a grid of subplots
    num_methods = len(importance_dfs)
    fig, axs = plt.subplots(2, num_methods, figsize=(5 * num_methods, 10), sharey='row')

    mannwhitney_p_values = []
    logrank_p_values = []
    # Iterate through each feature selection methodology
    for i, (optimal_point, importance_df, name) in enumerate(zip(optimal_points, importance_dfs, names)):
        # Filter top features and calculate the biomarker
        important_feats = filter_top_features_with_output(X, y, importance_df, optimal_point)
        custom_biomarker = calculate_biomarker(important_feats, significance_level, True)
        surv_dat_comb = pd.merge(custom_biomarker, surv_data, right_on='sampleName', left_on='id', how='left')
        surv_dat_comb[name + '_biomarker'] = custom_biomarker['biomarker']

        # Plot differential expression
        differential_p_value = plot_differential_violin(surv_dat_comb, name + '_biomarker', 'survival_event', ax=axs[0, i])
        mannwhitney_p_values.append(differential_p_value)
        
        # Plot survival curves
        survival_p_value = plot_survival_curves(surv_dat_comb, name + '_biomarker', ax=axs[1, i])
        logrank_p_values.append(survival_p_value)
        
        # Set subplot titles
        axs[0, i].set_title(f'{name} Differential Expression\nMann-Whitney p-value: {differential_p_value:.4f}')
        axs[1, i].set_title(f'{name} Survival Analysis\nLog-rank Test p-value: {survival_p_value:.4f}')
    
    # Adjust the layout
    plt.tight_layout()
    plt.show()
    
    return mannwhitney_p_values, logrank_p_values

def plot_single_differential_violin(df, column, group_by):
    """
    Plots a violin plot to compare the distribution of values in a specified column between two groups in a DataFrame.

    Parameters
    ----------
    df : pandas DataFrame
        The DataFrame containing the data to be plotted.
    column : str
        The column in the DataFrame whose distribution is to be visualized.
    group_by : str
        The column in the DataFrame used to group the data. Should be categorical with two distinct groups (0 and 1).

    Returns
    -------
    p_value : float
        The p-value from the Mann-Whitney U test, indicating the statistical significance of differences between the two groups.
    """
    
    # Split the data into two groups based on the 'group_by' column
    group0 = df[df[group_by] == 0][column].dropna()
    group1 = df[df[group_by] == 1][column].dropna()
    
    # Perform Mann-Whitney U test to check if distributions of the two groups are significantly different
    _, p_value = mannwhitneyu(group0, group1)
    
    # Create a new DataFrame in a format suitable for seaborn's violinplot function
    plot_df = pd.DataFrame({0: group0, 1: group1}).reset_index().melt(id_vars='index', var_name=group_by, value_name=column)
    
    # Create the violin plot with inner quartiles
    plt.figure(figsize=(6, 5))
    sns.violinplot(x=group_by, y=column, data=plot_df, inner='quartile')

    # Set the title of the plot to include the column name, group_by column, and p-value of the Mann-Whitney U test
    plt.title(f'Mann-Whitney U Test p-value: {p_value:.4f}')
    plt.show()
    return(p_value)


def plot_single_survival_curves(df, column):
    """
    Plots Kaplan-Meier survival curves for two groups in a DataFrame, based on the median or a specified value of a given column.

    Parameters
    ----------
    df : pandas DataFrame
        The DataFrame containing survival data.
    column : str
        The column name in the DataFrame based on which the two groups are split. If 'output', uses a fixed median value of 0.5.

    Returns
    -------
    p_value : float
        The p-value from the log-rank test, comparing the survival distributions of the two groups.
    """
    kmf = KaplanMeierFitter()
    
    if (column == "ki67_status"):
        median_value = 0.5
    else:
        median_value = df[column].median()
    low_group = df[df[column] <= median_value]
    high_group = df[df[column] > median_value]
    print(len(low_group), len(high_group))

    fig, ax = plt.subplots(figsize=(6, 5))

    # Fit the data for the low group and high group
    kmf.fit(low_group['survival_day'], event_observed=low_group['survival_event'], label=f"Ki-67 Negative")
    kmf.plot(ax=ax)
    
    kmf.fit(high_group['survival_day'], event_observed=high_group['survival_event'], label=f"Ki-67 Positive")
    kmf.plot(ax=ax)
    
    # Perform the log-rank test
    results = logrank_test(
        low_group['survival_day'], high_group['survival_day'], 
        event_observed_A=low_group['survival_event'], event_observed_B=high_group['survival_event']
    )
    
    plt.title(f"Survival curves of Ki-67 Status\nLog-rank Test p-value: {results.p_value:.4f}")
    plt.show()

    return results.p_value


def create_model(model_type, input_dim):
    if model_type == 'mlp':
        model =  tf.keras.Sequential()
        model.add(Dense(64, activation='tanh', input_dim=input_dim))
        model.add(layers.Dropout(0.5))
        model.add(Dense(32, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(Dense(32, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(Dense(16, activation='tanh'))
        model.add(layers.Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        return model
    
    elif model_type == 'cnn':
        sqrt_p = math.ceil(math.sqrt(input_dim))
        padded_dim = sqrt_p**2
        
        model = tf.keras.Sequential()
        model.add(layers.Reshape((sqrt_p, sqrt_p, 1), input_shape=(padded_dim,)))
        model.add(layers.Conv2D(16, kernel_size=(3, 3), activation='relu', padding='SAME'))  # Use SAME padding
        
        # Conditionally add pooling layer based on dimensions
        if sqrt_p >= 2:
            model.add(layers.MaxPooling2D(pool_size=(2, 2)))
            sqrt_p = sqrt_p // 2
        
        model.add(layers.Dropout(0.1))
        
        if sqrt_p >= 3:  # Only add this conv layer if dimensions allow
            model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='SAME'))  # Use SAME padding
            
            if sqrt_p >= 2:  # And then check again for pooling
                model.add(layers.MaxPooling2D(pool_size=(2, 2)))
                sqrt_p = sqrt_p // 2

            model.add(layers.Dropout(0.1))

        model.add(layers.Flatten())
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dropout(0.1))
        model.add(layers.Dense(1, activation='sigmoid'))
        return model
    
    
    elif model_type == 'log':
        model = tf.keras.Sequential()
        model.add(layers.Dense(1, activation='sigmoid', input_dim=input_dim))
        return model
    
    else:
        raise ValueError(f"Invalid model_type: {model_type}. Choose from ['mlp', 'cnn', 'log'].")

def filter_top_features_with_output(x_test, y_test, importances_df, cutoff):
    """
    Add y_test to x_test as the first column named 'output', then filter the x_test DataFrame to include only the top 
    features based on the importances dataframe up to the specified cutoff.
    
    Parameters:
    x_test (DataFrame): Feature values.
    y_test (Series): Target values.
    importances_df (DataFrame): Dataframe with feature importances.
    cutoff (int): The number of top features to include.
    
    Returns:
    DataFrame: A new dataframe with 'output' column and the top features based on importances.
    """
    # Add y_test to x_test with column named "output"
    x_test_with_output = x_test.copy()
    x_test_with_output.insert(0, 'output', y_test)

    # Get the top cutoff feature names from importances dataframe
    top_features = importances_df.index[:cutoff].tolist()
    
    # Filter the x_test dataframe to keep only the top features
    filtered_x_test_with_output = x_test_with_output[['output'] + top_features]
    
    return filtered_x_test_with_output


# ------------------------------------------------------------------------
# ---------------- 2-cuttoffModeling_biologicalRelevence -----------------
# ------------------------------------------------------------------------

def load_keras_model(file_path, custom_objects=None):
    """
    Load a Keras model from a specified file path.
    
    Parameters:
    - file_path: The path to the .h5 model file.
    - custom_objects: Dictionary of custom objects, if any, required to load the model.
    
    Returns:
    - Loaded Keras model.
    """
    return load_model(file_path, custom_objects=custom_objects)

def average_rank(dfs, df_names):
    """
    Compute the average rank.
    
    Parameters:
    - dfs: List of dataframes [cnn_importances, mlp_importances, log_importances]
    - df_names: Names of the dataframes ['cnn', 'mlp', 'log'] for labeling
    
    Returns:
    - results_df: A dataframe with ranks given by each dataframe and their average rank.
    """
    
    # Initialize the results dataframe
    results_df = pd.DataFrame(index=dfs[0].index)
    
    for df, name in zip(dfs, df_names):
        results_df[name] = df.reindex(results_df.index)['rank_fin']
        
    # Compute average rank and add to results_df
    results_df['avg_importance'] = results_df.mean(axis=1)
    
    # Sort by average rank (highest to lowest)
    results_df = results_df.sort_values(by='avg_importance', ascending=False)
    results_df.index = results_df.index.str.replace('.rescaled', '', regex=False)
    
    return results_df

def compute_avg_prediction_change(dataset, importance_df, num_features, num_sims, model, pca_reducer):
    """
    Compute the average change in model predictions caused by permuting each top feature.

    Parameters
    ----------
    dataset : pandas DataFrame
        The dataset containing features and output.
    importance_df : pandas DataFrame
        DataFrame containing feature importances.
    num_features : int
        Number of top features to consider.
    num_sims : int
        Number of simulations to run for each feature.
    model : Model object
        Trained model for making predictions.
    pca_reducer : PCA object
        PCA reducer used for dimensionality reduction.

    Returns
    -------
    result_df : pandas DataFrame
        DataFrame showing the average prediction change for each feature.
    """
    # Separate the y values (outputs) from the feature matrix
    y = dataset.iloc[:, 0].values
    X = dataset.iloc[:, 1:]
    
    # Apply the PCA transformation
    X_reduced = pca_reducer.transform(X)
    
    # Get baseline predictions using the provided model on the reduced data
    baseline_preds = model.predict(X_reduced)
    
    # Get the top important features
    top_features = importance_df.index[:num_features]
    
    # Initialize a dictionary to store results
    results = {}
    
    # For each top important feature
    for feature in tqdm(top_features, desc="Processing feature effects"):
        changes = []
        
        # For each simulation
        for _ in range(num_sims):
            # Create a copy of the feature matrix
            X_temp = X.copy()
            
            # Permute the current feature values
            X_temp[feature] = np.random.permutation(X_temp[feature].values)
            
            # Apply the PCA transformation to the permuted feature matrix
            X_temp_reduced = pca_reducer.transform(X_temp)
            
            # Predict using the permuted reduced feature matrix
            permuted_preds = model.predict(X_temp_reduced)
            
            # Calculate the change in prediction for each patient
            change = baseline_preds - permuted_preds
            changes.append(change)
        
        # Calculate the average change in prediction for the current feature
        avg_change = np.mean(changes, axis=0)
        
        # Store the results in the dictionary
        results[feature] = avg_change.squeeze()

    # Convert the dictionary to a dataframe
    result_df = pd.DataFrame(results)
    
    return result_df



def save_dataframe(df, location, filename):
    """
    Save a DataFrame to a specified location with a given filename.

    Parameters:
    - df: DataFrame to be saved
    - location: Directory where the DataFrame should be saved
    - filename: Name of the file (default is "random_results_cnn")

    Returns:
    None
    """
    full_path = f"{location}/{filename}.txt"
    df.to_csv(full_path, sep='\t', index=False, mode='w+')
    print(f"DataFrame saved at {full_path}")
    
def load_dataframe(location, filename):
    """
    Load a DataFrame from a specified location with a given filename.

    Parameters:
    - location: Directory from where the DataFrame should be loaded
    - filename: Name of the file (default is "random_results_cnn")

    Returns:
    - df: Loaded DataFrame
    """
    full_path = f"{location}/{filename}.txt"
    df = pd.read_csv(full_path, sep='\t')
    return df

def compute_gene_correlations(df):
    """
    Compute pairwise correlations between columns (genes) of the given dataframe.
    
    :param df: DataFrame where columns represent genes and rows represent patients.
               The values are the average patient probability change when permuting each gene.
    :return: Pairwise correlation matrix of genes.
    """
    return df.corr()

def model_evaluation_duo(model, X1, y1, X2, y2):
    """
    Generate AUROC and AUPRC plots for model predictions on two datasets.
    
    Parameters
    ----------
    model : object
        The trained model to be evaluated.
    X1 : numpy array
        The features of the first dataset to predict on.
    y1 : numpy array
        The true labels of the first dataset.
    X2 : numpy array
        The features of the second dataset to predict on.
    y2 : numpy array
        The true labels of the second dataset.
    """
    
    # Predict the probabilities for the positive class for both datasets
    y_score1 = model.predict(X1)
    y_score2 = model.predict(X2)
    
    # Compute ROC curve and ROC area for both datasets
    fpr1, tpr1, _ = roc_curve(y1, y_score1)
    roc_auc1 = auc(fpr1, tpr1)
    fpr2, tpr2, _ = roc_curve(y2, y_score2)
    roc_auc2 = auc(fpr2, tpr2)
    
    # Compute precision-recall curve and PR area for both datasets
    precision1, recall1, _ = precision_recall_curve(y1, y_score1)
    pr_auc1 = auc(recall1, precision1)
    precision2, recall2, _ = precision_recall_curve(y2, y_score2)
    pr_auc2 = auc(recall2, precision2)
    
    # Plot ROC curve for both datasets
    plt.figure(figsize=(11, 6))
    plt.subplot(1, 2, 1)
    plt.plot(fpr1, tpr1, color='blue', lw=2, label='AUROC - Test Set (%0.2f)' % roc_auc1)
    plt.plot(fpr2, tpr2, color='orange', lw=2, label='AUROC - Validation Set (%0.2f)' % roc_auc2)
    plt.plot([0, 1], [0, 1], color='grey', lw=2, label='Baseline (0.50)' % roc_auc2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    # Plot precision-recall curve for both datasets
    plt.subplot(1, 2, 2)
    plt.plot(recall1, precision1, color='blue', lw=2, label='AUPRC - Test Set (%0.2f)' % pr_auc1)
    plt.plot([0, 1], [np.mean(y1), np.mean(y1)], label='Baseline - Test Set (%0.2f)' % np.mean(y1), color='blue', lw=2, linestyle='--')

    plt.plot(recall2, precision2, color='orange', lw=2, label='AUPRC - Validation Set (%0.2f)' % pr_auc2)
    plt.plot([0, 1], [np.mean(y2), np.mean(y2)], label='Baseline - Validation Set (%0.2f)' % np.mean(y2), color='orange', lw=2, linestyle='--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")

    
    plt.tight_layout()
    plt.show()
    

def combine_gene_correlations(df1, df2, n_samples1, n_samples2):
    """
    Combine pairwise correlations between genes of two dataframes by taking a weighted average based on the number of samples.

    :param df1: Correlation matrix of genes from the first dataframe.
    :param df2: Correlation matrix of genes from the second dataframe.
    :param n_samples1: Number of samples used to compute correlations in df1.
    :param n_samples2: Number of samples used to compute correlations in df2.
    :return: Combined correlation matrix of genes.
    """

    # Ensure the two dataframes have the same columns and index
    if set(df1.columns) != set(df2.columns) or set(df1.index) != set(df2.index):
        raise ValueError("Both correlation matrices must have the same genes as columns and index.")

    combined_corr = (df1 * n_samples1 + df2 * n_samples2) / (n_samples1 + n_samples2)

    return combined_corr

def dunn_index(X, labels):
    unique_labels = np.unique(labels)
    
    inter_cluster_distances = []
    intra_cluster_distances = []

    # Compute inter-cluster distances
    for i in range(len(unique_labels)):
        for j in range(i+1, len(unique_labels)):
            cluster_i = X[labels == unique_labels[i]]
            cluster_j = X[labels == unique_labels[j]]
            
            # Efficiently compute pairwise distances between points in cluster_i and cluster_j
            distances = cdist(cluster_i, cluster_j, 'euclidean')
            inter_cluster_distances.append(np.min(distances))
                        
    # Compute intra-cluster distances
    for label in unique_labels:
        cluster = X[labels == label]
        if len(cluster) > 1:  # Avoid single-point clusters for this computation
            # Efficiently compute pairwise distances within the same cluster
            distances = cdist(cluster, cluster, 'euclidean')
            intra_distance = np.max(distances + np.eye(distances.shape[0]) * np.max(distances))
            intra_cluster_distances.append(intra_distance)
    
    # If intra_cluster_distances is empty (i.e., every cluster has only one point), return 0
    if not intra_cluster_distances:
        return 0

    return min(inter_cluster_distances) / max(intra_cluster_distances)

def compare_clustering_methods(df, methods, min_clusters=5, max_clusters=20, scoring_type='silhouette',
                              simulations = 10):
    """
    Compare different clustering algorithms and return the one with the best performance based on the selected scoring metric.
    
    Args:
    - df (pd.DataFrame): Data to be clustered. It is expected that the DataFrame has already been preprocessed (e.g., normalized).
    - methods (list): List of clustering methods to be evaluated.
    - min_clusters (int, optional): Minimum number of clusters. Default is 5.
    - max_clusters (int, optional): Maximum number of clusters. Default is 20.
    - scoring_type (str, optional): Type of scoring metric. Choices are 'silhouette', 'davies', or 'calinski'. Default is 'silhouette'.
    
    Returns:
    - list: A list of tuples, where each tuple consists of:
        - (str) clustering method name,
        - (dict) best parameters for that method based on the selected scoring metric,
        - (float) the score achieved by the best parameter set for that method.
    """
    rand_seed = 27
    np.random.seed(rand_seed)
    # Parameter grids for various clustering methods
    param_grid = {
        'kmeans': {
            'n_clusters': list(range(min_clusters, max_clusters + 1)),
            'init': ['k-means++', 'random'],
            'n_init': [10, 15, 20],
            'max_iter': [300, 400, 500, 1000],
            'random_state': [rand_seed]
        },
        'spectral': {
            'n_clusters': list(range(min_clusters, max_clusters + 1)),
            'eigen_solver': [None, 'arpack', 'lobpcg'],
            'affinity': ['nearest_neighbors', 'rbf'],
            'random_state': [rand_seed]
        },
        'agglomerative': {
            'n_clusters': list(range(min_clusters, max_clusters + 1)),
            'affinity': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine'],
            'linkage': ['ward', 'complete', 'average', 'single']
        },
        'dbscan': {
            'eps': list(np.linspace(0.1, 5, 20)),
            'min_samples': [5, 10, 15],
            'metric': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
        },
        'affinity_propagation': {
            'damping': list(np.linspace(0.5, 0.99, 20)),
            'affinity': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
        },
        'optics': {
            'min_samples': list(range(3, 30)),
            'metric': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine'],
            'cluster_method': ['xi', 'dbscan']
        },
        'gmm': {
            'n_components': list(range(min_clusters, max_clusters + 1)),
            'covariance_type': ['full', 'tied', 'diag', 'spherical'],
            'random_state': [rand_seed]
        }
    }

    # Dictionaries to store best scores, parameters, and clusters for each method
    best_scores = {}
    best_params = {}
    best_clusters = {}
    best_davies = {}
    best_calinski = {}
    best_dunn = {}
    
    # Determine which scoring function to use based on scoring_type
    scoring_funcs = {
        'silhouette': silhouette_score,
        'davies': davies_bouldin_score,
        'calinski': calinski_harabasz_score,
        'dunn': dunn_index
    }

    
    if scoring_type not in scoring_funcs:
        raise ValueError(f"Invalid scoring_type: {scoring_type}. Must be one of ['silhouette', 'davies', 'calinski'].")
    
    scoring_func = scoring_funcs[scoring_type]
    
    # Iterate over specified clustering methods
    for method in methods:
        if method in param_grid:
            if scoring_type == "silhouette":
                local_best_score = -1
            elif scoring_type == "calinski":
                local_best_score = -1
            elif scoring_type == "dunn":
                local_best_score = -1
            else:
                local_best_score = 1
            local_best_params = None
            local_best_clusters = None
            
            # Iterate over combinations of parameters for the current method
            for params in tqdm(product(*param_grid[method].values()), desc=f"Trying {method} parameters"):
                param_dict = dict(zip(param_grid[method].keys(), params))

                for i in range(simulations):
                
                    # Suppress specific warnings related to SpectralClustering
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message=".*affinity=precomputed.*", category=UserWarning, module="sklearn.cluster._spectral")

                        try:
                            # Select appropriate clustering algorithm based on the method and apply it to data
                            if method == "kmeans":
                                clusters = KMeans(**param_dict).fit_predict(df)
                            elif method == "dbscan":
                                clusters = DBSCAN(**param_dict).fit_predict(df)
                                unique_clusters = len(np.unique(clusters))
                                if unique_clusters < min_clusters or unique_clusters > max_clusters:
                                    continue
                            elif "agglomerative" in method:
                                clusters = AgglomerativeClustering(**param_dict).fit_predict(df)
                            elif method == "spectral":
                                clusters = SpectralClustering(**param_dict).fit_predict(df)
                            elif method == "affinity_propagation":
                                clusters = AffinityPropagation(**param_dict).fit_predict(df)
                            elif method == "optics":
                                clusters = OPTICS(**param_dict).fit_predict(df)
                                unique_clusters = len(np.unique(clusters))
                                if unique_clusters < min_clusters or unique_clusters > max_clusters:
                                    continue
                            elif method == "gmm":
                                gmm = GaussianMixture(**param_dict).fit(df)
                                clusters = gmm.predict(df)
                            else:
                                continue  # Unknown method
                        
                            # Calculate scoring metric
                            score = scoring_func(df, clusters)

                            # Update best scores, parameters, and clusters if the current configuration is better
                            # For silhouette and calinski higher is better, for davies lower is better
                            is_better = (score > local_best_score) if scoring_type in ['silhouette', 'calinski','dunn'] else (score < local_best_score)
                            if is_better:
                                local_best_score = score
                                local_best_params = param_dict
                                local_best_clusters = clusters

                        except Exception as e:
                            continue
                # Store best results for the current method
                best_scores[method] = local_best_score
                best_params[method] = local_best_params
                best_clusters[method] = local_best_clusters

    # Rank clustering methods by their best achieved score
    reverse_order = scoring_type in ['silhouette', 'calinski', 'dunn']
    rank_score = {k: v for k, v in sorted(best_scores.items(), key=lambda item: item[1], reverse=reverse_order)}

    # Create tuples of method names, their best parameters, and achieved scores
    best_methods_with_params_and_scores = [
        (method, best_params[method], best_scores.get(method, None))
        for method in methods if method in best_scores
    ]

    # Rank the methods according to the scoring_type selected
    best_methods_with_params_and_scores = sorted(
        best_methods_with_params_and_scores, 
        key=lambda x: x[2], 
        reverse=reverse_order
    )
    
    def cluster_size_violation(clusters):
        violations = 0
        for label in np.unique(clusters):
            if np.sum(clusters == label) < 4:
                violations += 1
        return violations

    
    # Plot the performance of each clustering method
    plt.figure(figsize=(12, 6))
    grey_out = [cluster_size_violation(best_clusters.get(method, [])) for method in rank_score.keys()]
    bars = plt.bar(rank_score.keys(), rank_score.values(), color=['lightgrey' if violation else 'lightgrey' for violation in grey_out])
    plt.title('Clustering Algorithms Performance')
    plt.ylabel(scoring_type.capitalize())
    xticks_pos = [bar.get_x() + bar.get_width() / 2 for bar in bars]
    plt.xticks(xticks_pos, rank_score.keys(), rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    # Plot the clustering results for each method
    n_rows = int(np.ceil(len(methods) / 3))
    fig, axs = plt.subplots(n_rows, 3, figsize=(18, 6 * n_rows))
    axs = axs.ravel()

    for i, method in enumerate(methods):
        if method in best_clusters and best_clusters[method] is not None:
            sorted_idx = np.argsort(best_clusters[method])
            df_sorted = df.iloc[sorted_idx, sorted_idx]
            im = axs[i].imshow(df_sorted, cmap="RdBu", aspect="equal", vmin=-1, vmax=1)

            current_idx = 0
            for cluster_label in np.unique(best_clusters[method]):
                size = np.sum(best_clusters[method][sorted_idx] == cluster_label)
                rect = plt.Rectangle((current_idx-0.5, current_idx-0.5), size, size, fill=False, edgecolor="black", linewidth=2)
                axs[i].add_patch(rect)
                current_idx += size
                
            title_method = method.replace("_", " ").replace("agglomerative", "agglomerative-")
            title_first_line = f"{title_method} - Clusters: {len(np.unique(best_clusters[method]))}"
            if scoring_type == 'silhouette':
                title_second_line = f"Silhouette: {best_scores[method]:.3f}"
            elif scoring_type == 'davies':
                title_second_line = f"Davies: {best_scores[method]:.3f}"
            elif scoring_type == 'calinski':
                title_second_line = f"Calinski: {best_scores[method]:.3f}"
            elif scoring_type == 'dunn':
                title_second_line = f"Dunn: {best_scores[method]:.3f}"
            else:
                raise ValueError(f"Invalid scoring_type: {scoring_type}. Must be one of ['silhouette', 'davies', 'calinski'].")            
            

            axs[i].set_title(title_first_line + "\n" + title_second_line)
            axs[i].axis('off')

    for i in range(len(methods), n_rows * 3):
        axs[i].axis('off')

    plt.tight_layout()
    plt.show()


    best_method_by_rank_score = list(rank_score.keys())[0]
    
    return best_methods_with_params_and_scores


def save_data(data, location):
    # Extract file name and extension
    file_name = os.path.basename(location)
    if isinstance(data, pd.DataFrame):
        # Save as CSV for DataFrames
        data.to_csv(location + '.csv', index=True)
    elif isinstance(data, list):
        # Save as TXT for lists
        with open(location + '.txt', 'w') as f:
            for item in data:
                f.write("%s\n" % str(item))
    elif isinstance(data, np.ndarray):
        # Save as NPY for numpy arrays
        np.save(location + '.npy', data)
    else:
        print("Unsupported data type.")


def load_data(location):
    # Extract file name and extension
    file_name, file_extension = os.path.splitext(os.path.basename(location))
    
    if file_extension == '.csv':
        # Load CSV for DataFrames
        return pd.read_csv(location, index_col=0)
    elif file_extension == '.txt':
        # Load as simple list
        with open(location, 'r') as f:
            lines = f.readlines()
            return [eval(line.strip()) for line in lines]
    elif file_extension == '.npy':
        # Load NPY for numpy arrays
        return np.load(location)
    else:
        print("Unsupported file type.")
        return None
    
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering, AffinityPropagation, OPTICS
from sklearn.mixture import GaussianMixture
import numpy as np
import seaborn as sns
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.distance import pdist


def run_and_plot_best_clustering(df, log_best_params, n_sims=10, scoring_type='silhouette', 
                                 force_method=None, save_location_name=None):
    """
    Runs clustering using the best parameters and plots the results.

    Parameters
    ----------
    df : pandas DataFrame
        The dataset for clustering.
    log_best_params : list
        Log of the best parameters for each clustering method.
    n_sims : int, optional
        Number of simulations to run, by default 10.
    scoring_type : str, optional
        The scoring type to evaluate the clustering (e.g., 'silhouette', 'calinski'), by default 'silhouette'.
    force_method : str, optional
        Force a specific clustering method, by default None.
    save_location_name : str, optional
        The location and name to save the plot, by default None.

    Returns
    -------
    tuple
        Returns the best cluster assignments, linkage matrix, column names, and sorted best clusters.
    """
    # Extract the best parameters for each method
    method_dict = {method: params for method, params, score in log_best_params}
    
    # Determine the best method and parameters
    if force_method:
        best_method = force_method
        best_params = method_dict.get(force_method, {})
    else:
        best_method = log_best_params[0][0]
        best_params = log_best_params[0][1]
    
    # Initialize variables for tracking the best and worst scores
    best_score = float('-inf') if scoring_type in ['silhouette', 'calinski','dunn'] else float('inf')
    worst_score = float('inf') if scoring_type in ['silhouette', 'calinski','dunn'] else float('-inf')
    best_clusters = None
    
    best_score = float('-inf') if scoring_type in ['silhouette', 'calinski','dunn'] else float('inf')
    worst_score = float('inf') if scoring_type in ['silhouette', 'calinski','dunn'] else float('-inf')
    best_clusters = None
    
    # Define scoring functions
    scoring_functions = {
        'silhouette': silhouette_score,
        'calinski': calinski_harabasz_score,
        'davies': davies_bouldin_score,
        'dunn': dunn_index
    }
    
    # Run simulations to find the best clustering
    for _ in tqdm(range(n_sims)):
        method = best_method
        param_dict = best_params
        
        # Choose the clustering algorithm based on the method
        if method == "kmeans":
            clusters = KMeans(**param_dict).fit_predict(df)
        elif method == "dbscan":
            clusters = DBSCAN(**param_dict).fit_predict(df)
            unique_clusters = len(np.unique(clusters))
            if unique_clusters < min_clusters or unique_clusters > max_clusters:
                continue
        elif "agglomerative" in method:
            clusters = AgglomerativeClustering(**param_dict).fit_predict(df)
        elif method == "spectral":
            clusters = SpectralClustering(**param_dict).fit_predict(df)
        elif method == "affinity_propagation":
            clusters = AffinityPropagation(**param_dict).fit_predict(df)
        elif method == "optics":
            clusters = OPTICS(**param_dict).fit_predict(df)
            unique_clusters = len(np.unique(clusters))
            if unique_clusters < min_clusters or unique_clusters > max_clusters:
                continue
        elif method == "gmm":
            gmm = GaussianMixture(**param_dict).fit(df)
            clusters = gmm.predict(df)
        else:
            continue  # Unknown method
                
        # Calculate the score for the current clustering
        score = scoring_functions[scoring_type](df, clusters)
        
        # Check and update the best score and best clusters
        if scoring_type in ['silhouette', 'calinski', 'dunn']:
            if score > best_score:
                best_score = score
                best_clusters = clusters
        else:  # For 'davies' where lower is better
            if score < best_score:
                best_score = score
                best_clusters = clusters         
        
        # Update the score range
        worst_score = min(worst_score, score) if scoring_type in ['silhouette', 'calinski', 'dunn'] else max(worst_score, score)
            
    # Plot the best clustering
    sorted_idx = np.argsort(best_clusters)
    df_sorted = df.iloc[sorted_idx, :].iloc[:, sorted_idx]
    
    # Plot heatmap of correlations
    g = sns.clustermap(df_sorted, method='average', cmap="RdBu", figsize=(20, 20),
                       row_cluster=False, col_cluster=False, center=0, cbar_pos=(0.02, 0.65, 0.03, 0.18))
    
    distance_matrix = pdist(df_sorted)

    g.ax_heatmap.set_xticks(np.arange(df_sorted.shape[0]))
    g.ax_heatmap.set_yticks(np.arange(df_sorted.shape[1]))
    
    x_labels = [label.replace(".rescaled", "") for label in df_sorted.columns]
    y_labels = [label.replace(".rescaled", "") for label in df_sorted.index]
    
    g.ax_heatmap.set_xticklabels(x_labels, size=8, rotation=90)
    g.ax_heatmap.set_yticklabels(y_labels, size=8)
    
    unique_clusters = np.unique(best_clusters)
    start_x = 0
    for cluster_num in unique_clusters:
        cluster_size = sum(best_clusters == cluster_num)
        g.ax_heatmap.add_patch(plt.Rectangle((start_x, start_x), cluster_size, cluster_size, facecolor='none', edgecolor='black', lw=3))
        start_x += cluster_size
    
    # g.fig.suptitle(f"Method: {best_method} --- Params: {best_params} --- Score: {best_score:.3f}\nScore Range: ({worst_score:.3f} - {best_score:.3f})")
    g.ax_heatmap.text(0, 1.2, f"Method: {best_method} \nParams: {best_params} \nScore: {best_score:.3f} ({worst_score:.3f} - {best_score:.3f})", 
                      size=20, ha="center", va="top", transform=g.ax_heatmap.transAxes)

    print(f"Recommended number of clusters: {len(unique_clusters)}")
    plt.show()
    
     # Save the figure to the specified location if provided
    if save_location_name:
        save_path = f"./results/plots/{save_location_name}.png"
        g.savefig(save_path, dpi=600, bbox_inches='tight')  # Using high DPI for quality and bbox_inches to ensure labels are not cut off.
    
    linkage_matrix = hierarchy.linkage(distance_matrix, method='average')
    #return best_clusters, linkage_matrix
    return best_clusters, linkage_matrix, df_sorted.columns, best_clusters[sorted_idx]



def get_scaled_node_sizes(log_importances):
    """
    Scale the rank_fin values in log_importances to a suitable range for visualization.
    
    :param log_importances: pandas DataFrame with rank_fin values.
    :return: A dictionary with nodes and their scaled sizes.
    """
    # Selecting the 'rank_fin' column for the top 100 nodes
    top_nodes = log_importances['rank_fin'].head(100)
    
    # No suffix removal; names remain unchanged
    cleaned_names = top_nodes.index.tolist()

    # Scaling values to be between 500 and 2000 for visualization purposes
    scaled_sizes = 2000 * (top_nodes - top_nodes.min()) / (top_nodes.max() - top_nodes.min()) + 1000
    
    return dict(zip(cleaned_names, scaled_sizes))

def plot_individual_clusters(df, clusters, title_prefix, importances, 
                             critical_gene_number, K_space=None, 
                             min_node_count=None, save=None):
    """
    Visualizes individual clusters from a dataset as network graphs.

    Parameters
    ----------
    df : pandas DataFrame
        The dataset for clustering analysis.
    clusters : array-like
        Cluster labels for each node in the dataset.
    title_prefix : str
        A prefix to be added to the title of each plot.
    importances : pandas Series
        Feature importances to determine node sizes in the network graph.
    critical_gene_number : int
        Number of top important genes to be highlighted.
    K_space : float, optional
        The optimal distance between nodes in the network graph, by default None.
    min_node_count : int, optional
        Minimum number of nodes required to form a cluster, by default None.
    save : str, optional
        Directory to save the plots, by default None.

    Returns
    -------
    None
        This function does not return anything. It plots and optionally saves network graphs.
    """

    unique_clusters = set(clusters)
    
    cluster_similarities = {}
    for cluster_num in unique_clusters:
        nodes = [idx for idx, cluster_id in enumerate(clusters) if cluster_id == cluster_num]
        sub_df = df.iloc[nodes, nodes]
        avg_similarity = np.mean(np.absolute(sub_df.values[np.triu_indices_from(sub_df, k=1)]))  
        cluster_similarities[cluster_num] = avg_similarity
        
    sorted_clusters = sorted(cluster_similarities, key=cluster_similarities.get, reverse=True)
    
    node_sizes_dict = get_scaled_node_sizes(importances)
    
    critical_genes = set(importances.index[:critical_gene_number])

    for cluster_num in sorted_clusters:
        nodes = [idx for idx, cluster_id in enumerate(clusters) if cluster_id == cluster_num]
        sub_df = df.iloc[nodes, nodes]
        G = nx.from_pandas_adjacency(sub_df)
        G.remove_edges_from(nx.selfloop_edges(G))
        
        edges_sorted_by_weight = sorted(G.edges(data=True), key=lambda x: abs(x[2]['weight']))
        removed_edges = []

        for edge in edges_sorted_by_weight:
            u, v, data = edge
            G.remove_edge(u, v)
            removed_edges.append((u, v, data))
                    
            if min_node_count is not None and len(nodes) < min_node_count:
                if any(len(edges) < 2 for node, edges in G.adjacency()):
                    last_removed = removed_edges.pop()
                    u, v, data = last_removed
                    G.add_edge(u, v, **data)
                    break
            else:
                if not nx.is_connected(G):
                    last_removed = removed_edges.pop()
                    u, v, data = last_removed
                    G.add_edge(u, v, **data)
                    break
        
        plt.figure(figsize=(20, 20))
        pos = nx.spring_layout(G, k=K_space)
        colors = ['black' if G[u][v]['weight'] > 0 else 'red' for u, v in G.edges]
        
        node_colors = ["lightgreen" if node in critical_genes else "lightblue" for node in G.nodes]
        node_alpha = 1.0
        labels = {node: node.replace('.rescaled', '') for node in G.nodes()}

                
        nx.draw_networkx_edges(G, pos, edge_color=colors, width=2)
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=14)
        node_sizes = [node_sizes_dict.get(node, 100) * 5 for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=node_alpha)
        
        title = f"{title_prefix} - Cluster {cluster_num+1} (Nodes: {len(nodes)}, Avg Absolute Similarity: {cluster_similarities[cluster_num]:.3f})"
        plt.title(title, fontsize=16)
        plt.axis('off')
        plt.tight_layout()

        # If a save path is provided, save the figure.
        if save is not None:
            if not os.path.isdir(save):
                os.makedirs(save)
            save_path = os.path.join(save, f"{title_prefix}_cluster_{cluster_num+1}_AvgSimilarity_{cluster_similarities[cluster_num]:.3f}.png")
            plt.savefig(save_path, format='png', dpi=600)  # Save as high resolution for paper
            print(f"Saved: {save_path}")
        
        plt.show()

