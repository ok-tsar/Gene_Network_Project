# Gene Network Project

Load neccessary libraries and functions.

```

import math
import pickle
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

functions = importlib.machinery.SourceFileLoader('module_name', './functions_used.py').load_module()
```

## --- DATA LOADING AND PROCESSING ---

Load Data Set and Split into Training and Test (make sure first column of data is 'output' - phenotype you want to explore)
```
GSE96058 = functions.data_load('./data/GSE96058_cleaned.csv')
x_train, x_test, y_train, y_test = functions.train_test_splitting(GSE96058)
```

Visualizing Data 
```
functions.visualize_training_data(x_train, y_train)
```
![image](https://github.com/ok-tsar/Gene_Network_Project/assets/54241448/7da83681-9d1c-44fb-8398-5f771d27305b)


Dimensionality Reduction (optional but highly recommend)

any of following are available ('PCA', 'FA', 'LDA', 'SVD', 'KernelPCA', 'SparsePCA', 'Isomap', 'ICA', 'NMF')

```
k = 100
print("--- kernel pca ---")
x_train_reduced, x_test_reduced, kernelpca_reducer = functions.apply_dimensionality_reduction(x_train, x_test, 'KernelPCA', k)
functions.visualize_training_data(x_train_reduced, y_train)
```

![image](https://github.com/ok-tsar/Gene_Network_Project/assets/54241448/ec431f15-f249-462e-8b2d-e37400a9b6ee)

## --- BUILDING MODEL ---

Build desired model to evaluate below are two example models 

```
# SKLEARN MLP (FULLY CONNECTED NETWORK) MODEL
model_mlp = MLPClassifier(activation='logistic', alpha=0.05, hidden_layer_sizes= (32,32,32,16), learning_rate='adaptive', max_iter=10000)
functions.model_evaluation(model_mlp, x_train_reduced, y_train, folds = 5)

# KERAS (CNN NETWORK) MODEL
def create_model():
    sqrt_p = math.ceil(math.sqrt(k))
    model = Sequential()
    model.add(Reshape((sqrt_p, sqrt_p, 1), input_shape=(k,)))
    model.add(Conv2D(16, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.1))    
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['AUC'])
    return model

encoder = LabelBinarizer()
y_train = encoder.fit_transform(y_train)

model_cnn = KerasClassifier(build_fn=create_model, epochs=100, batch_size=32, verbose=0)
functions.model_evaluation(model_cnn, x_train_reduced, y_train, folds = 5)
```

### MLP MODEL
![image](https://github.com/ok-tsar/Gene_Network_Project/assets/54241448/af7a2d96-08f1-4b3f-b30a-9df8827023fb)

Avg F1 score: 0.80 --- Avg Accuracy: 0.77 --- Avg AUROC: 0.83 --- Avg AUPRC: 0.88

### CNN MODEL
![image](https://github.com/ok-tsar/Gene_Network_Project/assets/54241448/9aa67260-4de4-4da6-b06f-3cf1b603886a)

Avg F1 score: 0.77 --- Avg Accuracy: 0.73 --- Avg AUROC: 0.79 --- Avg AUPRC: 0.84

Save Model of Choice 
```
# for sklearn models
joblib.dump(model_mlp, './model/model_mlp')
# for keras models
model_cnn.model.save('./model/model_cnn.h5')
```

## --- GENE IMPORTANCE GENERATION ---

 The get_feature_importance function evaluates feature importance by permuting the features of a test dataset and
 comparing the model's performance with the permuted data to its performance with the original data.
 The difference in performance serves as an indication of the feature's importance.

- Parameters:
- x_test (DataFrame): The test data features
- y_test (Series): The test data target
- reducer (PCA or other dimensionality reduction object, optional): If provided, this is used to transform the data before prediction
- model_loc (str): The location of the model file to be loaded for prediction
- location (str): The directory to save the output files
- N (int): The number of times to permute each feature
   
 The fetch_imporances function retrieves all run results of get_feature_importance function 

Important - if doing multiple runs of this function make sure to keep same N (i.e. if you wanted 100 permutations, you could run below function with n = 25 four times, or n = 50 two times, or n = 100 one time. But you can't run n = 75 once and n = 25 once.

```
functions.get_feature_importance(x_test, y_test, kernelpca_reducer, 
                                 './model/model_mlp','./feature_importance/model_mlp', 100)
                                 
importances = functions.fetch_importances(GSE96058, './feature_importance/model_mlp')
```

Selecting important genes
This function evaluates a model's performance for varying numbers of the most important features and plots 
the resulting metrics. Additionally, it identifies an recomended cutoff for the number of features 
to include based on a threshold for the derivative of the AUROC curve.

```
genes_for_consideration = np.array(range(1, 100, 1))
mlp_int = MLPClassifier(activation='logistic',
              hidden_layer_sizes= (32,32,32,16), learning_rate='adaptive',
              max_iter=10000)
cutoff = functions.plot_metrics_vs_genes(mlp_int, x_train, y_train, importances, genes_for_consideration, 0.001)
importances.head(cutoff).to_pickle('./gene_importances/top_genes.pkl')
imp_genes = pd.read_pickle('./gene_importances/top_genes.pkl')
```
![image](https://github.com/ok-tsar/Gene_Network_Project/assets/54241448/a2f777f6-bb48-4089-963e-2358546e69a3)

## --- IMPORTANT GENE EXPLORATION ---

Now that we have the important genes in terms of our models decision making regarding the phenotype we can see what only modeling those genes looks like
```
important_gene = list(imp_genes.index)
important_gene.insert(0, 'output')

x_train_topGene, x_test_topGene, y_train, y_test = functions.train_test_splitting(GSE96058[important_gene])
# uncomment to visualize top genes 
# functions.visualize_training_data(x_train_topGene, y_train)

del mlp_int
mlp_int = MLPClassifier(hidden_layer_sizes=(32,32,32,16),
                        learning_rate='adaptive', max_iter=10000)
mlp_int.fit(x_train_topGene, y_train)
acc, f1, auc, prc, prec, rec = functions.test_evaluation(mlp_int, x_test_topGene, x_train_topGene, 
                                                         y_test, y_train)
```

![image](https://github.com/ok-tsar/Gene_Network_Project/assets/54241448/44dabfe8-135b-423f-b1cd-3cc863e5f1e9)
![image](https://github.com/ok-tsar/Gene_Network_Project/assets/54241448/f6f43269-409d-4d39-bf0f-88cea5df6591)

- Test Accuracy: 0.72
- f1: 0.771513353115727
- precision: 0.7303370786516854
- recall: 0.8176100628930818

How good are these results? We can use the function below to capture random selections of same number of genes in the same model
```
del mlp_int
mlp_int = MLPClassifier(hidden_layer_sizes=(32,32,32, 16),
                        learning_rate='adaptive', max_iter=10000)
f1_scores, accuracies, aurocs, auprs, recalls, precisions = functions.random_feature_eval(mlp_int,
                                                                                          x_train, x_test, 
                                                                                          y_train, y_test, 
                                                                                          5000, cutoff)
functions.plot_hist_and_value(aurocs, auc)
functions.plot_hist_and_value(auprs, prc)
```
AUROC of our top genes in 99.3 top percentile of (5000) draws of same amount of randomly selected genes
![image](https://github.com/ok-tsar/Gene_Network_Project/assets/54241448/7002cb89-785e-441e-8eda-9d78f64e307c)

AUPRC of our top genes in 99.3 top percentile of (5000) draws ofsame amount of randomly selected genes
![image](https://github.com/ok-tsar/Gene_Network_Project/assets/54241448/669bbe1b-ef84-429b-9dad-28c2d2f58594)

Additionally when looking at same analysis for compleatly seperate dataset similar results.
(Skip if no other dataset)

```
GSE81538 = functions.data_load('./data/GSE81538_cleaned.csv')
x_train_GSE81538_topGene, x_test_GSE81538_topGene, y_train_GSE81538, y_test_GSE81538 = functions.train_test_splitting(GSE81538[important_gene])
acc_GSE81538, f1_GSE81538, auc_GSE81538, prc_GSE81538, prec_GSE81538, rec_GSE81538 = functions.test_evaluation(mlp_int, x_test_GSE81538_imp, 
                                                                                                               x_train_GSE81538_imp, y_test_GSE81538, y_train_GSE81538)
```
![image](https://github.com/ok-tsar/Gene_Network_Project/assets/54241448/db7f2168-3841-4cc1-b7d9-c0988055bb3a)
![image](https://github.com/ok-tsar/Gene_Network_Project/assets/54241448/3e3609bf-ab7c-49e1-9701-5d99e79c15e9)

- Test Accuracy: 0.77
- f1: 0.7246376811594202
- precision: 0.7352941176470589
- recall: 0.7142857142857143

```
del mlp_int
mlp_int = MLPClassifier(hidden_layer_sizes=(32,32,32, 16),
                        learning_rate='adaptive', max_iter=10000)
f1_scores_GSE81538, accuracies_GSE81538, aurocs_GSE81538, auprs_GSE81538, recalls_GSE81538, precisions_GSE81538 = functions.random_feature_eval(mlp_int, x_train_GSE81538, 
                                                                                                                                                x_test_GSE81538, y_train_GSE81538,       
                                                                                                                                                y_test_GSE81538, 5000, cutoff)
```
AUROC of our top genes in 99.88 top percentile of (5000) draws of same amount of randomly selected genes
![image](https://github.com/ok-tsar/Gene_Network_Project/assets/54241448/c992cb52-ca16-4ceb-9a9c-935cf0bee6f9)

AUPRC of our top genes in 100 top percentile of (5000) draws ofsame amount of randomly selected genes
![image](https://github.com/ok-tsar/Gene_Network_Project/assets/54241448/f4665440-6356-4a18-9f15-c399a42ebcd2)

Differential Expression Analysis of Top Genes in Two Data Sets

```
functions.plot_differencial_expression_analysis(GSE96058[important_gene])
```
![image](https://github.com/ok-tsar/Gene_Network_Project/assets/54241448/67d8ad87-6749-4c97-8e04-e47df2e4e31a)
```
functions.plot_differencial_expression_analysis(GSE81538[important_gene])
```
![image](https://github.com/ok-tsar/Gene_Network_Project/assets/54241448/87280130-637e-45af-be04-5f9a1df84076)

```
custom_biomarker = functions.calculate_biomarker(GSE96058[important_gene], 0.05)
functions.plot_differential_violin(custom_biomarker, 'biomarker', 'output')
survival_dat = pd.read_csv('./data/GSET96058_info2.csv', delimiter='\,')
survival_dat = pd.merge(custom_biomarker, survival_dat, right_on='sampleName', left_on='id', how='left')
functions.plot_differential_violin(surv_dat, 'ki67_status', 'survival_event')
```
![image](https://github.com/ok-tsar/Gene_Network_Project/assets/54241448/76f118ea-579b-4d9a-9f48-a4ad92e79278)
![image](https://github.com/ok-tsar/Gene_Network_Project/assets/54241448/1892c124-d0a0-485a-bcba-e4e4cbb3f6c1)

```
functions.plot_survival_curves(surv_dat, 'biomarker')
functions.plot_survival_curves(surv_dat, 'ki67_status')
```

![image](https://github.com/ok-tsar/Gene_Network_Project/assets/54241448/b3b0609d-3415-443d-8616-fc9ebb0fe251)
![image](https://github.com/ok-tsar/Gene_Network_Project/assets/54241448/5e21d5dd-1fc7-40ca-a50e-f2decdbd64c0)

## --- GENE PHENOTYPIC IMPACT SIMILARITY ---

```
# for sklearn
model = joblib.load('./model/model_mlp')
# for keras models
# model = load_model('./model/model_cnn.h5')

similarity_matrix = functions.permute_and_predict(x_test, list(GSE96058[important_gene].index), model, kernelpca_reducer, 500)
G, subs = functions.create_dendrogram_and_clustering_graph(similarity_matrix, 70)
```
![image](https://github.com/ok-tsar/Gene_Network_Project/assets/54241448/712a0208-bc8f-4fc0-90cb-f5fa08af41a7)
![image](https://github.com/ok-tsar/Gene_Network_Project/assets/54241448/8827873b-5df1-4b9e-b7b7-48cbd47777a1)
![image](https://github.com/ok-tsar/Gene_Network_Project/assets/54241448/d6bbc026-7932-4344-8464-c7a8c6e133f1)

```
plot_subgraphs(G, subs)
```
![image](https://github.com/ok-tsar/Gene_Network_Project/assets/54241448/124b645d-a705-41c1-8cdc-9f8d036f44c6)
![image](https://github.com/ok-tsar/Gene_Network_Project/assets/54241448/def05f08-dbcb-4c6d-960a-728a9a57c745)
![image](https://github.com/ok-tsar/Gene_Network_Project/assets/54241448/d06bba2e-643d-4d42-8c84-c2aecb086a0a)
![image](https://github.com/ok-tsar/Gene_Network_Project/assets/54241448/39118352-af6f-4a36-9899-d7aa0c856cf2)
![image](https://github.com/ok-tsar/Gene_Network_Project/assets/54241448/8f18457c-c23c-4866-bdfd-b14f31672690)
![image](https://github.com/ok-tsar/Gene_Network_Project/assets/54241448/b6e7c393-7c9f-4b8b-a4ab-29fcbd4ee3d5)
![image](https://github.com/ok-tsar/Gene_Network_Project/assets/54241448/b2b17e49-0dcd-4155-b645-edabf0080e4a)
![image](https://github.com/ok-tsar/Gene_Network_Project/assets/54241448/0d2d216f-c483-4914-873c-b858720d5f25)
![image](https://github.com/ok-tsar/Gene_Network_Project/assets/54241448/44d83c92-f7f8-4064-a401-ad0393b00ed0)
![image](https://github.com/ok-tsar/Gene_Network_Project/assets/54241448/9fc3e5cf-c2e7-4bf9-bd31-10656f5a071c)
![image](https://github.com/ok-tsar/Gene_Network_Project/assets/54241448/3fefabd8-e695-445e-b653-4a06b335b268)
![image](https://github.com/ok-tsar/Gene_Network_Project/assets/54241448/f11dc186-1bd6-4d48-994f-2ef6c6637389)
![image](https://github.com/ok-tsar/Gene_Network_Project/assets/54241448/6046f14b-0a40-4771-9c31-a92a47883236)
![image](https://github.com/ok-tsar/Gene_Network_Project/assets/54241448/962937f3-6789-477f-9890-149f5dfee1c8)
![image](https://github.com/ok-tsar/Gene_Network_Project/assets/54241448/2426f444-ab77-4d39-b042-e03bd0337066)
![image](https://github.com/ok-tsar/Gene_Network_Project/assets/54241448/7e1b81af-03ab-49de-b2ec-090bbcb0df32)
![image](https://github.com/ok-tsar/Gene_Network_Project/assets/54241448/3b8fcbb8-f6c8-4b3c-8286-be6f6588f174)

```
model = joblib.load('./model/model_mlp')
similarity_matrix_GSE81538 = functions.permute_and_predict(x_test_GSE81538, list(GSE81538[important_gene].index), model, kernelpca_reducer, 500)
G_2, subs_2 = functions.create_dendrogram_and_clustering_graph(similarity_matrix_2, 60,17)
```
![image](https://github.com/ok-tsar/Gene_Network_Project/assets/54241448/6c6de343-52e0-4b5b-a08f-fa51f19b2a0d)
![image](https://github.com/ok-tsar/Gene_Network_Project/assets/54241448/6cd13eb1-4b7c-4634-a65e-162d0789889f)



