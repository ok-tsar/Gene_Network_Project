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
GSE81538 = functions.data_load('./data/GSE81538_cleaned.csv')

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
                                 './model/model_cnn.h5','./feature_importance/model_cnn', 100)
                                 
importances = functions.fetch_importances(GSE96058, './feature_importance/model_cnn')
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
cutoff = functions.plot_metrics_vs_genes(mlp_int, x_train, y_train, importances, genes_for_consideration, 0.002)
```
![image](https://github.com/ok-tsar/Gene_Network_Project/assets/54241448/43dccf54-9ca0-4fc0-bc9a-623ca2d7fdda)

