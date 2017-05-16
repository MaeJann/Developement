### Script to find the best hyperparameter for each classifier

#%% Import modules & functions:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression

from sklearn.grid_search import GridSearchCV

from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import scale
#%% Load data (labels and features)
    
data = pd.DataFrame(pd.read_csv('C:/Users/j-mae/Desktop/Master Thesis/Image Data/Test Data/Merged Data/Naxos_Boudin_8/Extracted/All_Minerals.csv', header=None, sep = ';' ))  
data = data.as_matrix()
labels = data[:,3]
label_names = ['Cal','Fsp','Bt','Amp','Op']
    
Features = pd.DataFrame(pd.read_csv('C:/Users/j-mae/Desktop/Master Thesis/Image Data/Test Data/Merged Data/Naxos_Boudin_8/Extracted/Featurematrix.csv', sep = ';' ))
Features = Features.as_matrix()
X = Features[:, 1:]

# Option: Use scikit learn preprocessing functions to scale or standartize features:
#X = minmax_scale(X)
X = scale(X)  

#%% Parameter tuning for KNN

# Find best parameters using GridSearchCV:
knn = KNeighborsClassifier() # Instantiate classifier with default parameter                                   
knn_param_grid = dict(n_neighbors = np.arange(1, 10), weights = ('uniform', 'distance'), p = (1,2), algorithm = ('auto', 'ball_tree', 'kd_tree', 'brute')) # Define parameter grid with all parameter ranges or options
knn_grid = GridSearchCV(knn, knn_param_grid, cv=10, scoring='accuracy') # Instantiate grid object for the classifier, specifiy n folds and performance measure
knn_grid.fit(X, labels) # fit model grid with data

# Plot accuracy vs. k - works only if other parameters are fixed          
#knn_mean_scores = [result.mean_validation_score for result in knn_grid.grid_scores_]
#plt.plot( np.arange(1, 10), knn_mean_scores)
#plt.show()

# Print results:
print('Best score           :', knn_grid.best_score_)
print('Best parameters found:', knn_grid.best_params_)
print(knn_grid.best_estimator_)

#%% Parameter tuning for Logistic Regression

# Find best parameters using GridSearchCV:
logreg = LogisticRegression()                                
logreg_param_grid = dict(penalty = ('l1','l2'), C = np.arange(0.1, 10, 0.1)) 
logreg_grid = GridSearchCV(logreg, logreg_param_grid, cv=10, scoring='accuracy') 
logreg_grid.fit(X, labels)                    

# Print results:
print('Best score           :', logreg_grid.best_score_)
print('Best parameters found:', logreg_grid.best_params_)
print(logreg_grid.best_estimator_)                        
                           
#%% Parameter tuning for Logistic Regression 

# Find best parameters using GridSearchCV:
supvm = svm.SVC() #l1 regularization approach appears to give better results.                                
supvm_param_grid = dict(kernel = ('linear','poly', 'rbf'), degree = np.arange(1,11), C = np.arange(0.1, 10, 0.1)) # Define parameter grid where all parameter ranges/options to test are specified
supvm_grid = GridSearchCV(supvm, supvm_param_grid, cv=10, scoring='accuracy') # Instantiate a grid object for the classifier, specifiy folds and performance measure
supvm_grid.fit(X, labels) #fit model grid with data                        

# Print results:
print('Best score           :', supvm_grid.best_score_)
print('Best parameters found:', supvm_grid.best_params_)
print(supvm_grid.best_estimator_)   