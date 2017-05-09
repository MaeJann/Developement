import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score


#%% Loading the data

data = pd.DataFrame(pd.read_csv('C:/Users/j-mae/Desktop/Master Thesis/Image Data/Test Data/Merged Data/Naxos_Boudin_8/Extracted/All_Minerals.csv', header=None, sep = ';' ))  
data = data.as_matrix()
labels = data[:,3]

Features = pd.DataFrame(pd.read_csv('C:/Users/j-mae/Desktop/Master Thesis/Image Data/Test Data/Merged Data/Naxos_Boudin_8/Extracted/Featurematrix.csv', sep = ';' ))
Features = Features.as_matrix()
X = Features[:, 1:]


#%%
knn3 = KNeighborsClassifier(n_neighbors = 3)
scores_knn3 = cross_val_score(knn3, X, labels, cv = 10, scoring = 'accuracy')
print("Classification Accuracy KNN: (K = 3):", np.mean(scores_knn3))

#%%
logreg = LogisticRegression(penalty='l1')  #l1 regularization approach appears to give better results.
scores_logreg = cross_val_score(logreg, X, labels, cv = 10, scoring = 'accuracy')
print("Classification Accuracy Logistic Regression:", np.mean(scores_logreg))

#%%
supvecm= svm.SVC(kernel='poly', degree=3)
scores_supvecm = cross_val_score(supvecm, X, labels, cv = 10, scoring = 'accuracy')
print("Classification Supported Vector Machines:", np.mean(scores_supvecm))