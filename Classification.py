import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix


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

y_train_pred_knn = cross_val_predict(knn3, X, labels, cv=10)
print("Confusion Matrix for KNN (3):")
print(confusion_matrix(labels, y_train_pred_knn))
print()

#%%
logreg = LogisticRegression(penalty='l1')  #l1 regularization approach appears to give better results.
scores_logreg = cross_val_score(logreg, X, labels, cv = 10, scoring = 'accuracy')
print("Classification Accuracy Logistic Regression:", np.mean(scores_logreg))

y_train_pred_log = cross_val_predict(logreg, X, labels, cv=10)
print("Confusion Matrix for Logistic Regression:")
print(confusion_matrix(labels, y_train_pred_log))
print()

#%%
supvecm= svm.SVC(kernel='poly', degree=3)
scores_supvecm = cross_val_score(supvecm, X, labels, cv = 10, scoring = 'accuracy')
print("Classification Supported Vector Machines:", np.mean(scores_supvecm))


y_train_pred_svm = cross_val_predict(supvecm, X, labels, cv=10)
print("Confusion Matrix for SVM:")
print(confusion_matrix(labels, y_train_pred_svm))
print()