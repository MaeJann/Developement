import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import itertools

#%% Function for plotting confusion matrix: Code to plot conf. matrix partly modified after http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

def plt_conf(cm, classnames, title, norm = "False"):

    if norm == True:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # Normalize confusion matrix
        
    plt.imshow(cm, interpolation = 'none', cmap = 'Blues')
    plt.colorbar()
    
    tickmarks = np.arange(len(classnames))
    plt.xticks(tickmarks, classnames, rotation=45)
    plt.yticks(tickmarks, classnames, rotation=45)
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    print(cm)
    
#%%
    
data = pd.DataFrame(pd.read_csv('C:/Users/j-mae/Desktop/Master Thesis/Image Data/Test Data/Merged Data/Naxos_Boudin_8/Extracted/All_Minerals.csv', header=None, sep = ';' ))  
data = data.as_matrix()
labels = data[:,3]
label_names = ['Cal','Fsp','Bt','Amp','Op']
    
Features = pd.DataFrame(pd.read_csv('C:/Users/j-mae/Desktop/Master Thesis/Image Data/Test Data/Merged Data/Naxos_Boudin_8/Extracted/Featurematrix.csv', sep = ';' ))
Features = Features.as_matrix()
X = Features[:, 1:]
 #%%

knn = KNeighborsClassifier(n_neighbors = 3)
knn_score = cross_val_score(knn, X, labels, cv = 10, scoring = 'accuracy')
print("kNN accuracy:", np.mean(knn_score))
y_train_pred_knn = cross_val_predict(knn, X, labels, cv=10)
cm_knn = confusion_matrix(labels, y_train_pred_knn) 
 
#%%   
logreg = LogisticRegression(penalty='l1')  #l1 regularization approach appears to give better results.
logreg_score = cross_val_score(logreg, X, labels, cv = 10, scoring = 'accuracy')
print("Logistic Regression accuracy:", np.mean(logreg_score))
y_train_pred_log = cross_val_predict(logreg, X, labels, cv=10)
cm_log = confusion_matrix(labels, y_train_pred_log) 

#%%  
supvecm= svm.SVC(kernel='poly', degree=3)
svm_score = cross_val_score(supvecm, X, labels, cv = 10, scoring = 'accuracy')
print("SVM Accuracy:", np.mean(svm_score))
y_train_pred_svm = cross_val_predict(supvecm, X, labels, cv=10)
cm_svm = confusion_matrix(labels, y_train_pred_svm) 

plt_conf(cm_svm, label_names, "Confusion Matrix for SVM")


