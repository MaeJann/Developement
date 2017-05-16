### Script to evaluate different classifiers in more detail
### Allows visualization of confusion matrix, learning curves & propability histograms
### Best hyper parameter should be determined beforehead using gridsearchcv

#%% Import modules & functions

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import scale
from sklearn.utils import shuffle

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
    plt.show()
    
    
# Function to plot propability distribution for a desired class
def plot_hist(labels, propabilities, classlabel): # labels = true labels, propabilities = predicted propabilities for all classes for all samples, classlabel = class to be analyzed
    
    props_all_samples = propabilities[:,classlabel-1] # get all probabilities for all samples for the desired class
    props_true_class = props_all_samples[labels == classlabel] # from that, extract only the samples that actually are part of the class
    
    # Visualize predicted propabilities for all samples of the class as a histogram:
    plt.hist(props_true_class) 
    plt.xlim(0,1)
    plt.title("Probability Histogram")
    plt.xlabel("Probability")
    plt.ylabel("Counts")
    thresh = 0.5 
    plt.axvline(x=thresh, color='k', linestyle='dotted', linewidth=1.5)
    plt.show()
    
    
# Function to plot learning curves, code orientd on example in scikit learn online documentation: http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html    
def plot_learningcurves(model, features, labels, fold, title):
    
    train_size, train_score, test_score = learning_curve(model, features, labels, cv = fold, train_sizes=np.linspace(.1, 1.0, 10), scoring = 'accuracy')
    
    train_score_mean = np.mean(train_score, axis=1)
    train_score_std = np.std(train_score, axis=1)
    test_score_mean = np.mean(test_score, axis=1)
    test_score_std = np.std(test_score, axis=1)
    
    plt.plot(train_size, train_score_mean, 'ro-', label = 'Training accuracy')
    plt.plot(train_size, test_score_mean, 'go-', label = 'Cross-validation accuracy')
    
    plt.fill_between(train_size, train_score_mean - train_score_std,
                     train_score_mean + train_score_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_size, test_score_mean - test_score_std,
                     test_score_mean + test_score_std, alpha=0.1, color="g")
    
    plt.xlabel("No. of training samples")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend(loc = 'lower right')
    plt.grid()
    plt.show()
    
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

#necessary in order to calculate learning curves: shuffle the data! (kind of a bug in scikit learn?) Set random state to ensure reproducible results!         
X, labels = shuffle(X, labels, random_state = 0) 

#%% Evaluation k-nearest neighbors

# Generate classifier and determine cros-validation score:
knn = KNeighborsClassifier(n_neighbors = 4) 
knn_score = cross_val_score(knn, X, labels, cv = 5, scoring = 'accuracy')

# Print validation results:
print('--- Evaluation Results for KNN:------------' )
print()
print("KNN cross-validation score:", np.mean(knn_score), '+- %.4f' % np.std(knn_score))

# Get predicted labels (cross-validated) and compute confusion matrix:
y_train_pred_knn = cross_val_predict(knn, X, labels, cv=5, n_jobs = -1)
cm_log = confusion_matrix(labels, y_train_pred_knn) 
plt_conf(cm_log, label_names, "Confusion Matrix for K nearest neighbors")

# Compute learning curve:
plot_learningcurves(knn, X, labels, 5, "Learning curve for K nearest neighbors")

#%% Evaluation Logistic Regression

# Generate classifier and determine cros-validation score
logreg = LogisticRegression(penalty = 'l1', C = 6)  
logreg_score = cross_val_score(logreg, X, labels, cv = 5, scoring = 'accuracy')

# Print validation results
print('--- Evaluation Results for Logistic Regression:------------' )
print()
print("Log. Reg. cross-validation score:", np.mean(logreg_score), '+- %.4f' % np.std(logreg_score))

# Get predicted labels (cross-validated) and compute confusion matrix:
y_train_pred_log = cross_val_predict(logreg, X, labels, cv=5)
cm_log = confusion_matrix(labels, y_train_pred_log) 
plt_conf(cm_log, label_names, "Confusion Matrix for Logistic Regression")

# Get propabilities of predictions and plot histograms for certain classes (eg. with many missclassifications):
log_reg_prob = cross_val_predict(logreg, X, labels, cv=5, method='predict_proba') 
plot_hist(labels, log_reg_prob, 2)
  
# Compute learning curve:                                                                    
plot_learningcurves(logreg, X, labels, 5, "Learning curve for Logistic Regression")

#%% Evaluation Support Vector Machines:

    
# Generate classifier and determine cros-validation score
supvecm= svm.SVC(kernel = 'poly', degree = 2, C = 6)
supvecm_score = cross_val_score(supvecm, X, labels, cv = 5, scoring = 'accuracy')

# Print validation results
print('--- Evaluation Results for SVM ------------' )
print()
print("SVM cross-validation score:", np.mean(supvecm_score), '+- %.4f' %  np.std(supvecm_score))
print()

# Get predicted labels (cross-validated) and compute confusion matrix:
y_train_pred_svm = cross_val_predict(supvecm, X, labels, cv=5)
cm_svm = confusion_matrix(labels, y_train_pred_svm) 

# Get propabilities of predictions and plot histograms for certain classes (eg. with many missclassifications):
plt_conf(cm_svm, label_names, "Confusion Matrix for SVM")

# Compute learning curve: 
plot_learningcurves(supvecm, X, labels, 5, "Learning curve for SVM")