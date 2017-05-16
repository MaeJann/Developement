import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
import itertools

from sklearn.grid_search import GridSearchCV


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

def plot_hist(data1, data2):
    plt.hist(data1)
    plt.hist(data2)
    plt.xlim(0,1)
    plt.title("Probability Histogram")
    plt.xlabel("Probability")
    plt.ylabel("Counts")
    thresh = 0.5
    plt.axvline(x=thresh, color='k', linestyle='dotted', linewidth=1.5)
#%%
    
data = pd.DataFrame(pd.read_csv('C:/Users/j-mae/Desktop/Master Thesis/Image Data/Test Data/Merged Data/Naxos_Boudin_8/Extracted/All_Minerals.csv', header=None, sep = ';' ))  
data = data.as_matrix()
labels = data[:,3]
label_names = ['Cal','Fsp','Bt','Amp','Op']
    
Features = pd.DataFrame(pd.read_csv('C:/Users/j-mae/Desktop/Master Thesis/Image Data/Test Data/Merged Data/Naxos_Boudin_8/Extracted/Featurematrix.csv', sep = ';' ))
Features = Features.as_matrix()
X = Features[:, 1:]
#X = minmax_scale(X)
#X = scale(X)  # way to scale features... did not improved results so far
 #%% Parameter tuning and testing for KNN


knn = KNeighborsClassifier() # Instantiate t classifier with default parameter                                   
knn_param_grid = dict(n_neighbors = np.arange(1, 5), weights = ('uniform', 'distance'), p = (1,2), algorithm = ('auto', 'ball_tree', 'kd_tree', 'brute')) # Define parameter grid where all parameter ranges/options to test are specified
knn_grid = GridSearchCV(knn, knn_param_grid, cv=10, scoring='accuracy') # Instantiate a grid object for the classifier, specifiy folds and performance measure
knn_grid.fit(X, labels) #fit model grid with data

#Plotting accuracy vs. k          
#knn_mean_scores = [result.mean_validation_score for result in knn_grid.grid_scores_]
#plt.plot( np.arange(1, 100), knn_mean_scores)
#plt.show()



# Print Results:
print('Best score           :', knn_grid.best_score_)
print('Best parameters found:', knn_grid.best_params_)
print(knn_grid.best_estimator_)


 
#%%   

logreg = LogisticRegression()                                
#logreg_param_grid = dict(penalty = ('l1','l2'), C = np.arange(0.1, 10, 0.1)) 
#logreg_grid = GridSearchCV(logreg, logreg_param_grid, cv=10, scoring='accuracy') 
#logreg_grid.fit(X, labels)                    
#   
#print('Best score           :', logreg_grid.best_score_)
#print('Best parameters found:', logreg_grid.best_params_)
#print(logreg_grid.best_estimator_)                        
                           
      
      
logreg_score = cross_val_score(logreg, X, labels, cv = 10, scoring = 'accuracy')
print("Logistic Regression accuracy:", np.mean(logreg_score))
y_train_pred_log = cross_val_predict(logreg, X, labels, cv=10, n_jobs = -1)
cm_log = confusion_matrix(labels, y_train_pred_log) 

logreg.fit(X, labels)
log_reg_prob = logreg.predict_proba(X) # get propabilities in case of logistic regression!



#%%  


#supvm = svm.SVC() #l1 regularization approach appears to give better results.                                
#supvm_param_grid = dict(kernel = ('linear','poly', 'rbf'), degree = np.arange(1,11), C = np.arange(0.1, 10, 0.1)) # Define parameter grid where all parameter ranges/options to test are specified
#supvm_grid = GridSearchCV(supvm, supvm_param_grid, cv=10, scoring='accuracy') # Instantiate a grid object for the classifier, specifiy folds and performance measure
#supvm_grid.fit(X, labels) #fit model grid with data                        
#   
#print('Best score           :', supvm_grid.best_score_)
#print('Best parameters found:', supvm_grid.best_params_)
#print(supvm_grid.best_estimator_)   



supvecm= svm.SVC(kernel='poly', degree=2)
svm_score = cross_val_score(supvecm, X, labels, cv = 10, scoring = 'accuracy')
print("SVM Accuracy:", np.mean(svm_score))
y_train_pred_svm = cross_val_predict(supvecm, X, labels, cv=10)
cm_svm = confusion_matrix(labels, y_train_pred_svm) 

plt_conf(cm_svm, label_names, "Confusion Matrix for SVM")

#%%

