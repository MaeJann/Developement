import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
import itertools

from collections import Counter

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

X = scale(X)  # way to scale features... did not improved results so far
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

logreg.fit(X, labels)
log_reg_prob = logreg.predict_proba(X) # get propabilities in case of logistic regression!



#%%  
supvecm= svm.SVC(kernel='poly', degree=3)
svm_score = cross_val_score(supvecm, X, labels, cv = 10, scoring = 'accuracy')
print("SVM Accuracy:", np.mean(svm_score))
y_train_pred_svm = cross_val_predict(supvecm, X, labels, cv=10)
cm_svm = confusion_matrix(labels, y_train_pred_svm) 

#plt_conf(cm_svm, label_names, "Confusion Matrix for SVM")


#%%

def purity_map(predictions):
# Plotting Clustering results ("Purity Map")

    predictions = predictions.reshape((50,42))
    plt.imshow(predictions)
    plt.imshow(predictions)
    plt.colorbar()
    figure = plt.gca()
    figure.axes.get_xaxis().set_visible(False)
    figure.axes.get_yaxis().set_visible(False)


def purity_metric(predictions, labels, n_cluster):
    class_corr = []
    
    for i in range(n_cluster):
    # create array wich contains all label instances in a certain cluster:
        positions = predictions == i
        label_instances = labels[positions]
        
        # compute max value
        maxx,minn = max(label_instances),min(label_instances)
        c = Counter(label_instances)
        frequency = [c[i] for i in range(minn,maxx+1)]
        max_frequency = np.max(frequency)

        class_corr.append(max_frequency)
        
    purity = np.sum(class_corr)/len(labels)
    n_missed =  len(labels) - np.sum(class_corr)
    return purity, n_missed


#%%        

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


# Create GMM Classifier
clf_GMM = GaussianMixture(n_components = 6)
clf_GMM.fit(X)
GMM_results = clf_GMM.predict(X)

GMM_purity, GMM_missed = purity_metric(GMM_results, labels,6 )
print(GMM_purity)
print(GMM_missed)
purity_map(GMM_results)

clf_KM = KMeans(n_clusters = 6)
clf_KM.fit(X)
KM_results = clf_KM.predict(X)

KM_purity, KM_missed = purity_metric(KM_results, labels,6 )
print(KM_purity)
print(KM_missed)



#%%

