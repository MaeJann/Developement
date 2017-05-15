#%% Import of Modules

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

#%% Loading the Data
    
data = pd.DataFrame(pd.read_csv('C:/Users/j-mae/Desktop/Master Thesis/Image Data/Test Data/Merged Data/Naxos_Boudin_8/Extracted/All_Minerals.csv', header=None, sep = ';' ))  
data = data.as_matrix()
labels = data[:,3]
label_names = ['Cal','Fsp','Bt','Amp','Op']
    
Features = pd.DataFrame(pd.read_csv('C:/Users/j-mae/Desktop/Master Thesis/Image Data/Test Data/Merged Data/Naxos_Boudin_8/Extracted/Featurematrix.csv', sep = ';' ))
Features = Features.as_matrix()
X = Features[:, 1:]

X = scale(X)  # scale features 


#%% Define functions to evaluate and visualize clustering results

def purity_map(predictions, label_classes, n_samples):
# Plotting Clustering results ("Purity Map"):
    
    
    # Sort clustering results for each label class i:
    for i in range(label_classes):
        predictions [i*n_samples: i*n_samples + n_samples+1] = np.sort(predictions [i*n_samples: i*n_samples + n_samples+1])
    
    # Reshape it to map format:
    new_shape = (label_classes * 10, n_samples/10) # = (50,42) due to 5 classes with 420 samples each)
    predictions = predictions.reshape(new_shape) 
    
    
    # Visualization:
    plt.imshow(predictions)
    plt.colorbar()
    figure = plt.gca()
    figure.axes.get_xaxis().set_visible(False)
    figure.axes.get_yaxis().set_visible(False)
    plt.savefig('purity.jpg')
    plt.show()

    

def purity_metric(predictions, labels, n_cluster):
    class_corr = []
    
    for i in range(n_cluster):
    
        positions = predictions == i            # get positions of samples in a certain cluster i 
        label_instances = labels[positions]     # extract labels of these samples within cluster i
        
        # from list of sample_labels occuring in cluster i: find maximum frequency:
        maxx,minn = max(label_instances),min(label_instances)
        c = Counter(label_instances)
        frequency = [c[i] for i in range(minn,maxx+1)]
        max_frequency = np.max(frequency)

        class_corr.append(max_frequency)   # store maximum frequency in array containing all "correctly" classified samples
        
    purity = np.sum(class_corr)/len(labels) # compute purity by deviding number of correctly classified samples by N
    n_missed =  len(labels) - np.sum(class_corr) # calculate number of missclassified samples
    return purity, n_missed


#%% Create simple GMM and KM Clusterers and apply to data       

clf_KM = KMeans(n_clusters = 6)
clf_KM.fit(X)
KM_results = clf_KM.predict(X)

KM_purity, KM_missed = purity_metric(KM_results, labels, 6)
print('KMeans Purity:',KM_purity)
print('No. of false classifications',KM_missed)
purity_map(KM_results, 5, 420)


clf_GMM = GaussianMixture(n_components = 6)
clf_GMM.fit(X)
GMM_results = clf_GMM.predict(X)

GMM_purity, GMM_missed = purity_metric(GMM_results, labels,6)
print('GMM Purity:', GMM_purity)
print('No. of false classifications', GMM_missed)
purity_map(GMM_results, 5, 420)

