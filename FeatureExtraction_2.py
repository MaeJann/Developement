#%% Load Modules

import cv2
import numpy as np
import pandas as pd 

#%% Load Data


# --- Load coordinates and labels 
data = pd.DataFrame(pd.read_csv('C:/Users/j-mae/Desktop/Master Thesis/Image Data/Test Data/Merged Data/Naxos_Boudin_8/Extracted/Observations_Composite.csv', header=None, sep = ';' ))  
data = data.as_matrix()
x,y = data[:, 1], data[:, 2]
labels = data[:,3]

# --- define quadratic slide around coordinates 
extend = 20
x_min = x - extend 
x_max = x + extend
y_min = y - extend
y_max = y + extend

# --- Load maximum intensity images:
max_ppl = cv2.imread('C:/Users/j-mae/Desktop/Master Thesis/Image Data/Test Data/Merged Data/Naxos_Boudin_8/Images/plane/Naxos_Boudin_8_ppl_0.jpg')
max_xpl = cv2.imread('C:/Users/j-mae/Desktop/Master Thesis/Image Data/Test Data/Merged Data/Naxos_Boudin_8/Images/processed/MaxInt.jpg')


#%% Functions for feature extraction

# Extract average color value for each channel around the coordinates:
def extract_col_val(image, x_min,x_max, y_min, y_max):
    col_vals = np.zeros((len(x_min), 3), dtype="uint8")
    for i in range(len(x_min)):
        col_vals[i, 0:3] = np.array([np.mean(image[y_min[i]:y_max[i], x_min[i]:x_max[i],0]), np.mean(image[y_min[i]:y_max[i], x_min[i]:x_max[i], 1]), np.mean(image[y_min[i]:y_max[i], x_min[i]:x_max[i], 2])])   
    return col_vals

def comp_std(image, x_min,x_max, y_min, y_max):
    std_vals = np.zeros((len(x_min), 3), dtype="uint8")
    for i in range(len(x_min)):
        std_vals[i, 0:3] = np.array([np.std(image[y_min[i]:y_max[i], x_min[i]:x_max[i],0]), np.std(image[y_min[i]:y_max[i], x_min[i]:x_max[i], 1]), np.std(image[y_min[i]:y_max[i], x_min[i]:x_max[i], 2])])
    return std_vals



#%% extract individual features

PPL_Color = extract_col_val(max_ppl, x_min,x_max, y_min, y_max)
XPL_Color = extract_col_val(max_xpl, x_min,x_max, y_min, y_max)

PPL_std = comp_std(max_ppl, x_min,x_max, y_min, y_max)
XPL_std = comp_std(max_xpl, x_min,x_max, y_min, y_max)

#%% Combine extracted features to feature matrix

X = np.hstack((PPL_Color, XPL_Color, PPL_std, XPL_std))









#%% Quick Feature Test

from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X ,labels, test_size=0.4, random_state = 1) 

knn3 = KNeighborsClassifier(n_neighbors = 3)
knn3.fit(X_train, y_train)
y_pred = knn3.predict(X_test)
print()
print()
print("Classification Accuracy KNN: (K = 3):", int(metrics.accuracy_score(y_test, y_pred)*100), "%")

logreg = LogisticRegression()
logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)
print("Classification Accuracy Logistic Regression:", int(metrics.accuracy_score(y_test, y_pred)*100), "%")


supvecm= svm.SVC(kernel='poly', degree=3)
supvecm.fit(X_train,y_train)
y_pred = supvecm.predict(X_test)
print("Classification Supported Vector Machines:", int(metrics.accuracy_score(y_test, y_pred)*100), "%")

