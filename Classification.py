import pandas as pd
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split


#%% Loading the data

data = pd.DataFrame(pd.read_csv('C:/Users/j-mae/Desktop/Master Thesis/Image Data/Test Data/Merged Data/Naxos_Boudin_8/Extracted/All_Minerals.csv', header=None, sep = ';' ))  
data = data.as_matrix()
labels = data[:,3]

Features = pd.DataFrame(pd.read_csv('C:/Users/j-mae/Desktop/Master Thesis/Image Data/Test Data/Merged Data/Naxos_Boudin_8/Extracted/Featurematrix.csv', sep = ';' ))
Features = Features.as_matrix()
X = Features[:, 1:]



X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.4, random_state = 1) 

knn3 = KNeighborsClassifier(n_neighbors = 3)
knn3.fit(X_train, y_train)
y_pred = knn3.predict(X_test)
print()
print()
print("Classification Accuracy KNN: (K = 3):", metrics.accuracy_score(y_test, y_pred))

logreg = LogisticRegression(penalty='l1')  #l1 regularization approach appears to give better results.
logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)
print("Classification Accuracy Logistic Regression:", metrics.accuracy_score(y_test, y_pred))


supvecm= svm.SVC(kernel='poly', degree=3)
supvecm.fit(X_train,y_train)
y_pred = supvecm.predict(X_test)
print("Classification Supported Vector Machines:", metrics.accuracy_score(y_test, y_pred))