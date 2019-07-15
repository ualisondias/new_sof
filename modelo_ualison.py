# Define nยบ of samples
n_a = 1

print(__doc__)


# Code source: Gaรซl Varoquaux
#              Andreas Mรผller
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA




names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
   # GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

# preprocess dataset, split into training and test part
Accuracy = np.zeros((n_a, len(names)))
kappa = np.zeros((n_a, len(names)))
fscore = np.zeros((n_a, len(names)))
mse = np.zeros((n_a, len(names)))
stime = np.zeros((n_a, len(names)))
s = 'todos_juntos.csv'
X = np.genfromtxt(('X_' + s) , delimiter=',')    
y = np.genfromtxt( 'y_' + s, delimiter=',') 
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=.4, random_state=42)

#Loop into numbeer od samples
for i in range(Accuracy.shape[0]):
    k = 0
    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        start = time. time()
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        y_pred = clf.predict(X_test)
        end = time. time() 
        kappa[0,k]=cohen_kappa_score(y_test, y_pred)
        fscore[0,k]=f1_score(y_test, y_pred, average='macro')
        mse[0,k]=mean_squared_error(y_test, y_pred)        
        Accuracy[i,k] = (score*100)
        stime[0,k]=(end-start)
        k +=1
    print(i*100/n_a)
    print ((i*100/n_a), end="\r")
print(Accuracy.shape)

#Creating Matrix for Mean an Std. Derivatio

results = np.zeros((len(names),2))

#Calculinng Mean and Std. Derivation 
for i in range(len(names)):
    results[i,0] = round (np.mean(Accuracy[:,i]), 2 )
    results[i,1] = round (np.std(Accuracy[:,i]), 2)


results = pd.DataFrame(results, index = names, columns = ['Media','Desvio'])       
results.to_csv(("Class_result_" + s) )
print(results)


