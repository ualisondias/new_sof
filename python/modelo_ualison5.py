#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 21:44:17 2020

@author: dias
"""

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
import scipy.io as sio
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


n=0
names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
             "Naive Bayes", "QDA"]


results2 = np.zeros((len(names),5))
results3 = np.zeros((len(names),5))

# preprocess dataset, split into training and test part
Accuracy_treino = np.zeros((33,5))
Accuracy_teste = np.zeros((33,5))
kappa_treino = np.zeros((33,5))
kappa_teste = np.zeros((33,5))
fscore_treino = np.zeros((33,5))
fscore_teste = np.zeros((33,5))
mse_treino = np.zeros((33,5))
mse_teste = np.zeros((33,5))
stime = np.zeros((33,5))

dados = []
dados2 = []
for n in range(33):
    n=n+1
    for j in range(5):  
        j=j+1
        classifiers = [
            KNeighborsClassifier(3),
            #SVC(kernel="linear", C=0.025),
            #SVC(gamma=2, C=1),
           ## GaussianProcessClassifier(1.0 * RBF(1.0)),
#            DecisionTreeClassifier(max_depth=5),
#            RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
#            MLPClassifier(alpha=1, max_iter=1000),
#            AdaBoostClassifier(),
#            GaussianNB(),
#            QuadraticDiscriminantAnalysis()
                ]
        
        s = 'todos_juntos' +str(n) +'_' + str(j) +'.csv'
        X = np.genfromtxt(('X_' + s) , delimiter=',')    
        y = np.genfromtxt( 'y_' + s, delimiter=',') 
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=.2, random_state=42)
        k = 0    
            
        
        #Loop into numbeer od samples

      
         
        # iterate over classifiers
        clf=GaussianProcessClassifier(1.0 * RBF(1.0))
        start = time. time()
        clf.fit(X_train, y_train)
        score_treino = clf.score(X_train, y_train)
        score_teste = clf.score(X_test, y_test)
        y_pred_treino = clf.predict(X_train)
        y_pred_teste = clf.predict(X_test)              
        
        end = time. time() 
        kappa_treino[n-1,j-1]=cohen_kappa_score(y_train, y_pred_treino)
        kappa_teste[n-1,j-1]=cohen_kappa_score(y_test, y_pred_teste)
        fscore_treino[n-1,j-1]=f1_score(y_train, y_pred_treino, average='macro')
        fscore_teste[n-1,j-1]=f1_score(y_test, y_pred_teste, average='macro')
        mse_treino[n-1,j-1]=mean_squared_error(y_train, y_pred_treino)
        mse_teste[n-1,j-1]=mean_squared_error(y_test, y_pred_teste)
#                score=np.random.random_sample()
        Accuracy_treino[n-1,j-1] = (score_treino*100)
        Accuracy_teste[n-1,j-1] = (score_teste*100)


        k +=1
  
        #Calculinng Mean and Std. Derivation 
    
                
        
        
        
#        results = pd.DataFrame(results, index = names, columns = ['Media','Desvio'])       
#        results.to_csv(("Class_result_" + s) )
#    #        print(results)   
#    dados.append(results2)
#    results2 = np.zeros((len(names),5))
#  

        
        

