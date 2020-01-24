# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 12:49:18 2020

@author: Win
"""

import numpy as np
import matplotlib.pyplot as plt  
import pandas as pd 
import warnings
from sklearn import datasets
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
warnings.filterwarnings("ignore")


# import dataseta 
xls = pd.ExcelFile("./Dataset_modified.xlsx")
# mapiranje sheet-a u map 
sheet_to_df_map = {}
for sheet_name in xls.sheet_names:
    sheet_to_df_map[sheet_name] = xls.parse(sheet_name, skiprows=2, index_col=None)


print ('dolazi do skipa')
sheet_to_df_map['temp.'].fillna(0,inplace = True) # zamijenimo NaN sa nulama    
# uzimamo vrijednosti od 1 do 15 po kolonama
print ('dolazi do zamjene nan-a sa nulama')
X = sheet_to_df_map['temp.'].iloc[:, 2:15].values
print ('dolazi do pakovanja X-a')
# uzimamo prvih 15 redova
Y = sheet_to_df_map['temp.'].iloc[:, 14].values
print ('dolazi pakovanja Y-a')

for num in X:
    print(num)
print("KRAJ IKSA")
for num in Y:
    print(num)
print("KRAJ IPSILONA")
Xmultiplied = []
Ymultiplied = []
for num in X:
    Xmultiplied.append( num * 10 )

for iks in Xmultiplied:
    print(iks)
    print("KRAJ REDA")

for num in Y:
    Ymultiplied.append (num * 10 )
    
for iks in Ymultiplied:
    print(iks)
    print("KRAJ REDA IPSILON")
    
clf1 = LogisticRegression(random_state = 1)
clf2 = RandomForestClassifier(random_state = 1)
clf3 = GaussianNB()

print ('5-fold cross validation: \n')

labels = ['Logistic regression', 'Random forest', 'Naive Bayes']
print ('prolazi labele')
for clf, label in zip([clf1, clf2, clf3], labels):
    print ('upada u petlju')
    scores = model_selection.cross_val_score(clf, Xmultiplied, Ymultiplied, cv =2, scoring = 'accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
   
