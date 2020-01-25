# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 12:49:18 2020

@author: Win
"""

import numpy as np
import matplotlib.pyplot as plt  
import pandas as pd 
import warnings
import matplotlib.gridspec as gridspec
import itertools
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
warnings.filterwarnings("ignore")


# import dataseta 
xls = pd.ExcelFile("./Dataset_modified.xlsx")
# mapiranje sheet-a u map 
sheet_to_df_map = {}
for sheet_name in xls.sheet_names:
    sheet_to_df_map[sheet_name] = xls.parse(sheet_name, skiprows=2, index_col=None)


print ('dolazi do skipa')
sheet_to_df_map['rel.vlaznost'].fillna(0, inplace = True ) # zamijenimo NaN sa nulama    
# uzimamo svaku trecu godinu za podatke
print ('dolazi do zamjene nan-a sa nulama')
X = sheet_to_df_map['rel.vlaznost'].iloc[:, 2:15].values
print ('dolazi do pakovanja X-a')
# uzimamo prvih 15 redova
Y = sheet_to_df_map['rel.vlaznost'].iloc[:, 14].values
print ('dolazi pakovanja Y-a')

for num in X:
    print(num)
print("KRAJ IKSA")
for num in Y:
    print(num)
print("KRAJ IPSILONA")
#Xmultiplied = []
#Ymultiplied = []
#for num in X:
#    Xmultiplied.append( num * 10 )

#for iks in Xmultiplied:
#    print(iks)
#    print("KRAJ REDA")

#for num in Y:
#    Ymultiplied.append (num * 10 )
    
#for iks in Ymultiplied:
#    print(iks)
#    print("KRAJ REDA IPSILON")
    
clf1 = LogisticRegression(random_state = 1)
clf2 = RandomForestClassifier(random_state = 1)
clf3 = GaussianNB()
clf4 = MultinomialNB()
print ('2-fold cross validation: \n')

labels = ['Logistic regression', 'Random forest', 'Gausian Naive Bayes', 'Multinomial Naive Bayes']
print ('prolazi labele')
for clf, label in zip([clf1, clf2, clf3, clf4], labels):
    print ('upada u petlju')
    scores = model_selection.cross_val_score(clf, X, Y, cv = 2, scoring = 'accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
# std standard deviation along the specified axis
#hard voting algoritam   
voting_clf_hard = VotingClassifier(estimators = [(labels[0], clf1),
                                                 (labels[1], clf2),
                                                 (labels[2],clf3)], voting = 'hard')
#soft voting algoritam
voting_clf_soft = VotingClassifier(estimators = [(labels[0], clf1),
                                                 (labels[1], clf2),
                                                 (labels[2],clf3)], voting = 'soft')

labels_new = ['Logistic Regression', 'Random forest', 'Naive bayes', 'Voting Classifier Hard', 'Voting Classifier Soft']
for (clf, label) in zip([clf1,clf2,clf3, voting_clf_hard,voting_clf_soft], labels_new):
    scores = model_selection.cross_val_score(clf, X, Y, cv = 2, scoring = 'accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
clf1.fit(X, Y)
clf2.fit(X, Y)
clf3.fit(X, Y)
clf4.fit(X, Y)

XT = X[:20]
print("stize do prije plott-a")
plt.figure()
print("stize posle figure")
plt.plot(clf1.predict(XT), 'gd' , label = 'LogisticRegression')
plt.plot(clf2.predict(XT), 'b^' , label = 'RandomForest')
plt.plot(clf3.predict(XT), 'ys' , label = 'GausianNB')
plt.plot(clf4.predict(XT), 'r*' , label = 'MultinomialNB')
plt.tick_params(axis='x', which='both', bottom=False, top=False,
                labelbottom=False)
plt.ylabel('predicted')
plt.xlabel('training samples')
plt.legend(loc="best")
plt.title('Poredjivanje individualnih predikcija sa prosjekom')
plt.show()