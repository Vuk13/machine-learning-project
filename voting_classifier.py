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

xls = pd.ExcelFile("./Dataset.xlsx")

sheet_to_df_map = {}
for sheet_name in xls.sheet_names:
    sheet_to_df_map[sheet_name] = xls.parse(sheet_name, skiprows=2, index_col=None)
    
X = sheet_to_df_map['temp.'].iloc[:, 1:15].values
Y = sheet_to_df_map['temp.'].iloc[:, 15].values


   