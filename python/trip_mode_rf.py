#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 12:24:53 2019
    input data must contain:
    'drive_time', 'cycle_time', 'walk_time', 'PT_time', 'walk_time_PT',
    'drive_cost','cycle_cost', 'walk_cost','PT_cost', 'main_mode'
    Other columns are treated as individual-specific variables
@author: doorleyr
""" 
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, make_scorer, classification_report
import matplotlib.pyplot as plt
import pickle
import json

#********************************************
#      Constants
#********************************************
city='Boston'
MODE_TABLE_PATH='./'+city+'/clean/trip_modes.csv'
PICKLED_MODEL_PATH='./models/trip_mode_rf.p'
RF_FEATURES_LIST_PATH='./models/rf_features.json'


# =============================================================================
# Functions
# =============================================================================
def weightedScore(y, y_pred):
    accuracies=[]
    for cat in set(y):
        accuracy=[y_i==cat&(~y_pred_i==cat) for y_i, y_pred_i in zip(y, y_pred)]
    cost=np.mean(accuracy)
    return -cost

custom_loss=make_scorer(weightedScore, greater_is_better=True)

#********************************************
#      Data
#********************************************
mode_table=pd.read_csv(MODE_TABLE_PATH)
features=[c for c in mode_table.columns if not c=='mode']

X=mode_table[features]
y=mode_table['mode']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


rf = RandomForestClassifier(n_estimators =32, random_state=0, class_weight='balanced')
# Test different values of the hyper-parameters:
# 'max_features','max_depth','min_samples_split' and 'min_samples_leaf'

# Create the parameter ranges
maxDepth = list(range(5,100,5)) # Maximum depth of tree
maxDepth.append(None)
minSamplesSplit = range(2,42,5) # Minimum samples required to split a node
minSamplesLeaf = range(1,101,10) # Minimum samples required at each leaf node

#Create the grid
randomGrid = {
               'max_depth': maxDepth,
               'min_samples_split': minSamplesSplit,
               'min_samples_leaf': minSamplesLeaf}

# Create the random search object
rfRandom = RandomizedSearchCV(estimator = rf, param_distributions = randomGrid,
                               n_iter = 512, cv = 5, verbose=1, random_state=0, 
                               refit=True, scoring='f1_macro', n_jobs=-1)
# f1-macro better where there class imbalances as it 
# computes f1 for each class and then takes an unweighted mean
# "In problems where infrequent classes are nonetheless important, 
# macro-averaging may be a means of highlighting their performance."

# Perform the random search and find the best parameter set
rfRandom.fit(X_train, y_train)
rfWinner=rfRandom.best_estimator_
bestParams=rfRandom.best_params_
#forest_to_code(rf.estimators_, features)


importances = rfWinner.feature_importances_
std = np.std([tree.feature_importances_ for tree in rfWinner.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
print("Feature ranking:")

for f in range(len(features)):
    print("%d. %s (%f)" % (f + 1, features[indices[f]], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure(figsize=(16, 9))
plt.title("Feature importances")
plt.bar(range(len(features)), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(len(features)), [features[i] for i in indices], rotation=45, fontsize=15)
plt.xlim([-1, len(features)])
plt.show()


predicted=rfWinner.predict(X_test)
conf_mat=confusion_matrix(y_test, predicted)
print(conf_mat)
# rows are true labels and coluns are predicted labels
# Cij  is equal to the number of observations 
# known to be in group i but predicted to be in group j.
for i in range(len(conf_mat)):
    print('Total True for Class '+str(i)+': '+str(sum(conf_mat[i])))
    print('Total Predicted for Class '+str(i)+': '+str(sum([p[i] for p in conf_mat])))
pickle.dump( rfWinner, open( PICKLED_MODEL_PATH, "wb" ) )
json.dump(features,open(RF_FEATURES_LIST_PATH, 'w' ))

         
                
