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
from sklearn_porter import Porter
import matplotlib.pyplot as plt
from sklearn.tree import _tree
import pickle

# =============================================================================
# Functions
# =============================================================================
def forest_to_code(rf, feature_names):
    tab="    "
    # takes a fitted decision tree and outputs a python function
    with open(CHOICE_FUNCTION_PATH, 'w') as the_file: 
        the_file.write('model choiceModel\n\n')
        the_file.write('import "MoBalance.gaml"\n\n')
        the_file.write('global{\n\n')
        the_file.write('action choose_mode_per_people(people p,float walk_time, float drive_time, float PT_time, float cycle_time, float walk_time_PT,float drive_time_PT)'+'{ \n')
        the_file.write(tab+'list probs<-[0.0,0.0,0.0,0.0];\n');        
        for i in range(len(rf)):
            the_file.write('// '+ 'Tree #'+str(i)+'\n')
            tree=rf[i]
            tree_ = tree.tree_
            feature_name = [
                feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
                for i in tree_.feature
            ]
            def recurse(node, depth):
                indent = tab * (depth+1)
                if tree_.feature[node] != _tree.TREE_UNDEFINED:
                    name = feature_name[node]
                    threshold = tree_.threshold[node]
    #                print ("{}if {} <= {}:".format(indent, name, threshold))
    #                the_file.write("%sif (%s <= %s) { \n"%(indent, name, str(threshold)))
                    the_file.write("%s if (%s <= %.2f) "%(indent, name, threshold)+"{ \n")
                    recurse(tree_.children_left[node], depth + 1)
                    the_file.write(indent+'}'+'\n')
    #                the_file.write("{}else \{  # if {} > {} \n".format(indent, name, threshold))
                    the_file.write(indent+'else {'+ "// if %s > %.2f \n"%(name, threshold))   
    #                print ("{}else:  # if {} > {}".format(indent, name, threshold))
                    recurse(tree_.children_right[node], depth + 1)
                    the_file.write(indent+'}'+'\n')
                else:
                    n_samples=sum([int(v) for v in tree_.value[node][0]])
    #                the_file.write("{}p.mode<-['car', 'bike', 'walk', 'PT'][rnd_choice({})];".format(indent, [round(v/n_samples,2) for v in tree_.value[node][0]])+"} \n")
                    the_file.write("{}list pred<-{}".format(indent, [round(v/n_samples,2) for v in tree_.value[node][0]])+"; \n")
                    the_file.write(indent+"loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }")
    #                print ("{}return {}".format(indent, [int(v) for v in tree_.value[node][0]]))
            recurse(0, 1)
        the_file.write(tab+"p.mode<-['car', 'bike', 'walk', 'PT'][rnd_choice(probs)];\n")
        the_file.write(tab+'}\n')
        the_file.write('}')
#********************************************
#      Constants
#********************************************
#MODE_TABLE_PATH='../../data/Boston/clean/main_modes.csv'
city='Boston'
MODE_TABLE_PATH='./'+city+'/clean/trip_modes.csv'
TOUR_TABLE_PATH='./'+city+'/clean/tours.csv'
CHOICE_FUNCTION_PATH='../ABM/models/choiceModel.gaml'
PICKLED_MODEL_PATH='./models/trip_mode_rf.p'
#********************************************
#      Data
#********************************************
mode_table=pd.read_csv(MODE_TABLE_PATH)
#to work with GAMA, rename all personal variables to p.name
agent_specific_vars=['age', 'hh_income', 'male', 'bachelor_degree', 'pop_per_sqmile_home']
mode_table=mode_table.rename(columns={v:'p.'+v for v in agent_specific_vars})
features=[c for c in mode_table.columns if not c=='mode']

X=mode_table[features]
y=mode_table['mode']

rf = RandomForestClassifier(random_state = 0,n_estimators =5, max_depth=4)

rf.fit(X, y)

importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
print("Feature ranking:")

for f in range(len(features)):
    print("%d. %s (%f)" % (f + 1, features[indices[f]], importances[indices[f]]))

# Plot the feature importances of the forest
#plt.figure(figsize=(16, 9))
#plt.title("Feature importances")
#plt.bar(range(len(features)), importances[indices],
#       color="r", yerr=std[indices], align="center")
#plt.xticks(range(len(features)), [features[i] for i in indices], rotation=90, fontsize=15)
#plt.xlim([-1, len(features)])
#plt.show()

#forest_to_code(rf.estimators_, features)

pickle.dump( rf, open( PICKLED_MODEL_PATH, "wb" ) )


         
                
