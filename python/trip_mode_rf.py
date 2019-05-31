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
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import pickle
import json

#********************************************
#      Constants
#********************************************
city='Boston'
CITY_CDIVMSAR=11 #New England (ME, NH, VT, CT, MA, RI) MSA or CMSA of 1 million or more with heavy rail
REGION_CDIVMSARS=[11,12,13,14]

NHTS_PATH='NHTS/perpub.csv'
NHTS_TOUR_PATH='NHTS/tour17.csv'
NHTS_TRIP_PATH='NHTS/trippub.csv'
#MODE_TABLE_PATH='./'+city+'/clean/trip_modes.csv'
PICKLED_MODEL_PATH='./models/trip_mode_rf.p'
RF_FEATURES_LIST_PATH='./models/rf_features.json'


nhts_to_simple_mode={
        # map NHTS modes to a simpler list of modes
        # 0: drive, 1: cycle, 2: walk, 3: PT
        -7:-99,-8:-99,-9:-99,
        1:2,
        2:1,
        3:0,4:0,5:0,6:0,
        7:-99,
        8:0,9:0,
        10:3,11:3,12:3,13:3,14:3,15:3,16:3,
        17:0,18:0,
        19:3,20:3,
        97:-99}


# =============================================================================
# Functions
# =============================================================================

# Functions for recoding the NHTS variables to match the synth pop
def income_cat_nhts(row):
    if row['HHFAMINC'] >7: return "gt100"
    elif row['HHFAMINC'] >4: return "gt35-lt100"
    return "lt35"
    
def age_cat_nhts(row):
    if row['R_AGE_IMP' ]<= 19: return "19 and under"
    elif row['R_AGE_IMP' ] <= 35: return "20 to 35"
    elif row['R_AGE_IMP' ] <= 60: return "35 to 60"
    return "above 60"
    
def children_cat_nhts(row):
    if row['LIF_CYC']>2 and row['LIF_CYC']<9:
        return 'yes'
    return 'no'
    
def workers_cat_nhts(row):
    if row['WRKCOUNT'] >=2: return "two or more"
    elif row['WRKCOUNT'] == 1: return "one"
    return "none"  
  
def tenure_cat_nhts(row):
    if row['HOMEOWN'] ==1: return "owned"
    elif row['HOMEOWN'] ==2:  return "rented"
    return "other"

def sex_cat_nhts(row):
    if row['R_SEX_IMP'] == 1: return "male"
    return "female"    
    
def bach_degree_cat_nhts(row):
    if row['EDUC'] >=4: return "yes"
    return "no" 
    

#********************************************
#      Data
#********************************************
#mode_table=pd.read_csv(MODE_TABLE_PATH)
nhts_per=pd.read_csv(NHTS_PATH)
nhts_tour=pd.read_csv(NHTS_TOUR_PATH)
nhts_trip=pd.read_csv(NHTS_TRIP_PATH)


# add unique ids and merge some variables across the 3 tables
nhts_trip['uniquePersonId']=nhts_trip.apply(lambda row: str(row['HOUSEID'])+'_'+str(row['PERSONID']), axis=1)
nhts_per['uniquePersonId']=nhts_per.apply(lambda row: str(row['HOUSEID'])+'_'+str(row['PERSONID']), axis=1)
nhts_tour['uniquePersonId']=nhts_tour.apply(lambda row: str(row['HOUSEID'])+'_'+str(row['PERSONID']), axis=1)

nhts_tour=nhts_tour.merge(nhts_per[['HOUSEID', 'HH_CBSA']], on='HOUSEID', how='left')

tables={'trips': nhts_trip, 'persons': nhts_per, 'tours': nhts_tour}
# put tables in a dict so we can use a loop to avoid repetition
for t in ['trips', 'persons']:
# remove some records
    tables[t]=tables[t].loc[tables[t]['CDIVMSAR'].isin(REGION_CDIVMSARS)] 
    tables[t]=tables[t].loc[tables[t]['R_AGE_IMP']>15]
    tables[t]=tables[t].loc[tables[t]['R_AGE_IMP']>15]
    tables[t]['income']=tables[t].apply(lambda row: income_cat_nhts(row), axis=1)
    tables[t]['age']=tables[t].apply(lambda row: age_cat_nhts(row), axis=1)
    tables[t]['children']=tables[t].apply(lambda row: children_cat_nhts(row), axis=1)
    tables[t]['workers']=tables[t].apply(lambda row: workers_cat_nhts(row), axis=1)
    tables[t]['tenure']=tables[t].apply(lambda row: tenure_cat_nhts(row), axis=1)
    tables[t]['sex']=tables[t].apply(lambda row: sex_cat_nhts(row), axis=1)
    tables[t]['bach_degree']=tables[t].apply(lambda row: bach_degree_cat_nhts(row), axis=1)
    tables[t]=tables[t].rename(columns= {'HTPPOPDN': 'pop_per_sqmile_home'})

#with the tour file:
#    get the speed for each mode and the distance to walk/drive to transit for weach CBSA
#    we can use this to estimate the travel time for each potential mode in the trip file
speeds={c:{} for c in set(tables['persons']['HH_CBSA'])}
tables['tours']['main_mode']=tables['tours'].apply(lambda row: nhts_to_simple_mode[row['MODE_D']], axis=1)
for c in speeds:
    this_cbsa=tables['tours'][tables['tours']['HH_CBSA']==c]
    for m in [0,1,2, 3]:
        all_speeds=this_cbsa.loc[((this_cbsa['main_mode']==m) & (this_cbsa['TIME_M']>0))].apply(lambda row: row['DIST_M']/row['TIME_M'], axis=1)
        if len(all_speeds)>0:
            speeds[c]['km_per_minute_'+str(m)]=1.62* all_speeds.mean()
        else:
            speeds[c]['km_per_minute_'+str(m)]=float('nan')
    speeds[c]['walk_km_'+str(m)]=1.62*this_cbsa.loc[this_cbsa['main_mode']==3,'PMT_WALK'].mean()
    speeds[c]['drive_km_'+str(m)]=1.62*this_cbsa.loc[this_cbsa['main_mode']==3,'PMT_POV'].mean()


# with the trip table only
tables['trips']['network_dist_km']=tables['trips'].apply(lambda row: row['TRPMILES']/1.62, axis=1)
tables['trips']['mode']=tables['trips'].apply(lambda row: nhts_to_simple_mode[row['TRPTRANS']], axis=1) 
tables['trips']=tables['trips'].loc[tables['trips']['mode']>=0]
tables['trips'].loc[tables['trips']['TRPMILES']<0, 'TRPMILES']=0 # -9 for work-from-home

# create the mode choice table
mode_table=pd.DataFrame()
#    add the trip stats for each potential mode
mode_table['drive_time_minutes']=tables['trips'].apply(lambda row: row['network_dist_km']/speeds[row['HH_CBSA']]['km_per_minute_'+str(0)], axis=1)
mode_table['cycle_time_minutes']=tables['trips'].apply(lambda row: row['network_dist_km']/speeds[row['HH_CBSA']]['km_per_minute_'+str(1)], axis=1)
mode_table['walk_time_minutes']=tables['trips'].apply(lambda row: row['network_dist_km']/speeds[row['HH_CBSA']]['km_per_minute_'+str(2)], axis=1)
mode_table['PT_time_minutes']=tables['trips'].apply(lambda row: row['network_dist_km']/speeds[row['HH_CBSA']]['km_per_minute_'+str(3)], axis=1)
mode_table['walk_time_PT_minutes']=tables['trips'].apply(lambda row: speeds[row['HH_CBSA']]['walk_km_'+str(3)]/speeds[row['HH_CBSA']]['km_per_minute_'+str(2)], axis=1)
mode_table['drive_time_PT_minutes']=tables['trips'].apply(lambda row: speeds[row['HH_CBSA']]['drive_km_'+str(3)]/speeds[row['HH_CBSA']]['km_per_minute_'+str(0)], axis=1)

for col in ['income', 'age', 'children', 'workers', 'tenure', 'sex', 'bach_degree']:
    new_dummys=pd.get_dummies(tables['trips'][col], prefix=col)
    mode_table=pd.concat([mode_table, new_dummys],  axis=1)
 
for col in [ 'pop_per_sqmile_home', 'network_dist_km', 'mode']:
    mode_table[col]=tables['trips'][col]


# =============================================================================
# Fit Mode Choice Model
# =============================================================================
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
# f1-macro better where there are class imbalances as it 
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
       
                
