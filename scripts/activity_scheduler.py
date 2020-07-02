#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 13:20:55 2020

@author: doorleyr
"""
import pandas as pd
import random
import json
import numpy as np
import pickle
from transport_network import get_haversine_distance


class Activity():
    def __init__(self, activity_id, start_time, activity_name, location):
        self.activity_id=activity_id
        self.location=location
        self.name=activity_name
        self.start_time=start_time
    
class ActivityScheduler():
    
    def find_locations_for_activities(self, model):
        self.potential_locs={}
        for a_id in self.activity_names:
            activity_name=self.activity_names[a_id]
            self.potential_locs[activity_name]=model.geogrid.find_locations_for_activity(activity_name)
    
    def __init__(self, model):
        ACTIVITY_NAME_PATH='./cities/'+model.city_folder+'/mappings/activities.json'
        self.activity_names=json.load(open(ACTIVITY_NAME_PATH))
        MOTIF_SAMPLE_PATH='./cities/'+model.city_folder+'/clean/motif_samples.csv'
        self.sample_motifs=pd.read_csv(MOTIF_SAMPLE_PATH)
        self.training_data_attributes_path='./cities/'+model.city_folder+'/clean/person_sched_weekday.csv'
        self.training_data_profiles_path='./cities/'+model.city_folder+'/clean/profile_labels.csv'
        self.find_locations_for_activities(model)
        self.model_path='./cities/'+model.city_folder+'/models/profile_rf.p'
        self.model_features_path='./cities/'+model.city_folder+'/models/profile_rf_features.json'
        self.load_profile_rf()
        
    def assign_profiles(self, persons):
        persons_list=[]
        for ip, person in enumerate(persons):
            persons_list.append({'income': person.income,
                                 'age': person.age,
                                 'cars': person.cars,
                                 'male': person.sex=='male',
                                 'bach_degree':person.bach_degree=='yes',
                                 'worker': person.worker})
        feature_df=pd.DataFrame(persons_list)
        for feat in ['income', 'age', 'cars']:
            new_dummys=pd.get_dummies(feature_df[feat], prefix=feat)
            feature_df=pd.concat([feature_df, new_dummys],  axis=1)
        for rff in self.profile_rf_features:
            if rff not in feature_df.columns:
                feature_df[rff]=False
        feature_df=feature_df[self.profile_rf_features]
        chosen_profiles=list(self.profile_rf.predict(feature_df))
        for ip in range(len(chosen_profiles)):
            persons[ip].assign_motif(chosen_profiles[ip])

        
    def sample_activity_schedules(self, person, model):
        last_loc=person.home_loc
        activities=[]
        motif_options=self.sample_motifs.loc[self.sample_motifs['cluster']==person.motif].to_dict(orient='records')
        motif=random.choice(motif_options)
        hourly_activity_ids=[motif['P{}'.format(str(period).zfill(3))] for period in range(24)]
#        print(hourly_activity_ids)
        for t, a_id in enumerate(hourly_activity_ids):
            if ((t == 0) or (a_id != hourly_activity_ids[t-1])):
                activity_start_time=t*3600+ np.random.randint(3600)
                activity_name=self.activity_names[a_id]
#                print(activity_name)
                if activity_name=='Home':
                    activity_location=person.home_loc
                elif activity_name=='Work':
                    activity_location=person.work_loc
                else:
                    potential_locations=self.potential_locs[activity_name]
                    if len(potential_locations)==0:
#                        print('No locations for {} in geogrid'.format(activity_name))
                        potential_locations=[z for z in model.zones if not z.in_sim_area]
                    dist=[get_haversine_distance(last_loc.centroid, loc.centroid
                                                 )for loc in potential_locations]
#                    prob, chosen_idx =self.huff_model(dist, beta=1, predict_y=True, topN=5)
#                    chosen_idx=chosen_idx[0]
                    chosen_idx=np.random.choice(range(len(dist)))
                    activity_location=potential_locations[chosen_idx]
                activities.append(Activity(activity_id=a_id, 
                                           start_time=activity_start_time,
                                           activity_name=activity_name,
                                           location=activity_location))
                last_loc=activity_location
        person.assign_activities(activities)
        
    def load_profile_rf(self):
        try:
            self.profile_rf=pickle.load(open(self.model_path, 'rb'))
            self.profile_rf_features=json.load(open(self.model_features_path))
        except:
            self.train_profile_rf()    
        
    def train_profile_rf(self):
        print('Training Profile RF')
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split, RandomizedSearchCV 
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
        attributes_df=pd.read_csv(self.training_data_attributes_path)
        profiles_df=pd.read_csv(self.training_data_profiles_path)
        feature_dummys_df=pd.DataFrame()
        for col in ['income', 'age',  'cars']:
            new_dummys=pd.get_dummies(attributes_df[col], prefix=col)
            feature_dummys_df=pd.concat([feature_dummys_df, new_dummys],  axis=1)
        feature_dummys_df['male']=attributes_df['sex']=='male'
        feature_dummys_df['bach_degree']=attributes_df['bach_degree']=='yes'
        feature_dummys_df['worker']=attributes_df['worker']
        features=list(feature_dummys_df.columns)
        X=np.array(feature_dummys_df[features])
        y=np.array(profiles_df['cluster'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        
        rf = RandomForestClassifier(n_estimators =64, random_state=0, class_weight='balanced')
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
                                       n_iter = 128, cv = 5, verbose=1, random_state=0, 
                                       refit=True, scoring='f1_macro', n_jobs=-1)
        
        # Perform the random search and find the best parameter set
        rfRandom.fit(X_train, y_train)
        rfWinner=rfRandom.best_estimator_
        bestParams=rfRandom.best_params_
        
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
        plt.xticks(range(len(features)), [features[i] for i in indices], rotation=90, fontsize=15)
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
        pickle.dump( rfWinner, open( self.model_path, "wb" ) )
        json.dump(features,open(self.model_features_path, 'w' )) 
        self.profile_rf=rfWinner
        self.profile_rf_features=features
        
    def huff_model(self, dist, attract=None, alt_names=None, alpha=1, beta=2, predict_y=False, topN=None):
        """ 
        takes a distance matrix and a optional attraction matrix, calculates choice probabilities 
        and predicts choice outcomes by sampleing according to probabilities
        prob = (attract**alpha / dist**beta) / sum_over_all_alternatives(attract**alpha / dist**beta)
        
        Arguments:
        --------------------------------------------
        dist: distance matrix, ncs(number of choice situations) * nalt(number of alternatives), or 1-d array
        attract: optional attraction matrix, ncs * nalt, or 1-d array
        alt_names: optional matrix of alternative names, ncs * nalt, or 1-d array
        alpha, beta: coefficents of attraction and distance
        predict_y: whether or not to predict choice outcomes via sampling
        topN: when predicting choice outcomes, only alternatives with top N probabilities will be considered
        """
        dist = np.array(dist)
        dist = np.maximum(dist, np.ones_like(dist)*0.01)    # avoid dist=0
        if attract is None:
            attract = np.ones_like(dist)
        else:
            attract = np.array(attract)
        if dist.ndim == 1:
            dist = dist.reshape(1, -1)
            attract = attract.reshape(1, -1)
            if alt_names is not None:
                alt_names = alt_names.reshape(1, -1)
        ncs, nalt = dist.shape
        u = (attract ** alpha) / (dist ** beta)
        prob = u / u.sum(axis=1, keepdims=True)
        if predict_y:
            y = []
            if topN:
                use_prob = -np.sort(-prob, axis=1)[:, :topN]
                use_prob = use_prob / use_prob.sum(axis=1, keepdims=True)
                use_idx = np.argsort(-prob, axis=1)[:, :topN]
                if alt_names is None:
                    use_names = use_idx
                else:
                    use_names = np.asarray([alt_names[i, use_idx[i,:]] for i in range(ncs)])
            else:
                use_prob = prob
                if alt_names is None:
                    use_names = np.asarray([list(range(nalt)) for i in range(ncs)])
                else:
                    use_names = alt_names
            for i in range(ncs):
                this_y = np.random.choice(use_names[i, :], p=use_prob[i, :])
                y.append(this_y) 
        else:
            y = None
        return prob, y        
              
