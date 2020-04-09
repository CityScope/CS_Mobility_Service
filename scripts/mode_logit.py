#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
""" 
import pandas as pd
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import pickle
import json
import pylogit as pl
from collections import OrderedDict

#********************************************
#      Constants
#********************************************
city='Detroit'
REGION_CDIVMSARS_BY_CITY={"Boston": [11,21] ,
                           "Detroit": [31, 32]
                           }
region_cdivsmars=REGION_CDIVMSARS_BY_CITY[city]

NHTS_PATH='scripts/NHTS/perpub.csv'
NHTS_TOUR_PATH='scripts/NHTS/tour17.csv'
NHTS_TRIP_PATH='scripts/NHTS/trippub.csv'
#MODE_TABLE_PATH='./'+city+'/clean/trip_modes.csv'
PICKLED_MODEL_PATH='./scripts/cities/'+city+'/models/trip_mode_logit.p'
LOGIT_FEATURES_LIST_PATH='./scripts/cities/'+city+'/models/logit_features.json'
PERSON_SCHED_TABLE_PATH='./scripts/cities/'+city+'/clean/person_sched_weekday.csv'

# =============================================================================
# Functions
# =============================================================================

def run_time(func):
    def wrapper(*args, **kw):
        t1 = time.time()
        res = func(*args, **kw)
        t2 = time.time()
        print('{:4.4f} secodns elasped for {}'.format(t2-t1, func.__name__))
        return res
    return wrapper

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

def cars_cat_nhts(row):
    if row['HHVEHCNT'] == 0: return "none"
    elif row['HHVEHCNT'] == 1: return "one"
    return "two or more"

def race_cat_nhts(row):
    if row['R_RACE'] == 1: return "white"
    elif row['R_RACE'] == 2: return "black"
    elif row['R_RACE'] == 3: return "asian"
    return "other"
    
def purpose_cat_nhts(row):
    if row['TRIPPURP'] == 'HBW': return "HBW"
    elif row['TRIPPURP'] == "NHB": return "NHB"
    return "HBO"
    
def mode_cat(nhts_mode):
    # map NHTS modes to a simpler list of modes
    if nhts_mode in [3,4,5,6,8,9,17,18]:
        return 0 # drive
    elif nhts_mode ==2:
        return 1 # cycle
    elif nhts_mode ==1:
        return 2 # walk
    elif nhts_mode in [10,11,12,13,14,15,16, 19,20]:
        return 3 # PT
    else:
        return -99

def get_main_mode(row):
    if row['WRKTRANS']>0:
        return mode_cat(row['WRKTRANS'])
    elif row['SCHTRN1']>0:
        return mode_cat(row['SCHTRN1'])
    elif row['SCHTRN2']>0:
        return mode_cat(row['SCHTRN2'])
    else: return -99
    
def get_main_dist_km(row):
    if row['WRKTRANS']>0:
        return row['DISTTOWK17']*1.62
    elif row['SCHTRN1']>0:
        return row['DISTTOSC17']*1.62
    elif row['SCHTRN2']>0:
        return row['DISTTOSC17']*1.62
    else: return -99
    
why_dict={
  1: "H",
  2: "H",
  3: "W",
  4: "W",
  5: "W",
  6: "D", #drop-off, pick-up
  7: None,
  8: "C", # school or college
  9: None,
  10: None,
  11: "G", # groceries
  12: "S", # buy services
  13: "E", # eat
  14: "S", # buy services
  15: "R", # recreation
  16: "X", # exercise
  17: "V", # visit friends
  18: "P", # hoepital or health center
  19: "Z", # religion
  97: None,
  -7: None,
  -8: None,
  -9: None
}        
            
def activity_schedule(person_row):
    unique_id=person_row['uniquePersonId']
    this_person_trips=tables['trips'].loc[tables['trips']['uniquePersonId']==unique_id]
    if len(this_person_trips)==0:
        person_row['activities']='H'
        person_row['start_times']=''
        return person_row
    else:
        sched=[this_person_trips['why_from_mapped'].iloc[0]]
        if sched[0] is None:
            sched=['H']
        strt_times=[]
        activities, start_times=[list(this_person_trips['why_to_mapped']),
                                 list(this_person_trips['STRTTIME'])]
        start_times_padded=[str(st).zfill(4) for st in start_times]
        start_times_s=[int(st[:2])*3600+int(st[2:3])*60 for st in start_times_padded]
        for actInd in range(len(activities)):
            if ((activities[actInd] is not None) and 
                (not activities[actInd] == sched[-1])):
                sched.extend([activities[actInd]])
                strt_times.extend([start_times_s[actInd]])
        if len(sched)==1:
            sched=['H']
        if not sched[0]=='H':
            sched=['H']+sched
            strt_times=[strt_times[0]-3600]+strt_times                
        person_row['activities']=  '_'.join(sched) 
        person_row['start_times']=  '_'.join([str(st) for st in strt_times])
        return person_row
        
def long_form_data(mode_table, alt_attrs, generic_attrs, nalt=4, y_true=True):
    """
    generate long form data for logit model from mode table
    
    Arguments:
    ---------------------------------
    mode_table: pandas dataframe with mocho information
    alt_attrs: alternative-specific attributes, dict: key=varname in long form dataframe, 
               value=varname for each alternative in mode_table
    generic_attrs: case-specific attributes, generally demographic vars, list, ele=varname in mode_table.
    nalt: the number of alternatives
    
    Returns:
    -----------------------------------
    long_data_df: pandas dataframe in logit long data form
    """
    basic_columns = ['group', 'alt', 'choice']
    alt_tmp, choice_tmp = list(range(nalt)), [0 for i in range(nalt)]
    keys = basic_columns + list(alt_attrs.keys()) + generic_attrs
    long_data_obj = {key: [] for key in keys}
    for rid, row in mode_table.iterrows():
        long_data_obj['group'] += [rid for i in range(nalt)]
        long_data_obj['alt'] += alt_tmp
        mode_choice = choice_tmp.copy()
        if y_true:
            mode_choice[int(row['mode'])] = 1
        long_data_obj['choice'] += mode_choice
        for alt_attr in alt_attrs:
            long_data_obj[alt_attr] += [row.get(row_attr, 0) for row_attr in alt_attrs[alt_attr]]
        for g_attr in generic_attrs:
            long_data_obj[g_attr] += [row[g_attr] for i in range(nalt)]
    long_data_df = pd.DataFrame.from_dict(long_data_obj)
    return long_data_df
    
@run_time
def long_form_data_upsample(long_data_df_in, upsample_new={0: '+0', 1: '+0', 2: '+0', 3: '+0'},
    seed=None, disp=True):
    """
    make the long_form_data more balanced by upsampling 
    (add randomly sampled new cases less represented alternatives) 
    
    Arguments:
    ---------------------------------
    long_data_df_in: input long form dataframe
    upsample_new: a dict defining how many new cases are added for each alternative, 
                  key: index of alternaive
                  value: "+N" to add N cases or "*N" to increase the number of cases by N times. 
    seed: random seed
    
    Returns:
    -----------------------------------
    long_data_df_out: output long form dataframe after upsampling
    """
    print('upsampling...')
    long_data_df_out = long_data_df_in.copy()
    casedata_list = [data for caseID, data in long_data_df_out.copy().groupby('group')]
    caseIDs = list(set(long_data_df_out['group']))
    # alt_spec_casedata_list = dict()
    dist_before, dist_after = [], []
    new_casedata_list = []
    if seed is not None:
        np.random.seed(seed)
    for alt_idx in upsample_new:
        this_alt_casedata_list = [data for data in casedata_list if list(data['choice'])[alt_idx]==1]
        num_this_alt_casedata = len(this_alt_casedata_list)
        dist_before.append('{}-{}'.format(alt_idx, num_this_alt_casedata))
        if upsample_new[alt_idx].startswith('+'):
            num_new = int(upsample_new[alt_idx][1:])
        elif upsample_new[alt_idx].startswith('*'):
            num_new = int(num_this_alt_casedata * (float(upsample_new[alt_idx][1:])-1))
        # alt_spec_casedata_list[alt_idx] = this_alt_casedata_list
        new_casedata_list += [this_alt_casedata_list[i].copy() for i in np.random.choice(
            range(len(this_alt_casedata_list)), size=num_new)]
        dist_after.append('{}-{}'.format(alt_idx, num_this_alt_casedata+num_new))
    maxID = np.array(caseIDs).max()
    for idx, new_casedata in enumerate(new_casedata_list):
        new_casedata['group'] = maxID+idx+1
    long_data_df_out = pd.concat([long_data_df_out] + new_casedata_list, axis=0)
    if disp:
        print('Before: {}'.format(', '.join(dist_before)))
        print('After: {}'.format(', '.join(dist_after)))
            
    return long_data_df_out
    
    
    
def logit_spec(long_data_df, alt_attr_vars, generic_attrs=[], constant=True, 
               alts={0:'drive', 1:'cycle', 2:'walk', 3:'PT'}, ref_alt_ind=0):
    """
    generate specification & varnames for pylogit
    
    Arguments:
    ------------------------------
    long_data_df: pandas dataframe, long data, generated by long_form_data
    alt_attr_vars: list of alternative specific vars
    generic_attrs: list of case specific vars, generally demographic vars
    constant: whether or not to include ASCs
    alts: a dict or list to define indices and names of alternative
    ref_alt_ind: index of reference alternative for ASC specification
    
    Returns:
    --------------------------------
    model: pylogit MNL model object
    numCoef: the number of coefficients to estimated
    """
    specifications = OrderedDict()
    names = OrderedDict()
    nalt = len(alts)
    if isinstance(alts, list):
        alts = {i:i for i in alts}
    for var in alt_attr_vars:
        specifications[var] = [list(range(nalt))]
        names[var] = [var]
    for var in generic_attrs:
        specifications[var] = [i for i in range(nalt) if i != ref_alt_ind]
        names[var] = [var + ' for ' + alts[i] for i in alts if i != ref_alt_ind]
    if constant:
        specifications['intercept'] = [i for i in range(nalt) if i != ref_alt_ind]
        names['intercept'] = ['ASC for ' + alts[i] for i in alts if i != ref_alt_ind]
    model = pl.create_choice_model(data = long_data_df.copy(),
                        alt_id_col="alt",
                        obs_id_col="group",
                        choice_col="choice",
                        specification=specifications,
                        model_type = "MNL",
                        names = names
    )
    numCoef = sum([len(specifications[s]) for s in specifications])
    return model, numCoef

def logit_est_disp(model, numCoef, nalt=4, disp=True):
    """
    estimate a logit model and display results, using just_point=True in case of memory error
    
    Arguments:
    ---------------------------
    model & numCoef: see logit_spec; nalt: the number of alternatives
    disp: whether or not to display estimation results.
    
    Return:
    ----------------------------
    modelDict: a dict, "just_point" indicates whether the model is point-estimate only (no std.err / ttest / p-value)
                       "model" is the pylogit MNL model object, it is better used when just_point=False
                       "params": a dict with key=varible_name and value=parameter, only valid for just_point=True
    """
    try:
        model.fit_mle(np.zeros(numCoef))
        if disp:
            print(model.get_statsmodels_summary())
        return {'just_point': False, 'model': model}
    except:
        model_result = model.fit_mle(np.zeros(numCoef), just_point=True)
        ncs = int(model.data.shape[0]/nalt)
        beta = model_result['x']
        if disp:
            ll0 = np.log(1/nalt) * ncs
            ll = -model_result['fun']
            mcr = 1 - ll / ll0
            print('\n\nLogit model summary\n---------------------------')
            print('number of cases: ', ncs)
            print('Initial Log-likelihood: ', ll0)
            print('Final Log-likelihood: ', ll)
            print('McFadden R2: {:4.4}\n'.format(mcr))
            print('\nLogit model parameters:\n---------------------------')
            for varname, para in zip(model.ind_var_names, beta):
                print('{}: {:4.6f}'.format(varname, para))
        params = {varname: param for varname, param in zip(model.ind_var_names, beta)}
        return {'just_point': True, 'model': model, 'params': params}
        
def logit_cv(data, alt_attr_vars, generic_attrs, constant=True, nfold=5, seed=None,
             alts = {0:'drive', 1:'cycle', 2:'walk', 3:'PT'},
             upsample_new = {0: '+0', 1: '+0', 2: '+0', 3: '+0'},
             method = 'max'
             ):
    """
    cross validation for logit model performance
    
    Arguments:
    ---------------------------
    data: input long form pandas dataframe
    alt_attr_vars, generic_attrs, constant, alts: logit model specification, see logit_spec
    nfold: number of folds in cv; seed: random seed for np.random
    upsample_new: upsampling specification for unbalanced data, see long_form_data_upsample
    method: 
    
    Return:
    ----------------------------
    cv_metrics: a dict with average accuracy and F1 macro score 
    cv_metrics_detail: a dict with accuracy and F1 macro score  for each fold
    """
    long_data_df = data.copy()
    if seed is not None:
        np.random.seed(seed)
    caseIDs = list(set(long_data_df['group']))
    np.random.shuffle(caseIDs)
    ncs = len(caseIDs)
    nsampe_fold = int(ncs/nfold)
    cv_data = {i: long_data_df.loc[long_data_df['group'].isin(
        caseIDs[i*nsampe_fold : (i+1)*nsampe_fold])].copy() for i in range(nfold)}
    cv_metrics_detail = {i: {'accuracy':None, 'f1_macro':None} for i in range(nfold)}
    accuracy_list, f1_macro_list = [], []
    for holdout_idx in cv_data:
        print('\ncv for fold=', holdout_idx)
        long_data_df_test = cv_data[holdout_idx].copy()
        train_list = [d.copy() for idx, d in cv_data.items() if idx!=holdout_idx]
        long_data_df_train = pd.concat(train_list, axis=0).sort_values(by=['group', 'alt'])
        long_data_df_train = long_form_data_upsample(long_data_df_train, upsample_new=upsample_new, seed=seed)
        model_train, numCoefs = logit_spec(long_data_df_train, alt_attr_vars, generic_attrs, constant=constant, alts=alts)
        modelDict_train = logit_est_disp(model_train, numCoefs, nalt=len(alts), disp=False)
        pred_prob_test, y_pred_test = asclogit_pred(long_data_df_test, modelDict_train, 
            customIDColumnName='group', alts=alts, method=method, seed=seed)
        y_true_test = np.array(long_data_df_test['choice']).reshape(-1, len(alts)).argmax(axis=1)
        ac, f1 = accuracy_score(y_true_test, y_pred_test), f1_score(y_true_test, y_pred_test, average='macro')
        cv_metrics_detail[holdout_idx]['accuracy'] = ac
        accuracy_list.append(ac)
        cv_metrics_detail[holdout_idx]['f1_macro'] = f1
        f1_macro_list.append(f1)
        print(confusion_matrix(y_true_test, y_pred_test))
    cv_metrics = {'accuracy': np.asarray(accuracy_list).mean(), 
                  'f1_macro': np.asarray(f1_macro_list).mean()}
    print('cv finished\n')
    return cv_metrics, cv_metrics_detail
        
        
    
def asclogit_pred(data_in, modelDict, customIDColumnName, method='random', seed=None,
                alts={0:'drive', 1:'cycle', 2:'walk', 3:'PT'}):
    """
    predict probabilities for logit model
    
    Arguments:
    -------------------------------
    data_in: pandas dataframe to be predicted
    modelDict: see logit_est_disp
    customIDColumnName: the column name of customer(case) ID
    alts: a dict or list defining the indices and name of altneratives
    
    Return:
    ----------------------------------
    a mat (num_cases * num_alts) of predicted probabilities, row sum=1
    """
    data = data_in.copy()
    numChoices = len(set(data[customIDColumnName]))
    # fectch variable names and parameters 
    if modelDict['just_point']:
        params, varnames = modelDict['params'].values(), modelDict['params'].keys()
    else:
        params, varnames = list(modelDict['model'].coefs.values), list(modelDict['model'].coefs.index)
    
    # case specific vars and alternative specific vars
    nalt = len(alts)
    if isinstance(alts, list):
        alts = {i:i for i in alts}
    dummies_dict = dict()
    case_varname_endswith_flag = []
    for alt_idx, alt_name in alts.items():
        case_varname_endswith_flag.append(' for '+alt_name)
        tmp = [0 for i in range(nalt)]
        tmp[alt_idx] = 1
        dummies_dict[alt_name] = np.tile(np.asarray(tmp), numChoices)
    case_varname_endswith_flag = tuple(case_varname_endswith_flag)
    
    # calc utilities
    data['utility'] = 0
    for varname, param in zip(varnames, params):
        if not varname.endswith(case_varname_endswith_flag):
            # this is an alternative specific varname
            data['utility'] += data[varname] * param
        else:
            # this is a case specific varname (ASC-like)
            main_varname, interact_with_alt = varname.split(' for ')
            use_dummy = dummies_dict[interact_with_alt]
            if main_varname == 'ASC':
                data['utility'] += use_dummy * param
            elif main_varname in data.columns:
                data['utility'] += data[main_varname] * use_dummy * param
            else:
                print('Error: can not find variable: {}'.format(varname))
                return
            
    # calc probabilities given utilities
    v = np.array(data['utility']).copy().reshape(numChoices, -1)
    v_raw = v.copy()
    v = v - v.mean(axis=1, keepdims=True) 
    v[v>700] = 700
    v[v<-700] = -700    
    expV = np.exp(v)
    p = expV / expV.sum(axis=1, keepdims=True)
    p = p.reshape(-1, nalt)
    if method == 'max':
        y = p.argmax(axis=1)
    elif method == 'random':
        if seed is not None:
            np.random.seed(seed)
        y = np.asarray([np.random.choice(list(alts.keys()), size=1, p=row)[0] for row in p])
    elif method == 'none':
        y = None
    return p, y, v_raw
    
if __name__ == "__main__":
    #********************************************
    #      Data
    #********************************************
    #mode_table=pd.read_csv(MODE_TABLE_PATH)
    nhts_per=pd.read_csv(NHTS_PATH) # for mode choice on main mode
    nhts_tour=pd.read_csv(NHTS_TOUR_PATH) # for mode speeds
    nhts_trip=pd.read_csv(NHTS_TRIP_PATH) # for motifs

    # Only use weekdays for motifs
    nhts_trip=nhts_trip.loc[nhts_trip['TRAVDAY'].isin(range(2,7))]

    # add unique ids and merge some variables across the 3 tables
    nhts_trip['uniquePersonId']=nhts_trip.apply(lambda row: str(row['HOUSEID'])+'_'+str(row['PERSONID']), axis=1)
    nhts_per['uniquePersonId']=nhts_per.apply(lambda row: str(row['HOUSEID'])+'_'+str(row['PERSONID']), axis=1)
    nhts_tour['uniquePersonId']=nhts_tour.apply(lambda row: str(row['HOUSEID'])+'_'+str(row['PERSONID']), axis=1)

    # Some lookups
    nhts_tour=nhts_tour.merge(nhts_per[['HOUSEID', 'HH_CBSA']], on='HOUSEID', how='left')
    nhts_trip=nhts_trip.merge(nhts_per[['uniquePersonId', 'R_RACE']], on='uniquePersonId', how='left')


    tables={'trips': nhts_trip, 'persons': nhts_per, 'tours': nhts_tour}
    # put tables in a dict so we can use a loop to avoid repetition
    for t in ['trips', 'persons']:
    # remove some records
        tables[t]=tables[t].loc[((tables[t]['CDIVMSAR'].isin(region_cdivsmars))&
                                 (tables[t]['URBAN']==1))]
        tables[t]=tables[t].loc[tables[t]['R_AGE_IMP']>15]
        tables[t]['income']=tables[t].apply(lambda row: income_cat_nhts(row), axis=1)
        tables[t]['age']=tables[t].apply(lambda row: age_cat_nhts(row), axis=1)
        tables[t]['children']=tables[t].apply(lambda row: children_cat_nhts(row), axis=1)
        tables[t]['workers']=tables[t].apply(lambda row: workers_cat_nhts(row), axis=1)
        tables[t]['tenure']=tables[t].apply(lambda row: tenure_cat_nhts(row), axis=1)
        tables[t]['sex']=tables[t].apply(lambda row: sex_cat_nhts(row), axis=1)
        tables[t]['bach_degree']=tables[t].apply(lambda row: bach_degree_cat_nhts(row), axis=1)
        tables[t]['cars']=tables[t].apply(lambda row: cars_cat_nhts(row), axis=1)
        tables[t]['race']=tables[t].apply(lambda row: race_cat_nhts(row), axis=1)
        tables[t]=tables[t].rename(columns= {'HTPPOPDN': 'pop_per_sqmile_home'})
    tables['trips']['purpose']=tables['trips'].apply(lambda row: purpose_cat_nhts(row), axis=1)

    # =============================================================================
    # Activities
    # =============================================================================

    # with the trip table only, map 'why' codes to a simpler list of activities
    tables['trips']['why_to_mapped']=tables['trips'].apply(lambda row: 
        why_dict[row['WHYTO']], axis=1)
    tables['trips']['why_from_mapped']=tables['trips'].apply(lambda row: 
        why_dict[row['WHYFROM']], axis=1)

    # get the full sequence of activities for each person
    tables['persons'] = tables['persons'].apply(activity_schedule, axis=1)
        
    # output the persons data with subset of columns for clustering of 
    # activity schedules
    tables['persons'][['income', 'age', 'children', 
              'sex', 'bach_degree', 'cars','activities', 'start_times']].to_csv(
          PERSON_SCHED_TABLE_PATH, index=False)
                        
    #with the tour file:
    #    get the speed for each mode and the distance to walk/drive to transit for each CBSA
    #    we can use this to estimate the travel time for each potential mode in the trip file

    #global_avg_speeds={}
    speeds={area:{} for area in set(tables['persons']['HH_CBSA'])}
    tables['tours']['main_mode']=tables['tours'].apply(lambda row: mode_cat(row['MODE_D']), axis=1)

    for area in speeds:
        this_cbsa=tables['tours'][tables['tours']['HH_CBSA']==area]
        for m in [0,1,2, 3]:
            all_speeds=this_cbsa.loc[((this_cbsa['main_mode']==m) & 
                                      (this_cbsa['TIME_M']>0))].apply(
                                        lambda row: row['DIST_M']/row['TIME_M'], axis=1)
            if len(all_speeds)>0:
                speeds[area]['km_per_minute_'+str(m)]=1.62* all_speeds.mean()
            else:
                speeds[area]['km_per_minute_'+str(m)]=float('nan')
        speeds[area]['walk_km_'+str(m)]=1.62*this_cbsa.loc[this_cbsa['main_mode']==3,'PMT_WALK'].mean()
        speeds[area]['drive_km_'+str(m)]=1.62*this_cbsa.loc[this_cbsa['main_mode']==3,'PMT_POV'].mean()

    # for any region where a mode is not observed at all, 
    # assume the speed of that mode is
    # that of the slowest region
    for area in speeds:
        for mode_speed in speeds[area]:
            if not float(speeds[area][mode_speed]) == float(speeds[area][mode_speed]):
                print('Using lowest speed')
                speeds[area][mode_speed] = np.nanmin([speeds[other_area][mode_speed] for other_area in speeds])


    # with the trips table: use all tirp data
    tables['trips']['network_dist_km']=tables['trips'].apply(lambda row: row['TRPMILES']*1.62, axis=1)
    tables['trips']['mode']=tables['trips'].apply(lambda row: mode_cat(row['TRPTRANS']), axis=1) 
    tables['trips']=tables['trips'].loc[tables['trips']['mode']>=0]                                 #get rid of some samples with -99 mode
    tables['trips'].loc[tables['trips']['TRPMILES']<0, 'TRPMILES']=0                # -9 for work-from-home   

    # create the mode choice table
    mode_table=pd.DataFrame()
    #    add the trip stats for each potential mode
    mode_table['drive_time_minutes']=tables['trips'].apply(lambda row: row['network_dist_km']/speeds[row['HH_CBSA']]['km_per_minute_'+str(0)], axis=1)
    mode_table['cycle_time_minutes']=tables['trips'].apply(lambda row: row['network_dist_km']/speeds[row['HH_CBSA']]['km_per_minute_'+str(1)], axis=1)
    mode_table['walk_time_minutes']=tables['trips'].apply(lambda row: row['network_dist_km']/speeds[row['HH_CBSA']]['km_per_minute_'+str(2)], axis=1)
    mode_table['PT_time_minutes']=tables['trips'].apply(lambda row: row['network_dist_km']/speeds[row['HH_CBSA']]['km_per_minute_'+str(3)], axis=1)
    mode_table['walk_time_PT_minutes']=tables['trips'].apply(lambda row: speeds[row['HH_CBSA']]['walk_km_'+str(3)]/speeds[row['HH_CBSA']]['km_per_minute_'+str(2)], axis=1)
    mode_table['drive_time_PT_minutes']=tables['trips'].apply(lambda row: speeds[row['HH_CBSA']]['drive_km_'+str(3)]/speeds[row['HH_CBSA']]['km_per_minute_'+str(0)], axis=1)

    for col in ['income', 'age', 'children', 'workers', 'tenure', 'sex', 
                'bach_degree',  'cars', 'race', 'purpose']:
        new_dummys=pd.get_dummies(tables['trips'][col], prefix=col)
        mode_table=pd.concat([mode_table, new_dummys],  axis=1)

    for col in [ 'pop_per_sqmile_home', 'network_dist_km', 'mode']:
        mode_table[col]=tables['trips'][col]

    # generate logit long form data
    alt_attrs = {'time_minutes': ['drive_time_minutes', 'cycle_time_minutes', 'walk_time_minutes', 'PT_time_minutes'], 
        'walk_time_PT_minutes': ['nan', 'nan', 'nan', 'walk_time_PT_minutes'], 
        'drive_time_PT_minutes': ['nan', 'nan', 'nan', 'drive_time_PT_minutes']}
    generic_attrs = ['income_gt100', 'income_gt35-lt100', 'income_lt35', 'age_19 and under',
        'age_20 to 35', 'age_35 to 60', 'age_above 60', 'children_no', 'children_yes', 'workers_none', 
        'workers_one', 'workers_two or more', 'tenure_other', 'tenure_owned', 'tenure_rented', 
        'sex_female', 'sex_male', 'bach_degree_no', 'bach_degree_yes', 'cars_none', 'cars_one',
        'cars_two or more', 'race_asian', 'race_black', 'race_other', 'race_white', 
        'purpose_HBW', 'purpose_HBO', 'purpose_NHB',
        'pop_per_sqmile_home', 'network_dist_km']
    # some of dummy vars have to be excluded as reference levels in categorical vars
    exclude_ref = ['income_gt100', 'age_19 and under', 'children_no', 'workers_none',
        'tenure_other', 'sex_female', 'bach_degree_no', 'cars_none', 'race_asian', 'purpose_HBW']
    exclude_others = ['tenure_other', 'tenure_owned', 'tenure_rented']  # tenure will cause very large parameters
    exclude_generic_attrs = exclude_ref + exclude_others
    exclude_generic_attrs = exclude_others

    # generic_attrs = []
    long_data_df = long_form_data(mode_table, alt_attrs=alt_attrs, generic_attrs=generic_attrs, nalt=4)

    # =============================================================================
    # Fit Mode Choice Model
    # =============================================================================
    alts = {0:'drive', 1:'cycle', 2:'walk', 3:'PT'}
    alt_attr_vars = ['time_minutes', 'drive_time_PT_minutes'] 
    generic_attrs = [var for var in generic_attrs if var not in exclude_generic_attrs]
    constant = True
    caseIDs = list(set(long_data_df['group']))
    np.random.seed(1)
    caseIDs_train = np.random.choice(caseIDs, size=int(len(caseIDs)*0.8), replace=False)
    caseIDs_test = [id for id in caseIDs if id not in caseIDs_train]
    long_data_df_train = long_data_df.loc[long_data_df['group'].isin(caseIDs_train)].copy()
    long_data_df_test = long_data_df.loc[long_data_df['group'].isin(caseIDs_test)].copy()

    predict_method = 'max'

    # # unbalanced data: drving is over represented, need to add cases for other modes, especially cycling
    # # use cross validation to specify how the data is adjusted by upsampling
    # # it took a long time to run...
    # upsample_new_spec = {1: [20, 25, 50, 75, 100], 2: [1.2, 1.5, 2.0], 3: [3, 5, 10]}
    # cv_results = []
    # for i in upsample_new_spec[1]:
        # for j in upsample_new_spec[2]:
            # for k in upsample_new_spec[3]:
                # upsample_new = {0:'+0', 1:'*'+str(i), 2:'*'+str(j), 3:'*'+str(k)}
                # print('\nupsample_new: ', upsample_new)
                # cv_metrics, cv_metrics_detail = logit_cv(long_data_df_train, alt_attr_vars, generic_attrs, 
                    # constant=constant, upsample_new=upsample_new, method=predict_method, nfold=4)
                # cv_results.append({'upsample_new': upsample_new, 'cv_metrics': cv_metrics})
    # best_f1 = 0
    # for ele in cv_results:
        # if ele['cv_metrics']['f1_macro'] > best_f1:
            # best_f1 = ele['cv_metrics']['f1_macro']
            # best_upsample_new = ele['upsample_new']
    # with open('cv_results.p', 'wb') as f:
        # pickle.dump(cv_results, f)

    upsample_new = {0: '+0', 1: '*20', 2: '*2', 3: '*5'} # best_upsample_new

    long_data_df_train = long_form_data_upsample(long_data_df_train, upsample_new=upsample_new, seed=1)
    model_train, numCoefs = logit_spec(long_data_df_train, alt_attr_vars, generic_attrs, constant=constant, alts=alts)
    modelDict_train = logit_est_disp(model_train, numCoefs, nalt=len(alts), disp=False)

    print('\nTraining data performance:\n--------------------------------')
    pred_prob_train, y_pred_train, v = asclogit_pred(long_data_df_train, modelDict_train, 
        customIDColumnName='group', alts=alts, method=predict_method, seed=1)
    y_true_train = np.array(long_data_df_train['choice']).reshape(-1, len(alts)).argmax(axis=1)
    conf_mat_train=confusion_matrix(y_true_train, y_pred_train)
    print(conf_mat_train)
    print('Accuracy: {:4.4f}, F1 macro: {:4.4f}'.format(
        accuracy_score(y_true_train, y_pred_train), f1_score(y_true_train, y_pred_train, average='macro')
        ))
    for i in range(len(conf_mat_train)):
        print('Total True for Class '+alts[i]+': '+str(sum(conf_mat_train[i])))
        print('Total Predicted for Class '+alts[i]+': '+str(sum([p[i] for p in conf_mat_train])))
        
    print('\nTest data performance:\n--------------------------------')
    pred_prob_test, y_pred_test, v = asclogit_pred(long_data_df_test, modelDict_train, 
        customIDColumnName='group', alts=alts, method=predict_method, seed=1)
    y_true_test = np.array(long_data_df_test['choice']).reshape(-1, len(alts)).argmax(axis=1)
    conf_mat_test = confusion_matrix(y_true_test, y_pred_test)
    print(conf_mat_test)
    print('Accuracy: {:4.4f}, F1 macro: {:4.4f}'.format(
        accuracy_score(y_true_test, y_pred_test), f1_score(y_true_test, y_pred_test, average='macro')
        ))
    for i in range(len(conf_mat_test)):
        print('Total True for Class '+alts[i]+': '+str(sum(conf_mat_test[i])))
        print('Total Predicted for Class '+alts[i]+': '+str(sum([p[i] for p in conf_mat_test])))


    # use all data
    long_data_df = long_form_data_upsample(long_data_df, upsample_new=upsample_new, seed=1)
    model, numCoefs = logit_spec(long_data_df, alt_attr_vars, generic_attrs, constant=constant, alts=alts)
    modelDict = logit_est_disp(model, numCoefs)


    pickle.dump(modelDict, open( PICKLED_MODEL_PATH, "wb" ) )
    json.dump({'alt_attrs': alt_attrs, 'alt_attr_vars': alt_attr_vars, 'generic_attrs': generic_attrs, 'alts': alts, 'constant': constant},
        open(LOGIT_FEATURES_LIST_PATH, 'w' ))

    #=================================#
    #       RandomForestClassifier    # 
    #=================================# 
    # this is just for comparison with the same train/test dataset
    print('')
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, RandomizedSearchCV

    features=[c for c in mode_table.columns if not c=='mode']

    mode_table_train = mode_table.loc[mode_table.index.isin(caseIDs_train)]
    mode_table_test = mode_table.loc[mode_table.index.isin(caseIDs_test)]
    X_train, y_train = mode_table_train[features], mode_table_train['mode']
    X_test, y_test = mode_table_test[features], mode_table_test['mode']

    # X=mode_table[features]
    # y=mode_table['mode']
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

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
                                   n_iter = 128, cv = 5, verbose=1, random_state=0, 
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

    predicted=rfWinner.predict(X_test)
    conf_mat=confusion_matrix(y_test, predicted)
    print('\nTest data performance (RF):\n--------------------------------')
    print(conf_mat)
    print('Accuracy: {:4.4f}, F1 macro: {:4.4f}'.format(
        accuracy_score(y_test, predicted), f1_score(y_test, predicted, average='macro')
        ))
    # rows are true labels and coluns are predicted labels
    # Cij  is equal to the number of observations 
    # known to be in group i but predicted to be in group j.
    for i in range(len(conf_mat)):
        print('Total True for Class '+str(i)+': '+str(sum(conf_mat[i])))
        print('Total Predicted for Class '+str(i)+': '+str(sum([p[i] for p in conf_mat])))






       
                
