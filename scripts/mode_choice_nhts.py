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
from collections import OrderedDict
import copy
import re

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import confusion_matrix

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
          

def create_mode_choice_trip_table(city_folder):
    NHTS_PATH='NHTS/perpub.csv'
    NHTS_TOUR_PATH='NHTS/tour17.csv'
    NHTS_TRIP_PATH='NHTS/trippub.csv'
    nhts_per=pd.read_csv(NHTS_PATH) # for person-level data
    nhts_tour=pd.read_csv(NHTS_TOUR_PATH) # for mode speeds
    nhts_trip=pd.read_csv(NHTS_TRIP_PATH) # for trip-level data
    
    REGION_CDIVMSARS_BY_CITY={"Boston": [11,21] ,
                       "Detroit": [32]
                       }
    region_cdivsmars=REGION_CDIVMSARS_BY_CITY[city_folder]

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

                        
    #with the tour file:
    #    get the speed for each mode and the distance to walk/drive to transit for each CBSA
    #    we can use this to estimate the travel time for each potential mode in the trip file

    #global_avg_speeds={}
    speeds={area:{} for area in set(tables['persons']['HH_CBSA'])}
    tables['tours']['main_mode']=tables['tours'].apply(lambda row: mode_cat(row['MODE_D']), axis=1)
    tables['trips']['mode']=tables['trips'].apply(lambda row: mode_cat(row['TRPTRANS']), axis=1) 
    
    tables['trips']['NUMTRANS']=tables['trips'].apply(lambda row: max(0, row['NUMTRANS']), axis=1) 

    for area in speeds:
        this_cbsa_tours=tables['tours'][tables['tours']['HH_CBSA']==area]
        this_cbsa_trips=tables['trips'][tables['trips']['HH_CBSA']==area]
        for m in [0, 3]:
            all_speeds=this_cbsa_tours.loc[((this_cbsa_tours['main_mode']==m) & 
                                      (this_cbsa_tours['TIME_M']>0))].apply(
                                        lambda row: row['DIST_M']/row['TIME_M'], axis=1)
            if len(all_speeds)>0:
                speeds[area]['km_per_minute_'+str(m)]=1.62* all_speeds.mean()
            else:
                speeds[area]['km_per_minute_'+str(m)]=float('nan')
        speeds[area]['km_per_minute_1']=12/60
        speeds[area]['km_per_minute_2']=4/60
        speeds[area]['walk_km_'+str(3)]=1.62*this_cbsa_tours.loc[this_cbsa_tours['main_mode']==3,'PMT_WALK'].mean()
        speeds[area]['drive_km_'+str(3)]=1.62*this_cbsa_tours.loc[this_cbsa_tours['main_mode']==3,'PMT_POV'].mean()
        speeds[area]['ntransfers_'+str(3)]=1.62*this_cbsa_trips.loc[this_cbsa_trips['mode']==3,'NUMTRANS'].mean()

    # for any region where a mode is not observed at all, 
    # assume the speed of that mode is
    # that of the slowest region
    for area in speeds:
        for mode_speed in speeds[area]:
            if not float(speeds[area][mode_speed]) == float(speeds[area][mode_speed]):
                print(mode_speed)
                print(speeds[area][mode_speed])
                speeds[area][mode_speed] = np.nanmean([speeds[other_area][mode_speed] for other_area in speeds])

    # with the trips table: use all trip data
    
    tables['trips']['network_dist_km']=tables['trips'].apply(lambda row: row['TRPMILES']*1.62, axis=1)
    
    tables['trips']=tables['trips'].loc[tables['trips']['mode']>=0]                                 #get rid of some samples with -99 mode
    tables['trips'].loc[tables['trips']['TRPMILES']<0, 'TRPMILES']=0                # -9 for work-from-home   

    # create the mode choice table
    mode_table=pd.DataFrame()
    #    add the trip stats for each potential mode
    mode_table['drive_vehicle_time_minutes']=tables['trips'].apply(lambda row: row['network_dist_km']/speeds[row['HH_CBSA']]['km_per_minute_'+str(0)], axis=1)
    mode_table['drive_cost']=tables['trips'].apply(lambda row: row['network_dist_km']*0.79/1.62, axis=1)
    mode_table['cycle_active_time_minutes']=tables['trips'].apply(lambda row: row['network_dist_km']/speeds[row['HH_CBSA']]['km_per_minute_'+str(1)], axis=1)
    mode_table['walk_active_time_minutes']=tables['trips'].apply(lambda row: row['network_dist_km']/speeds[row['HH_CBSA']]['km_per_minute_'+str(2)], axis=1)
    mode_table['PT_time_minutes']=tables['trips'].apply(lambda row: row['network_dist_km']/speeds[row['HH_CBSA']]['km_per_minute_'+str(3)], axis=1)
    mode_table['PT_cost']=tables['trips'].apply(lambda row: 1.5 + 0.25*speeds[row['HH_CBSA']]['ntransfers_'+str(3)] , axis=1)
    mode_table['PT_active_time_minutes']=tables['trips'].apply(lambda row: speeds[row['HH_CBSA']]['walk_km_'+str(3)]/speeds[row['HH_CBSA']]['km_per_minute_'+str(2)], axis=1)
    mode_table['drive_time_PT_minutes']=tables['trips'].apply(lambda row: speeds[row['HH_CBSA']]['drive_km_'+str(3)]/speeds[row['HH_CBSA']]['km_per_minute_'+str(0)], axis=1)

    for col in ['income', 'age', 'children', 'workers', 'tenure', 'sex', 
                'bach_degree',  'cars', 'race', 'purpose']:
        new_dummys=pd.get_dummies(tables['trips'][col], prefix=col)
        mode_table=pd.concat([mode_table, new_dummys],  axis=1)

    for col in [ 
#            'pop_per_sqmile_home', 
                'network_dist_km', 'mode']:
        mode_table[col]=tables['trips'][col]     
    return mode_table





        
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

def logit_est_disp(model, numCoef, nalt=4, disp=True, just_point=False):
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
    if just_point==False:
        try:
            model.fit_mle(np.zeros(numCoef))
            if disp:
                print(model.get_statsmodels_summary())
            return {'just_point': False, 'model': model}
        except:
            print('Failed to fit. Tying just point')
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
        modelDict_train = logit_est_disp(model_train, numCoefs, nalt=len(alts), disp=False, just_point=False)
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

    
def observable_utility_calc(data_in, modelDict, customIDColumnName, alts):
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
    v = np.array(data['utility']).copy().reshape(numChoices, -1)
    return v


class NhtsModeRF:
    def __init__(self, table_name, city_folder):
        self.city_folder=city_folder
        self.PICKLED_MODEL_PATH='./cities/{}/models/trip_mode_rf.p'.format(city_folder)
        self.RF_FEATURES_LIST_PATH='./cities/{}/models/rf_features.json'.format(city_folder)
        self.new_alt_specs=[]
        try:
            self.rf_model=pickle.load( open(self.PICKLED_MODEL_PATH, 'rb'))
            self.features=json.load(open(self.RF_FEATURES_LIST_PATH))
        except:
            self.train()
            self.rf_model=pickle.load( open(self.PICKLED_MODEL_PATH, 'rb'))
            self.features=json.load(open(self.RF_FEATURES_LIST_PATH))
    def train(self):
        print('Training mode choice RF')
        mode_table=create_mode_choice_trip_table(self.city_folder)
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
        pickle.dump( self.rfWinner, open( self.PICKLED_MODEL_PATH, "wb" ) )
        json.dump(self.features,open(self.RF_FEATURES_LIST_PATH, 'w' ))

    def generate_feature_df(self, trips):
        feature_df = pd.DataFrame(trips)  
        for feat in ['income', 'age', 'children', 'workers', 'tenure', 'sex', 
                     'bach_degree', 'race', 'cars', 'purpose']:
            new_dummys=pd.get_dummies(feature_df[feat], prefix=feat)
            feature_df=pd.concat([feature_df, new_dummys],  axis=1)
        feature_df['drive_vehicle_time_minutes'] = feature_df.apply(lambda row: row['driving_route']['driving'], axis=1)     
        feature_df['cycle_active_time_minutes'] = feature_df.apply(lambda row: row['cycling_route']['cycling'], axis=1)     
        feature_df['walk_active_time_minutes'] = feature_df.apply(lambda row: row['walking_route']['walking'], axis=1)     
        feature_df['PT_time_minutes'] = feature_df.apply(lambda row: row['pt_route']['pt'], axis=1)
        feature_df['walk_time_PT_minutes'] = feature_df.apply(lambda row: row['pt_route']['walking'], axis=1)  
        feature_df['drive_time_PT_minutes']=0 
#        feature_df['network_dist_km']=feature_df.apply(lambda row: row['drive_time_minutes']*30/60, axis=1)
        for rff in self.features:
            if rff not in feature_df.columns:
                feature_df[rff]=False
        feature_df=feature_df[self.features]
        self.feature_df = feature_df
        
    def predict_modes(self):
        mode_probs=self.rf_model.predict_proba(self.feature_df)
        chosen_modes=[int(np.random.choice(range(4), size=1, replace=False, p=mode_probs[i])[0]) for i in range(len(mode_probs))]
        self.predicted_prob, self.predicted_modes = mode_probs, chosen_modes
        

class NhtsModeLogit:
    def __init__(self, table_name, city_folder):
        self.city_folder=city_folder
        self.PICKLED_MODEL_PATH='./cities/{}/models/trip_mode_logit.p'.format(city_folder)
        self.LOGIT_FEATURES_LIST_PATH='./cities/{}/models/logit_features.json'.format(city_folder)
        try:
            logit_model=pickle.load( open(self.PICKLED_MODEL_PATH, 'rb'))
            logit_features=json.load(open(self.LOGIT_FEATURES_LIST_PATH))
        except:
            self.train()
            logit_model=pickle.load( open(self.PICKLED_MODEL_PATH, 'rb'))
            logit_features=json.load(open(self.LOGIT_FEATURES_LIST_PATH))
        if len(logit_features):
            self.logit_alt_attrs = logit_features['alt_attrs']
            self.logit_alt_attr_vars = logit_features['alt_attr_vars']
            self.logit_generic_attrs = logit_features['generic_attrs']
            self.logit_constant = logit_features['constant']
        else:
            self.logit_alt_attrs, self.logit_alt_attr_vars, self.logit_generic_attrs = {}, [], []
        self.logit_model = logit_model
        if self.logit_model is not None and self.logit_model['just_point'] is False:
            # for convenience, use just_point=True for all cases so that we can modify the model easily
            self.logit_model['just_point'] = True
            print('Not point estimate only')
#            self.logit_model['params'] = {v: p for v, p in zip(
#                list(logit_model['model'].coefs.index), list(logit_model['model'].coefs.values))}
        self.base_alts = {0:'drive', 1:'cycle', 2:'walk', 3:'PT'}
        self.new_alt_specs=[]
        self.nests_spec=None
        self.new_alts = []
        self.new_alts_like = {}
        self.update_alts()
        self.prob = []
        self.v = []     #observed utility for logit
        self.mode = []
        
    def generate_feature_df(self, trips):
        feature_df = pd.DataFrame(trips)  
        for feat in ['income', 'age', 'children', 'workers', 'tenure', 'sex', 
                     'bach_degree', 'race', 'cars', 'purpose']:
            new_dummys=pd.get_dummies(feature_df[feat], prefix=feat)
            feature_df=pd.concat([feature_df, new_dummys],  axis=1)
        feature_df['drive_vehicle_time_minutes'] = feature_df.apply(lambda row: row['driving_route']['driving'], axis=1)     
        feature_df['cycle_active_time_minutes'] = feature_df.apply(lambda row: row['cycling_route']['cycling'], axis=1)     
        feature_df['walk_active_time_minutes'] = feature_df.apply(lambda row: row['walking_route']['walking'], axis=1)     
        feature_df['PT_time_minutes'] = feature_df.apply(lambda row: row['pt_route']['pt'], axis=1)
        feature_df['PT_active_time_minutes'] = feature_df.apply(lambda row: row['pt_route']['walking'], axis=1)  
        feature_df['drive_time_PT_minutes']=0 
        feature_df['PT_vehicle_time_minutes']=feature_df['PT_time_minutes']+feature_df['drive_time_PT_minutes']
        feature_df['drive_cost']=feature_df['network_dist_km']*0.79/1.62
        feature_df['PT_cost']=1.5
        self.base_feature_df = copy.deepcopy(feature_df)
        self.feature_df = copy.deepcopy(feature_df)
        
    def set_feature_df(self, feature_df_in):
        self.features_df = copy.deepcopy(feature_df_in)
    
    def get_long_form_data(self):
        nalt = len(self.alts) 
        long_data_df = long_form_data(self.feature_df, alt_attrs=self.logit_alt_attrs, 
            generic_attrs=self.logit_generic_attrs, nalt=nalt, y_true=False)
        self.long_data_df = copy.deepcopy(long_data_df)
        
    def get_availability_mask(self):
        if not hasattr(self, 'alts'): self.update_alts
        pattern = re.compile('(.*?)([<=>]+.*)')
        mask = np.ones((self.feature_df.shape[0], len(self.alts)))
        for new_alt in self.new_alts:
            if 'availability' not in new_alt:
                continue
            parse = pattern.findall(new_alt['availability'])
#            print('parse: ', parse)
            if len(parse) != 1 or len(parse[0]) != 2 or parse[0][0] not in self.feature_df.columns:
                print('The availability setting of new alt {} is invalid'.format(new_alt['name']))
                continue
            availability_varname, availability_condition = parse[0]
            case_idx = eval('np.where(self.feature_df["{}"]{})'.format(availability_varname, availability_condition))
            alt_idx = self.alts_reverse[new_alt['name']]
            mask[:, [alt_idx]] = 0      
            mask[case_idx, [alt_idx]] = 1   # make this alt available only for case_idx
        self.mask = mask
        
    def update_alts(self):
        alts = self.base_alts.copy()
        for i, new_alt in enumerate(self.new_alts): alts[4+i] = new_alt['name']
        self.alts = alts
        self.alts_reverse = {v:k for k,v in self.alts.items()}
        
    def predict_modes(self, method='random', seed=None):
        if self.nests_spec is not None:
            self.gnl_predict(self.nests_spec, method=method, seed=seed)
        else:
            self.mnl_predict(method=method, seed=seed)
    
    def mnl_predict(self, method='random', seed=None):
        self.update_alts()
        self.get_long_form_data()
        long_data_df = copy.deepcopy(self.long_data_df)
        prob, mode, v = asclogit_pred(long_data_df, self.logit_model, customIDColumnName='group', 
            method=method, alts=self.alts, seed=seed)
        self.predicted_prob, self.predicted_modes, self.predicted_v = prob, mode, v
        
    def quasi_nl_predict(self, nests_spec, method='random', n_sample=10, seed=None):
        """
        nests_spec = [{'name': 'cycle_like', 'alts':['cycle','dockless'], 'sigma':0.5}, {}...]
        """
        self.update_alts()
        self.get_long_form_data()
        long_data_df = copy.deepcopy(self.long_data_df)
        logit_model_tmp = copy.deepcopy(self.logit_model)
        for nest in nests_spec:
            if 'name' not in nest: nest['name'] = '_'.join(nest['alts'])
            long_data_df[nest['name']] =  0
            idx_alts_in_nest = [self.alts_reverse[x] for x in nest['alts']]
            long_data_df.loc[long_data_df['alt'].isin(idx_alts_in_nest), nest['name']] = 1
        
        # mxlogit prediction by simulation
        if seed is not None: np.random.seed(seed) 
        std_normal_samples = np.random.randn(n_sample, len(nests_spec))
        std_normal_samples = -np.abs(std_normal_samples)   
        prob = np.zeros(long_data_df.shape[0]).reshape(-1, len(self.alts))
        for sample_row in std_normal_samples:
            for s, nest in zip(sample_row, nests_spec):
                logit_model_tmp['params'][nest['name']] = nest['sigma'] * s
            sample_prob, sample_mode, sample_v =  asclogit_pred(long_data_df, logit_model_tmp, 
                customIDColumnName='group', method='none', alts=self.alts, seed=seed)
            prob += np.asarray(sample_prob)
        prob /= n_sample
        if method == 'random':
            mode = np.asarray([np.random.choice(list(self.alts.keys()), size=1, p=row)[0] for row in prob])
        elif method == 'max':
            mode = prob.argmax(axis=1)
        self.predicted_modes, self.predicted_prob = mode, prob
        
    def gnl_predict(self, nests_spec, method='random', seed=None):
        """
        generalized nested logit prediction
        nests_spec: 
            a list of dicts, each dict speicifies a nest with keys='name', 'alts', 'lambda', ['alpha']
            'lambda': a float (normally 0~1) to represent degree of dissimilarity among alts in the nest,
                      larger lambda = greater dissimilarity, GNL=MNL if all lambdas are set to 1
            'alpha': an optional list with the same length of 'alts' to represent degree of membership that each alt
                     in 'alts' belongs to this nest. When an alt belongs to multiple nests, its alphas would be 
                     normalized to make sum(alphas)=1. If alpha list is not given, a default list with all ones is used.
        """
#        print('\t \t \t update alts')
        self.update_alts()
#        print('\t \t \t get long form')
        self.get_long_form_data()
        long_data_df = copy.deepcopy(self.long_data_df)
        alts = self.alts
#        print('\t \t \t get utilities')
        v = observable_utility_calc(long_data_df, self.logit_model, customIDColumnName='group',alts=alts)
        v_raw = v.copy()
        v = v - v.mean(axis=1, keepdims=True) 
        v[v>700] = 700
        v[v<-700] = -700
        very_large_num = np.exp(700)
        # set "others" nest for alts that are not in any specfied nest:
#        print('\t \t \t nesting')
        alts_in_any_nest = list(set([alt for nest in nests_spec for alt in nest['alts']]))
        alts_not_in_any_nest = [alt for alt in self.alts_reverse if alt not in alts_in_any_nest]
        nests_spec = copy.deepcopy(nests_spec)
        if len(alts_not_in_any_nest) > 0:
            nests_spec.append({'name': 'others', 'alts':alts_not_in_any_nest, 'lambda':1})
        # or set stand-alone nests for each alt that does not belong to any nest:
        # for alt in alts_not_in_any_nest: nests_spec.append({'name': alt+'_alone' ,'alts':[alt], 'lambda':1})
        
        # alpha_matrix is a #alt * #nest matrix, with row_sum=1, alpha_jk: the degree that alt_j belongs to nest_k
#        print('\t \t \t alphas')
        alpha_matrix = np.zeros((len(alts), len(nests_spec)))
        for nest_idx, nest in enumerate(nests_spec): 
            alts_idx = [self.alts_reverse[alt] for alt in nest['alts']]
            nest['alts_idx'] = alts_idx
            if 'alpha' not in nest: 
                alpha_matrix[alts_idx, nest_idx] = 1
            else:
                alpha_matrix[alts_idx, nest_idx] = nest['alpha']
        alpha_matrix = alpha_matrix / alpha_matrix.sum(axis=1, keepdims=True)
        
        # for each nest_k, calc nest_utility = sum((alpha_jk*exp(Vj))**(1/lambda_k)),
        # where alt_j belongs to nest_k with alpha_jk
#        print('\t \t \t nest utility')
        nest_utility = []
        for nest_idx, nest in enumerate(nests_spec):
            alts_idx = nest['alts_idx']
            tmp = ((np.exp(v[:, alts_idx]) * alpha_matrix[alts_idx, nest_idx]) ** (1/nest['lambda'])).sum(axis=1)
            nest_utility.append(tmp)
        nest_utility = np.asarray(nest_utility).T

        lambda_array = np.asarray([nest['lambda'] for nest in nests_spec])
        denominator = (nest_utility ** lambda_array).sum(axis=1)
        denominator[np.where(denominator==np.inf)] = very_large_num
#        print('\t \t \t calc probs')
        prob = []
        for alt_idx, alt in enumerate(alts):
            nests_idx = np.where(alpha_matrix[alt_idx,:]>0)[0]   #index of all nests this alt belongs to
            numerator_left_term = (np.exp(v[:,[alt_idx]]) * alpha_matrix[alt_idx, nests_idx]) ** (1/lambda_array[nests_idx])
            numerator_right_term = nest_utility[:, nests_idx] ** (lambda_array[nests_idx]-1) 
            numerator = (numerator_left_term * numerator_right_term).sum(axis=1)
            numerator[np.where(numerator==np.inf)] = very_large_num
            prob.append(numerator / denominator)
        prob = np.asarray(prob).T
#        print('\t \t \t deal with nan')
        # for nan values caused by overflow, use uniform probs (1/#nalt) for each alt
        for row_idx, row in enumerate(prob):
            if any(np.isnan(row)):
                prob[row_idx] = np.ones(len(row)) * (1/len(row))
                print('\n[warning] Probabilities of all modes for case #{} are set equal due to overflow'.format(row_idx))
                print('Raw data for case #{}: '.format(row_idx))
                for k, v in self.feature_df.iloc[row_idx].to_dict().items(): print('{}: {}'.format(k,v))
        # TODO: pass in matrix of availability bools for each mode
        # mask the prob matrix
        self.get_availability_mask()
        prob = prob * self.mask
        
#        print('\t \t \t pick results based on probs')
        if seed is not None: np.random.seed(seed)
        prob = prob / prob.sum(axis=1, keepdims=True)
        if method == 'random':
            mode = np.asarray([np.random.choice(list(self.alts.keys()), size=1, p=row)[0] for row in prob])
        elif method == 'max':
            mode = prob.argmax(axis=1)
        self.predicted_modes, self.predicted_prob, self.predicted_v = mode, prob, v_raw
            
        
    def set_logit_model_params(self, params={}):
        for v, p in params.items(): # v=varname, p=parameter
            self.logit_model['params'][v] = p
            
    def add_new_alts(self, alt_specs):
        self.new_alt_specs=alt_specs
        
    def add_nests_spec(self, nests_spec):
        self.nests_spec=nests_spec
    
    def set_new_alt(self, new_alt_spec):
        """
        new_alt_spec = {'name': 'dockless ', 
                        'attrs': {'time_minutes':{'copy':'cycle', 'operation':'-5', 'min':0.1, 'max':None}, 
                                  'walk_time_PT_minutes': 'p-5'
                                  'drive_time_PT_minutes':0 / np.nan},
                        'copy': 'driving'
                        'params': {'ASC':3, 'income_gt100':0}
                       }
        """
        name = new_alt_spec['name']
        new_alt_attrs = new_alt_spec.get('attrs', {})
        new_alt_generic_params = new_alt_spec.get('params', {})
        self.new_alts.append(new_alt_spec)
        self.update_alts()
        
        # alternative specific attributes
        alias  = {'d': 'drive', 'c': 'cycle', 'w': 'walk', 'p': 'PT'}
        for alt_attr in self.logit_alt_attrs:
            if alt_attr in new_alt_attrs:
                attr_info = new_alt_attrs[alt_attr]
                if isinstance(attr_info, (int, float)):
                    self.feature_df['{}_{}'.format(name, alt_attr)] = attr_info
                elif isinstance(attr_info, (dict, str)):
                    if isinstance(attr_info, str):
                        attr_info = {'copy':alias[attr_info[0]], 'operation':attr_info[1:], 'min':0}
                    tmp = np.asarray(self.feature_df['{}_{}'.format(attr_info['copy'], alt_attr)])
                    tmp = eval('tmp' + attr_info['operation'])
                    if 'min' in attr_info: tmp[np.where(tmp < attr_info['min'])] = attr_info['min']
                    if 'max' in attr_info: tmp[np.where(tmp > attr_info['max'])] = attr_info['max']
                    self.feature_df['{}_{}'.format(name, alt_attr)] = tmp
            else:
                self.feature_df['{}_{}'.format(name, alt_attr)] = 0
#                print('[warning] no information for {}_{}, set to 0'.format(name, alt_attr))
            self.logit_alt_attrs[alt_attr].append('{}_{}'.format(name, alt_attr))
        # for new attributes first appeard and only for this new alternative 
        for alt_attr in new_alt_attrs:
            if alt_attr not in self.logit_alt_attrs:
                self.feature_df['{}_{}'.format(name, alt_attr)] = new_alt_attrs[alt_attr]   # only numerical values are valid
                tmp = ['nan' for i in range(len(self.alts) - 1)] + ['{}_{}'.format(name, alt_attr)]
                self.logit_alt_attrs[alt_attr] = tmp
        
        # logit model coefficient for this alternative
        for g_attr in self.logit_generic_attrs:
            if 'copy' in new_alt_spec:
                #if can not get, should be copying from the reference level, thus set to 0
                self.logit_model['params']['{} for {}'.format(g_attr, name)] = self.logit_model[
                    'params'].get('{} for {}'.format(g_attr, new_alt_spec['copy']), 0)   
            else: self.logit_model['params']['{} for {}'.format(g_attr, name)] = 0
        if self.logit_constant:
            if 'copy' in new_alt_spec:
                self.logit_model['params']['ASC for {}'.format(name)] = self.logit_model[
                    'params'].get('ASC for {}'.format(new_alt_spec['copy']), 0)
            else: self.logit_model['params']['ASC for {}'.format(name)] = 0
        for p in new_alt_generic_params:
            if p in self.logit_model['params']: 
                self.logit_model['params'][p] = new_alt_generic_params[p]
            else:
                print('[warning] invalid parameter name: {}'.format(p))
                
        if 'copy' in new_alt_spec: self.new_alts_like[name] = new_alt_spec['copy']
    
    def show_agg_prob(self):
        print('\nAggregated prob: \n----------------')
        ag_prob = self.prob.sum(axis=0)
        ag_prob = ag_prob / ag_prob.sum()
        for m, p in zip(self.alts.values(), ag_prob):
            print('{}: {:4.4f}'.format(m, p))
            
    def show_agg_outcome(self):
        print('\nAggregated outcome: \n----------------')
        ncs = len(self.mode)
        for idx, m in self.alts.items():
            this_ncs = len(np.where(self.mode==idx)[0])
            print('{}: {}, {:4.2f}%'.format(m, this_ncs, this_ncs/ncs*100))
    
    def mnl_elasticity(self, vars=[], disp=True):
        """
        calculate MNL elasiticity of P(alt_j) wrt changes of variables of alt_i 
        Results are calculated based on TRAINNING data used for MNL estimation
        vars: the list of vararibles to be anlyzed, only for attributes of alts (not choice makers)
        """
        mnl_ela = {}
        train_data = self.logit_model['model'].data
        pred_prob_train, __, __ = asclogit_pred(train_data, self.logit_model, 
            customIDColumnName='group', alts=self.base_alts, method='none')
        for var in vars:
            if not var in self.logit_alt_attrs:
                print('[Error] {} is not a valid attribute of alternative'.format(var))
                continue
            mnl_ela[var] = {}
            b = self.logit_model['params'][var]
            for alt_idx, alt_name in self.base_alts.items():
                idenfifier = np.zeros_like(pred_prob_train)
                idenfifier[:, alt_idx] = 1
                x = np.asarray(train_data[var]).reshape(-1, len(self.base_alts))
                ela_mat = (idenfifier - pred_prob_train[:, [alt_idx]]) * x[:, [alt_idx]] * b
                ela_mean = ela_mat.mean(axis=0)
                mnl_ela[var][alt_name] = {self.base_alts[alt_idx]: ela for alt_idx, ela in enumerate(ela_mean)}
        if disp:
            for var, ela in mnl_ela.items():
                print('\nAverage elasticity of prob(alt) wrt {}\n'.format(var) + '-'*60)
                header = '{:10s}'.format(var)
                for key in mnl_ela[var]: header += '{:10s}'.format(key)
                print(header)
                for alt_i, ela_i in mnl_ela[var].items():
                    line = '{:10s}'.format(alt_i)
                    for alt_j, ela_ij in ela_i.items():
                        line += '{:<10.4f}'.format(ela_ij)
                    print(line)
        return mnl_ela
    
    def train(self, just_point=False):
        import pylogit as pl
        mode_table=create_mode_choice_trip_table(self.city_folder)
        mode_table['PT_vehicle_time_minutes']=mode_table['PT_time_minutes']+mode_table['drive_time_PT_minutes']
        # generate logit long form data
        alt_attrs = {'vehicle_time_minutes': ['drive_vehicle_time_minutes', 'nan', 'nan', 'PT_vehicle_time_minutes'], 
            'active_time_minutes': ['nan', 'cycle_active_time_minutes', 'walk_active_time_minutes', 'PT_active_time_minutes'], 
            'cost': ['drive_cost', 'nan', 'nan', 'PT_cost']
            }
        generic_attrs = ['income_gt100', 'income_gt35-lt100', 'income_lt35', 'age_19 and under',
            'age_20 to 35', 'age_35 to 60', 'age_above 60', 'children_no', 'children_yes', 'workers_none', 
            'workers_one', 'workers_two or more', 'tenure_other', 'tenure_owned', 'tenure_rented', 
            'sex_female', 'sex_male', 'bach_degree_no', 'bach_degree_yes', 'cars_none', 'cars_one',
            'cars_two or more', 'race_asian', 'race_black', 'race_other', 'race_white', 
            'purpose_HBW', 'purpose_HBO', 'purpose_NHB',
#            'pop_per_sqmile_home', 
#            'network_dist_km'
            ]
        # some of dummy vars have to be excluded as reference levels in categorical vars
        exclude_ref = ['income_gt100', 'age_19 and under', 'children_no', 'workers_none',
            'tenure_other', 'sex_female', 'bach_degree_no', 'cars_none', 'race_asian', 'purpose_HBW']
        exclude_others = ['tenure_other', 'tenure_owned', 'tenure_rented']  # tenure will cause very large parameters
        exclude_generic_attrs = exclude_ref + exclude_others
#        exclude_generic_attrs = exclude_others
    
        # generic_attrs = []
        long_data_df = long_form_data(mode_table, alt_attrs=alt_attrs, generic_attrs=generic_attrs, nalt=4)
    
        # =============================================================================
        # Fit Mode Choice Model
        # =============================================================================
        alts = {0:'drive', 1:'cycle', 2:'walk', 3:'PT'}
        alt_attr_vars = ['vehicle_time_minutes', 'active_time_minutes', 'cost'] 
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
    
#        upsample_new = {0: '+0', 1: '*20', 2: '*2', 3: '*5'} # best_upsample_new
    
#        long_data_df_train = long_form_data_upsample(long_data_df_train, upsample_new=upsample_new, seed=1)
        model_train, numCoefs = logit_spec(long_data_df_train, alt_attr_vars, generic_attrs, constant=constant, alts=alts)
        modelDict_train = logit_est_disp(model_train, numCoefs, nalt=len(alts), disp=True, just_point=just_point)
    
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
#        long_data_df = long_form_data_upsample(long_data_df, upsample_new=upsample_new, seed=1)
        model, numCoefs = logit_spec(long_data_df, alt_attr_vars, generic_attrs, constant=constant, alts=alts)
        modelDict = logit_est_disp(model, numCoefs, just_point=just_point, disp=True)
            
        print('\nAll data performance:\n--------------------------------')
        pred_prob, y_pred, v = asclogit_pred(long_data_df, modelDict, 
            customIDColumnName='group', alts=alts, method=predict_method, seed=1)
        y_true = np.array(long_data_df['choice']).reshape(-1, len(alts)).argmax(axis=1)
        conf_mat = confusion_matrix(y_true, y_pred)
        print(conf_mat)
        print('Accuracy: {:4.4f}, F1 macro: {:4.4f}'.format(
            accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average='macro')
            ))
        for i in range(len(conf_mat)):
            print('Total True for Class '+alts[i]+': '+str(sum(conf_mat[i])))
            print('Total Predicted for Class '+alts[i]+': '+str(sum([p[i] for p in conf_mat])))
    
        # (impact/minute)/(impact/dollar) = dollar/minute
        print('VoT Vehicle: '+str(60*modelDict['params']['vehicle_time_minutes']/modelDict['params']['cost']))
        print('VoT Active: '+str(60*modelDict['params']['active_time_minutes']/modelDict['params']['cost']))
   
        pickle.dump(modelDict, open( self.PICKLED_MODEL_PATH, "wb" ) )
        json.dump({'alt_attrs': alt_attrs, 'alt_attr_vars': alt_attr_vars, 'generic_attrs': generic_attrs, 'alts': alts, 'constant': constant},
            open(self.LOGIT_FEATURES_LIST_PATH, 'w' ))

            
if __name__ == "__main__":
    mode_choice_model=NhtsModeLogit(table_name='corktown', city_folder='Detroit')







       
                
