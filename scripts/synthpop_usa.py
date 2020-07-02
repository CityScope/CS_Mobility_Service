 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 12:48:13 2019

@author: doorleyr
"""

from synthpop.census_helpers import Census
from synthpop import categorizer as cat
from synthpop import synthesizer
import pandas as pd
import numpy as np
import json
import random


# =============================================================================
# Functions
# =============================================================================

# define some functions for creating categorsed versins of variables in census data
# Person
def age_cat(r):
    if r.AGEP <= 19: return "19 and under"
    elif r.AGEP <= 35: return "20 to 35"
    elif r.AGEP <= 60: return "35 to 60"
    return "above 60"

def race_cat(r):
    if r.RAC1P == 1: return "white"
    elif r.RAC1P == 2: return "black"
    elif r.RAC1P == 6: return "asian"
    return "other"

def sex_cat(r):
    if r.SEX == 1: return "male"
    return "female"

def tenure_cat(r):
    if r.TEN ==1 or r.TEN ==2: return "owned"
    elif r.TEN ==3:  return "rented"
    return "other"

#Household
def cars_cat(r):
    if r.VEH == 0: return "none"
    elif r.VEH == 1: return "one"
    return "two or more"

def children_cat(r):
    if r.NOC > 0: return "yes"
    return "no"

def income_cat(r): # the example file used FINCP (family income) but the ACS control is Household income
    if r.HINCP > 100000: return "gt100"
    elif r.HINCP > 35000: return "gt35-lt100"
    return "lt35"

def workers_cat(r):
    if r.WIF == 3: return "two or more"
    elif r.WIF == 2: return "two or more"
    elif r.WIF == 1: return "one"
    return "none"

# =============================================================================
# Configuration
# =============================================================================
sample_factor=20 # for the simulation population
vacancy_rate=0.03

city='Detroit'

if city=='Boston': 
    state='25' 
    state_code='ma'
elif city=='Detroit': 
    state='26'
    state_code='mi'
    
ALL_ZONES_PATH='./scripts/cities/'+city+'/clean/model_area.geojson'
SIM_ZONES_PATH='./scripts/cities/'+city+'/clean/sim_zones.json'
OD_PATH='./scripts/cities/'+city+'/raw/LODES/'+state_code+'_od_main_JT00_2015.csv'
ALL_SYNTH_HH_PATH='./scripts/cities/'+city+'/clean/all_synth_hh.csv'
ALL_SYNTH_PERSONS_PATH='./scripts/cities/'+city+'/clean/all_synth_persons.csv'
PUMA_ATTR_PATH='./scripts/cities/'+city+'/models/puma_attr.json'
SIM_POP_PATH='./scripts/cities/'+city+'/clean/sim_pop.json'
VACANT_PATH='./scripts/cities/'+city+'/clean/vacant.json'
FLOATING_PATH='./scripts/cities/'+city+'/clean/floating.json'

c = Census('7a25a7624075d46f112113d33106b6648f42686a')

# load the block group geojson for the whole area
# get set of tracts covered

# identify the data we want at tract and block group level

#Households
income_columns = ['B19001_0%02dE'%i for i in range(1, 18)]
vehicle_columns = ['B08201_0%02dE'%i for i in range(1, 7)]
workers_columns = ['B08202_0%02dE'%i for i in range(1, 6)]
families_columns = ['B11001_001E', 'B11001_002E']
# year_built_columns= ['B25034_001E', 'B25034_002E', 'B25034_003E'] 
# includes vacant structures?
tenure_columns=['B25063_001E', 'B25075_001E']
block_group_columns = income_columns + families_columns + tenure_columns
tract_columns = vehicle_columns + workers_columns

#Persons
population = ['B01001_001E']
sex = ['B01001_002E', 'B01001_026E']
race = ['B02001_0%02dE'%i for i in range(1,11)]
male_age_columns = ['B01001_0%02dE'%i for i in range(3,26)]
female_age_columns = ['B01001_0%02dE'%i for i in range(27,50)]
# worker_class_columns=['B08128_0%02dE'%i for i in range(1, 7)]
worker_class_columns=['B24080_0%02dE'%i for i in [1,3,6,7,8,9,10,13,16,17,18,19,20]]
# One more column needed for the 16 and under not incuded in the worker_class population
all_columns = population + sex + male_age_columns + female_age_columns  + race
# +worker_class_columns

# =============================================================================
#  Load spatial data
# =============================================================================
#overall zones file
all_zones=json.load(open(ALL_ZONES_PATH))
#sim zones file
sim_zones=json.load(open(SIM_ZONES_PATH))

# =============================================================================
# # Get the ACS data and create the person and HH dataframes
# =============================================================================

# loop through the tracts, getting the data for each and appending to the dataframes
all_counties=list(set([f['properties']['COUNTY'] for f in all_zones['features']]))
h_acs=pd.DataFrame()
p_acs=pd.DataFrame()

for county in all_counties:
    print(county)
    h_acs_c = c.block_group_and_tract_query(block_group_columns,
                    tract_columns, state, county, 
                    merge_columns=['tract', 'county', 'state'],
                    block_group_size_attr="B11001_001E",
                    tract_size_attr="B08201_001E",
                    tract=None, year=2017)
    h_acs=pd.concat([h_acs, h_acs_c])
    p_acs_c = c.block_group_query(all_columns, state, county, tract=None, year=2017)
    p_acs=pd.concat([p_acs, p_acs_c])
h_acs=h_acs.reset_index(drop=True)
p_acs=p_acs.reset_index(drop=True)

# add puma information to the dataframes
h_acs['puma'] = h_acs.apply(lambda row: c.tract_to_puma(row['state'], row['county'], row['tract'])[0], axis=1)
p_acs['puma'] = p_acs.apply(lambda row: c.tract_to_puma(row['state'], row['county'], row['tract'])[0], axis=1)


# Since we queried by county, some of the block_groups may not be in our study area
# remove any entries not in those block groups
all_block_group_ids=list(set([f['properties']['TRACT']+'_'+ 
                              f['properties']['BLKGRP']
                              for f in all_zones['features']]))
h_acs['block_group_id']=h_acs.apply(lambda row: 
    row['tract']+'_'+row['block group'], axis=1)
p_acs['block_group_id']=p_acs.apply(lambda row: 
    row['tract']+'_'+row['block group'], axis=1)
h_acs=h_acs.loc[h_acs['block_group_id'].isin(all_block_group_ids)] 
p_acs=p_acs.loc[p_acs['block_group_id'].isin(all_block_group_ids)] 
# =============================================================================
# # create categorised versions
# =============================================================================
# Households
h_acs_cat = cat.categorize(h_acs, {
#     ("households", "total"): "B11001_001E",
    ("children", "yes"): "B11001_002E",
    ("children", "no"): "B11001_001E - B11001_002E",
    ("income", "lt35"): "B19001_002E + B19001_003E + B19001_004E + "
                        "B19001_005E + B19001_006E + B19001_007E",
    ("income", "gt35-lt100"): "B19001_008E + B19001_009E + "
                        "B19001_010E + B19001_011E + B19001_012E"
                        "+ B19001_013E",
    ("income", "gt100"): "B19001_014E + B19001_015E + B19001_016E"
                        "+ B19001_017E",
    ("cars", "none"): "B08201_002E",
    ("cars", "one"): "B08201_003E",
    ("cars", "two or more"): "B08201_004E + B08201_005E + B08201_006E",
    ("workers", "none"): "B08202_002E",
    ("workers", "one"): "B08202_003E",
    ("workers", "two or more"): "B08202_004E + B08202_005E",
    ("tenure", "owned"): "B25063_001E",
    ("tenure", "rented"): "B25075_001E"
}, index_cols=['NAME'])
h_acs_cat.head()



# Persons
p_acs_cat = cat.categorize(p_acs, {
#     ("population", "total"): "B01001_001E",
    ("age", "19 and under"): "B01001_003E + B01001_004E + B01001_005E + "
                             "B01001_006E + B01001_007E + B01001_027E + "
                             "B01001_028E + B01001_029E + B01001_030E + "
                             "B01001_031E",
    ("age", "20 to 35"): "B01001_008E + B01001_009E + B01001_010E + "
                         "B01001_011E + B01001_012E + B01001_032E + "
                         "B01001_033E + B01001_034E + B01001_035E + "
                         "B01001_036E",
    ("age", "35 to 60"): "B01001_013E + B01001_014E + B01001_015E + "
                         "B01001_016E + B01001_017E + B01001_037E + "
                         "B01001_038E + B01001_039E + B01001_040E + "
                         "B01001_041E",
    ("age", "above 60"): "B01001_018E + B01001_019E + B01001_020E + "
                         "B01001_021E + B01001_022E + B01001_023E + "
                         "B01001_024E + B01001_025E + B01001_042E + "
                         "B01001_043E + B01001_044E + B01001_045E + "
                         "B01001_046E + B01001_047E + B01001_048E + "
                         "B01001_049E", 
     ("race", "white"):   "B02001_002E",
     ("race", "black"):   "B02001_003E",
     ("race", "asian"):   "B02001_005E",
     ("race", "other"):   "B02001_004E + B02001_006E + B02001_007E + "
                          "B02001_008E",
    ("sex", "male"):     "B01001_002E",
    ("sex", "female"):   "B01001_026E",
#     ("worker_class", "private_for_profit"): "B24080_003E+ B24080_013E",
#     ("worker_class", "private_non_profit"): "B24080_006E+ B24080_016E",
#     ("worker_class", "government"): "B24080_007E+ B24080_008E + B24080_009E"
#                                     "B24080_017E+ B24080_018E + B24080_019E",
#     ("worker_class", "self_employed"): "B24080_010E+ B24080_020E",
#     ("worker_class", "not_employed"): "B24080_not_employed + B24080_011E + B24080_021E"
    
}, index_cols=['NAME'])
p_acs_cat.head()

h_acs=h_acs.set_index('NAME')
p_acs=p_acs.set_index('NAME')



# =============================================================================
# Synthsize a population from the PUMS data, with respect to the ACS marginals
# =============================================================================

# define the data we want 
h_pums_cols = ('serialno', 'RT', 'PUMA00', 'PUMA10',  'NP',
                            'TYPE', 'VEH', 'WIF', 'NOC', 'TEN', 'RNTP', 'BDSP', 'YBL', 'HINCP')
p_pums_cols = ('serialno', 'RT', 'PUMA00', 'PUMA10', 'AGEP', 'RAC1P', 'SEX', 'SCHL', 'COW')
all_pumas=list(set(h_acs['puma']))

# do the synthesis one PUMA at a time

all_households=pd.DataFrame()
all_persons=pd.DataFrame()

for puma in all_pumas:
    print(puma)
    # get the block groups in this puma
    this_puma_ind=[i for i in range(len(h_acs)) if h_acs.iloc[i]['puma']==puma]
    #download the pums data
    p_pums=c.download_population_pums(state, puma10=puma, usecols=p_pums_cols)
    h_pums=c.download_household_pums(state, puma10=puma, usecols=h_pums_cols)
    #get the joint distribution of pums data
    h_pums, jd_households = cat.joint_distribution(h_pums,
        cat.category_combinations(h_acs_cat.columns),
        {"cars": cars_cat, "children": children_cat, 
         "income": income_cat, "workers": workers_cat, "tenure": tenure_cat})
    p_pums, jd_persons = cat.joint_distribution(
        p_pums,
        cat.category_combinations(p_acs_cat.columns),
        {"age": age_cat, "sex": sex_cat, "race": race_cat}
    )
    # simulate households and persons for each person in each block-group of this PUMA
    for bg_ind in this_puma_ind:
        zone_name=h_acs_cat.index[bg_ind]
        print(zone_name)
        geoid=state+ h_acs.loc[zone_name,'county']+h_acs.loc[zone_name,'tract']+h_acs.loc[zone_name,'block group']
        print(geoid)
        best_households, best_people, people_chisq, people_p= synthesizer.synthesize(h_acs_cat.iloc[bg_ind].transpose(), p_acs_cat.iloc[bg_ind].transpose(), jd_households, jd_persons, h_pums, p_pums,
                   marginal_zero_sub=.01, jd_zero_sub=.001, hh_index_start=0)
    #     add the puma and bg id to each HH
        best_households['NAME']=zone_name
        best_households['home_geoid']=geoid
    #     add people and HH to global list
        all_households=pd.concat([all_households, best_households])
        all_persons=pd.concat([all_persons, best_people])

# save the full population data frames 
all_households.to_csv(ALL_SYNTH_HH_PATH, index=False)
all_persons.to_csv(ALL_SYNTH_PERSONS_PATH, index=False)

#all_households=pd.read_csv(ALL_SYNTH_HH_PATH)
#all_persons=pd.read_csv(ALL_SYNTH_PERSONS_PATH)
#all_households['home_geoid']=all_households['home_geoid'].astype('str')
# =============================================================================
#  Combine with O-D data to create sample of people living/ working in Sim Area
# =============================================================================

synth_hh_df=all_households
synth_persons_df=all_persons

# create/rename some features
synth_hh_df=synth_hh_df.rename(columns= {'RNTP': 'rent', 'BDSP': 'beds'})
synth_hh_df['built_since_jan2010']=synth_hh_df.apply(lambda row: row['YBL']>=14, axis=1)

synth_persons_df['bach_degree']=synth_persons_df.apply(lambda row: 
    "yes" if row['SCHL']>20 else "no", axis=1)
synth_persons_df['bach_degree']=synth_persons_df.apply(lambda row: 
    "yes" if row['SCHL']>20 else "no", axis=1)

# get lists of zones in the total area and the simulation area        
all_zones=[str(geo) for geo in list(set(synth_hh_df['home_geoid']))]
all_sim_zones=list(set([z.split('US')[1] for 
                        z in sim_zones]))
    
puma_attr=json.load(open(PUMA_ATTR_PATH))

# get and process the LODES O-D data
od=pd.read_csv(OD_PATH)
od['w_block_group']=od.apply(lambda row: str(row['w_geocode'])[0:12], axis=1)
od['h_block_group']=od.apply(lambda row: str(row['h_geocode'])[0:12], axis=1)
od=od.loc[((od['w_block_group'].isin(all_zones)) & (od['h_block_group'].isin(all_zones)))]
# from blocks to block groups
od_bg=od.groupby(['h_block_group', 'w_block_group'] , as_index=False)['S000'].agg('sum')


# define the attributes we need for the person objects
house_cols=['puma10','beds', 'rent', 'tenure','built_since_jan2010', 'home_geoid']
person_cols=['COW', 'bach_degree', 'age', 'sex', 'race']
person_cols_hh=['income', 'children', 'workers', 'tenure', 'HINCP', 'cars']
person_cols.extend(person_cols_hh)
# create empty objects for sim_people
sim_people=[]

# sample people with home or working locations within the simulation area
# create a person object for each including both person and HH attributes

#total_people=0
count=0
for ind, row in od_bg.iterrows():
    count+=1
    if count%1000==0: 
        print('{} of {} '.format(count, len(od_bg)))
    if (((row['h_block_group'] in all_sim_zones) or 
         (row['w_block_group'] in all_sim_zones)) and 
         (row['w_block_group'] in all_zones) and 
         (row['h_block_group'] in all_zones)):
        people_to_sample=row['S000']/sample_factor
#        total_people+=people_to_sample
        people_to_sample_int=int(people_to_sample)
        if people_to_sample%1>random.uniform(0, 1):
            people_to_sample_int+=1
        if people_to_sample_int>0:
            hh_same_home_bg=synth_hh_df.loc[synth_hh_df['home_geoid']==row['h_block_group']]
            hh_same_home_bg=hh_same_home_bg.drop_duplicates()
            people_candidates=synth_persons_df.loc[synth_persons_df['serialno'].isin(
                    hh_same_home_bg['serialno'].values)]
            sampled_persons=people_candidates.sample(n=people_to_sample_int)
            sampled_persons=sampled_persons.merge(hh_same_home_bg, on='serialno', how='left')
            puma=str(sampled_persons.iloc[0]['puma10_x']).zfill(5)
            for s_ind, s_row in sampled_persons.iterrows():
                add_person={col: s_row[col] for col in person_cols}
                add_person['pop_per_sqmile_home']=puma_attr['puma_pop_per_sqm'][puma]/3.86102e-7
                add_person['home_geoid']=row['h_block_group']
                add_person['work_geoid']=row['w_block_group']
                sim_people.append(add_person)

# TODO: vacant houses median income and density should come from ACS

json.dump(sim_people, open(SIM_POP_PATH, 'w'))
# a random sample of people and a random sample of housing units  
# to be used for the vacant houses, people moving house and new population
vacant_houses, floating_people=[], []
frac=vacancy_rate/sample_factor
sample_HHs=synth_hh_df.sample(frac=frac)
#for each:
for ind, row in sample_HHs.iterrows():
    # sample a home location
    puma=str(row['puma10']).zfill(5)
    house_obj={col: row[col] for col in house_cols}
    house_obj['puma_med_income']=puma_attr['med_income'][puma]
    house_obj['puma_pop_per_sqmeter']= puma_attr['puma_pop_per_sqm'][puma]
#    add housing info to housing stock object
    vacant_houses.append(house_obj)
#    get subset of people with this hh id
    if random.randint(1, 6)==1:
        # 3% houses vacant but fewer households seeking
        persons_in_house=synth_persons_df.loc[
                synth_persons_df['serialno']==row['serialno']]
        sample_persons=persons_in_house.sample(n=row['NP'])
        sample_persons=sample_persons.merge(sample_HHs, on='serialno', how='left')
        for p_ind, p_row in sample_persons.iterrows():
            add_person={col: p_row[col] for col in person_cols}
            add_person['pop_per_sqmile_home']=puma_attr['puma_pop_per_sqm'][puma]/3.86102e-7
            od_bg_subset=od_bg.loc[od_bg['h_block_group']==str(row['home_geoid'])]
            add_person['work_geoid']=np.random.choice(
                    od_bg_subset['w_block_group'].values,
                    p=od_bg_subset['S000'].values/sum(od_bg_subset['S000'].values))
            floating_people.append(add_person)

json.dump(vacant_houses, open(VACANT_PATH, 'w'))
json.dump(floating_people, open(FLOATING_PATH, 'w'))                  


