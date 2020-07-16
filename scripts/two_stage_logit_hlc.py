import pandas as pd
import numpy as np
import json
import pickle
import random
from collections import OrderedDict
import time


from transport_network import approx_shape_centroid, get_haversine_distance, Polygon_Location

def get_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    chunks=[]
    for i in range(0, len(lst), n):
        chunks.append(lst[i:i + n])
    return chunks
        
def utility_to_prob(v):
    """ takes a utility vector and predicts probability 
    """
    v = v - v.mean()
    v[v>700] = 700
    v[v<-700] = -700
    expV = np.exp(v)
    p = expV / expV.sum()
    return p
    
def unique_ele_and_keep_order(seq):
    """ same as list(set(seq)) while keep element order 
    """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

class Puma(Polygon_Location):
    def assign_attributes(self, df_row):
        self.med_income=df_row['med_income']
        self.media_norm_rent=df_row['media_norm_rent']
        # TODO: account for scaling of PUMS data
        self.num_houses=df_row['num_houses']
        self.puma_pop_per_sqm=df_row['puma_pop_per_sqm']
        self.num_new_houses=0
        self.base_houses=[]
        self.new_houses=[]
    def reset_new_houses(self):
        self.num_new_houses=0
        self.new_houses=[]


class TwoStageLogitHLC():
    def __init__(self, table_name, city_folder, geogrid, base_vacant_houses):
        self.table_name=table_name
        self.city_folder=city_folder
        self.FITTED_HOME_LOC_MODEL_PATH='./cities/'+self.city_folder+'/models/home_loc_logit.p'
        self.RENT_NORM_PATH='./cities/'+self.city_folder+'/models/rent_norm.json'
        # PUMAs        
        self.PUMA_SHAPE_PATH='./cities/'+city_folder+'/raw/PUMS/pumas.geojson'
        self.PUMAS_INCLUDED_PATH='./cities/'+city_folder+'/raw/PUMS/pumas_included.json'
        self.PUMA_ATTR_PATH = './cities/'+city_folder+'/models/puma_attr.json'
        self.top_n_pumas=5
        try:
            self.PUMA_model=pickle.load( open(self.FITTED_HOME_LOC_MODEL_PATH, "rb" ))['home_loc_mnl_PUMAs']
            self.hh_model=pickle.load( open(self.FITTED_HOME_LOC_MODEL_PATH, "rb" ))['home_loc_mnl_hh']
            self.rent_normalisation=json.load(open(self.RENT_NORM_PATH))
        except:
            print('Home location model not yet trained. Starting training')
            self.train()
            self.fitted_model=pickle.load( open(self.FITTED_HOME_LOC_MODEL_PATH, "rb" ) )
            self.rent_normalisation=json.load(open(self.RENT_NORM_PATH))
        self.build_pumas()
        self.assign_base_houses_to_pumas(base_vacant_houses)
        self.create_geogrid_index_to_puma_index(geogrid)
            
    def build_pumas(self):
        # build pumas from geojson. adds attributes from df
        # finds geogrid cells in each puma
        pumas=[]
        puma_geo=json.load(open(self.PUMA_SHAPE_PATH))
#        puma_order=[f['properties']['PUMACE10'] for f in puma_geo['features']]
        puma_included=json.load(open(self.PUMAS_INCLUDED_PATH)) 
        puma_df = pd.DataFrame(json.load(open(self.PUMA_ATTR_PATH, 'r')))
        # if the shape type is "Polygon", [0][0] would return only a point
        for feature in puma_geo['features']:
            geoid=feature["properties"]["GEOID10"][2:]
            if geoid in puma_included:
                new_puma=Puma(geometry=feature['geometry'], 
                     area_type='large_zone',
                     in_sim_area=False,
                     geoid=geoid)
                new_puma.assign_attributes(puma_df.loc[geoid])
                pumas.append(new_puma)
        self.pumas=pumas
        self.puma_order=[puma.geoid for puma in pumas]
        
    def assign_base_houses_to_pumas(self, base_vacant_houses):
        for house in base_vacant_houses:
            self.pumas[self.puma_order.index(str(house.puma10).zfill(5))].base_houses.append(house)

    def create_geogrid_index_to_puma_index(self, geogrid):
        self.geogrid_index_to_puma_index={}
        for ind_cell, cell in enumerate(geogrid.cells):
            ind_puma, puma=cell.get_containing_poly(self.pumas)
            self.geogrid_index_to_puma_index[ind_cell]=ind_puma
            
    def update_pumas(self, pop):
        # goes through list of new houses and updates the corresponding pumas
        for puma in self.pumas:
            puma.reset_new_houses()
        for house in pop.new_vacant:
            if house.loc.area_type=='grid':
                puma_index=self.geogrid_index_to_puma_index[house.loc.geogrid_id]
                self.pumas[puma_index].num_new_houses+=1
                self.pumas[puma_index].new_houses.append(house)
                
    def create_long_record_puma(self, person, puma, person_id):
        # TODO: dont use work_ll- lookup the location from geogrid locations and zone locations
        """ takes a puma object and a household object and 
        creates a row for the MNL long data frame 
        """
        return   {'puma_pop_per_sqm': puma.puma_pop_per_sqm,
                  'income_disparity': np.abs(person.HINCP - puma.med_income),
                  'work_dist': get_haversine_distance(person.work_loc.centroid, puma.centroid),
                  'media_norm_rent': puma.media_norm_rent,
                  'num_houses': puma.num_houses + puma.num_new_houses,
    #              'entertainment_den': puma['entertainment_den'],
    #              'medical_den': puma['medical_den'],
    #              'school_den': puma['school_den'],
                  'custom_id': person_id,
                  'choice_id': puma.geoid} 
    
    def create_long_record_house(self, person, house, choice_id, person_id):
        """ takes a house object and a household object and 
        creates a row for the MNL long data frame 
        """
        beds=min(3, max(1, house.beds))
        norm_rent=(house.rent-self.rent_normalisation['mean'][str(int(beds))])/self.rent_normalisation['std'][str(int(beds))]
        record = {'norm_rent': norm_rent,
                'built_since_jan2010': house.built_since_jan2010,
                'bedrooms': beds,
                'income': person.HINCP,
                'custom_id': person_id,
                'choice_id': choice_id,
                'actual_house_id':house.house_id} 
        nPersons = 0
        if person.workers == 'one':
            nPersons += 1
        elif person.workers == 'two or more':
            nPersons += 2
        if person.children == 'yes':
            nPersons += 1
        if nPersons == 0:
            nPersons = 1
        record['nPersons'] = nPersons
        return record
        
    #

        
    def pylogit_pred(self,  data, modelDict, customIDColumnName, even=True):
        """ predicts probabilities for pylogit models,
        this function is needed as the official 'predict' method cannot be used when the choice sets 
        in predicting is not exactly the same as in trainning,
        argument even: whether each choice situation has the same number of alternatives
        """
        # fectch variable names and parameters 
        if modelDict['just_point']:
            params, varnames = modelDict['params'], modelDict['var_names']
        else:
            params, varnames = list(modelDict['model'].coefs.values), list(modelDict['model'].coefs.index)
        # calc utilities
        data['utility'] = 0
        for varname, param in zip(varnames, params):
            data['utility'] += data[varname] * param
        # calc probabilities given utilities
        # if every choice situation has the same number of alternatives, use matrix, otherwise use list comprehension
        if even:
            numChoices = len(set(data[customIDColumnName]))
            v = np.array(data['utility']).copy().reshape(numChoices, -1)
            v = v - v.mean(axis=1, keepdims=True)  
            v[v>700] = 700
            v[v<-700] = -700
            expV = np.exp(v)
            p = expV / expV.sum(axis=1, keepdims=True)
            return p.flatten()
        else:
            uniqueCustomIDs = unique_ele_and_keep_order(data[customIDColumnName])
            vArrayList = [np.array(data.loc[data[customIDColumnName]==id, 'utility']) for id in uniqueCustomIDs]
            pArrayList = [utility_to_prob(v) for v in vArrayList]
            return [pElement for pArray in pArrayList for pElement in pArray ]
        
    def home_location_choices(self, pop):
        """ takes the population
        identifies the vacant houses and homeless persons
        chooses a housing unit for each person
        modifies the person objects in place
        """
        print('\t Home Location Choice: stage 1')
        self.update_pumas(pop)
#        valid_pumas=[p for p in self.pumas if ((len(p.base_houses)>0) or (len(p.new_houses)>0))]
        # create long data for puma model
        chosen_pumas=[]
        all_floating_persons=pop.new+pop.base_floating
        all_vacant_houses=pop.base_vacant+pop.new_vacant
        
        for p in all_floating_persons:
            p.assign_home_location(None)

        units_available_by_puma={}
        for p in self.pumas:
            total_available=2*len(p.base_houses)+len(p.new_houses)
            units_available_by_puma[p.geoid]=total_available
                
        chunk_size=50
        # split floating_persons into lists of ~50 persons
        chunks=get_chunks(all_floating_persons, chunk_size)
        # for each chunk
        for ch in chunks:
            valid_pumas=[p for p in self.pumas if units_available_by_puma[p.geoid]>0]
            if len(valid_pumas)>0:
                # create long data            
                # perform prediction
                # decrement available houses  
                long_data_puma = []
                for ind_p, p in enumerate(ch):
                    for puma in valid_pumas:
                        this_sample_long_record_puma = self.create_long_record_puma(p, puma, ind_p)
                        long_data_puma.append(this_sample_long_record_puma)
                long_df_puma = pd.DataFrame(long_data_puma)
                long_df_puma['predictions'] = self.pylogit_pred(long_df_puma, self.PUMA_model,'custom_id', even=True)   
                if self.top_n_pumas is None:
                    custom_specific_long_df_puma = {custom_id: group for custom_id, group in long_df_puma[['custom_id', 'choice_id', 'predictions']].groupby('custom_id')}
                else:
                    long_df_puma_sorted = long_df_puma[['custom_id', 'choice_id', 'predictions']].sort_values(['custom_id','predictions'], ascending=[True, False])
                    custom_specific_long_df_puma = {custom_id: group.iloc[:self.top_n_pumas,:] for custom_id, group in long_df_puma_sorted.groupby('custom_id')}
    
                for p_ind in range(len(ch)):
                    if self.top_n_pumas is None:
                        house_puma=np.random.choice(custom_specific_long_df_puma[p_ind]['choice_id'], p=custom_specific_long_df_puma[p_ind]['predictions'])
                    else:
                        house_puma=np.random.choice(custom_specific_long_df_puma[p_ind]['choice_id'], 
                                                    p=custom_specific_long_df_puma[p_ind]['predictions'] / custom_specific_long_df_puma[p_ind]['predictions'].sum())
                    chosen_pumas.append(house_puma)
                for puma in units_available_by_puma:
                    units_available_by_puma[puma]-=chosen_pumas.count(puma)
            else:
                chosen_pumas.extend([None for p in ch])
                
        # stage2: housing unit choice
        print('\t Home Location Choice: stage 2')
        long_data_house = []
        even = True  # use "even" to monitor if every choice situation has the same number of alternatives
        for ind_p, p in enumerate(all_floating_persons):
            if chosen_pumas[ind_p] is not None:
                ind_puma=self.puma_order.index(chosen_pumas[ind_p])
                houses_in_puma = self.pumas[ind_puma].base_houses+ self.pumas[ind_puma].new_houses
                if len(houses_in_puma) < 9:
                    house_alts = houses_in_puma 
                    even = False
                else:
                    house_alts = random.sample(houses_in_puma, 9)
                for hi, h in enumerate(house_alts):
                    this_sample_long_record_house = self.create_long_record_house(p, h, hi+1, ind_p)
                    long_data_house.append(this_sample_long_record_house)             
        long_df_house = pd.DataFrame(long_data_house)
        long_df_house.loc[long_df_house['norm_rent'].isnull(), 'norm_rent']=0
        long_df_house['income_norm_rent'] = long_df_house['income'] * long_df_house['norm_rent']
        long_df_house['income_bedrooms'] = long_df_house['income'] * long_df_house['bedrooms']
        long_df_house['nPerson_bedrooms'] = long_df_house['nPersons'] * long_df_house['bedrooms']
        long_df_house['predictions'] = self.pylogit_pred(long_df_house, self.hh_model, 'custom_id', even=even)
        custom_specific_long_df_house = {custom_id: group for custom_id, group in long_df_house[['custom_id', 'actual_house_id', 'predictions']].groupby('custom_id')}
        for p_ind in set(long_df_house['custom_id']):
            chosen_house_id = np.random.choice(custom_specific_long_df_house[p_ind]['actual_house_id'], 
                                        p=custom_specific_long_df_house[p_ind]['predictions'])             
            all_floating_persons[p_ind].assign_home_location(all_vacant_houses[chosen_house_id].loc)
        
        
    def train(self):
        import pylogit as pl
        state_codes={'Detroit': 'mi', 'Boston': 'ma'}
        state_fips={'Detroit': '26', 'Boston': '25'}
        NUM_ALTS=8
        sample_size=5000
        
        PUMA_POP_PATH='./cities/'+self.city_folder+'/raw/ACS/ACS_17_1YR_B01003/population.csv'
        PUMS_HH_PATH='./cities/'+self.city_folder+'/raw/PUMS/csv_h'+state_codes[self.city_folder]+'/ss16h'+state_codes[self.city_folder]+'.csv'
        PUMS_POP_PATH='./cities/'+self.city_folder+'/raw/PUMS/csv_p'+state_codes[self.city_folder]+'/ss16p'+state_codes[self.city_folder]+'.csv'
#        POI_PATH = './cities/'+self.city_folder+'/raw/OSM/poi.geojson'
        PUMA_TO_POW_PUMA_PATH='./puma_to_pow_puma.csv'
        
        
        hh=pd.read_csv(PUMS_HH_PATH)
        pop = pd.read_csv(PUMS_POP_PATH)
        hh['PUMA']=hh.apply(lambda row: str(int(row['PUMA'])).zfill(5), axis=1)
        pop['PUMA']=pop.apply(lambda row: str(int(row['PUMA'])).zfill(5), axis=1)
        pop['POWPUMA']=pop.apply(lambda row: str(int(row['POWPUMA'])).zfill(5) 
                                if not np.isnan(row['POWPUMA']) else 'NaN', axis=1)
        
#        all_PUMAs=list(set(hh['PUMA']))
        pumas_included=json.load(open(self.PUMAS_INCLUDED_PATH))                                         # For the whole MI
        pumas_shape=json.load(open(self.PUMA_SHAPE_PATH))
        pumas_order=[f['properties']['PUMACE10'] for f in pumas_shape['features']]
                  
        puma_pop=pd.read_csv(PUMA_POP_PATH)
        puma_pop['PUMA']=puma_pop.apply(lambda row: str(row['GEO.id2'])[2:].zfill(5), axis=1)
        puma_pop=puma_pop.set_index('PUMA')
        puma_pop=puma_pop.iloc[1:] # get rid of second header line
        
        
        # identify recent movers and vacant houses                                            
        hh_vacant_for_rent=hh[(hh['VACS']==1) & (hh['PUMA'].isin(pumas_included))].copy()          
        hh_rented=hh[(hh['TEN']==3) & (hh['PUMA'].isin(pumas_included))].copy()                                                      
        renters_recent_move=hh_rented[hh_rented['MV']==1].copy()     
        
        # get the area of each PUMA
        puma_land_sqm={str(int(f['properties']['PUMACE10'])).zfill(5): f['properties']['ALAND10']
                        for f in pumas_shape['features']}
        
        # =============================================================================
        # Distance Matrix
        # =============================================================================
        # get the distance between each puma and each pow-puma
        # first get a lookup between pow-pumas and pumas
        # because we only have the shapes of the PUMAS
        pow_puma_df=pd.read_csv(PUMA_TO_POW_PUMA_PATH, skiprows=1, header=1)
        pow_puma_df_state=pow_puma_df.loc[pow_puma_df[
                'State of Residence (ST)']==state_fips[self.city_folder]].copy()
        pow_puma_df_state['POW_PUMA']=pow_puma_df_state.apply(
                lambda row: str(int(row['PWPUMA00 or MIGPUMA1'])).zfill(5), axis=1)
        pow_puma_df_state['PUMA']=pow_puma_df_state.apply(
                lambda row: str(int(row['PUMA'])).zfill(5), axis=1)
        all_pow_pumas=set(pow_puma_df_state['POW_PUMA'])
        pow_puma_to_puma={}
        for p in all_pow_pumas:
            pow_puma_to_puma[p]=list(pow_puma_df_state.loc[
                    pow_puma_df_state['POW_PUMA']==p, 'PUMA'].values)
        
        # find the centroid of each puma
        puma_centroids={}
        pow_puma_centroids={}
        for puma in set(pow_puma_df_state['PUMA']):
            centr=approx_shape_centroid(pumas_shape['features'][pumas_order.index(puma)]['geometry'])
            puma_centroids[puma]=centr
            
        # and each pow-puma
        all_pow_pumas=set(pow_puma_df_state['POW_PUMA'])
        
        for pow_puma in all_pow_pumas:
            pumas=pow_puma_to_puma[pow_puma]
            puma_centr=[puma_centroids[puma] for puma in pumas]
            # TODO, shold be weighted by area- ok if similar size
            pow_puma_centroids[pow_puma]=[np.mean([pc[0] for pc in puma_centr]),
                                          np.mean([pc[1] for pc in puma_centr])]
        dist_mat={}
        for puma in puma_centroids:
            dist_mat[puma]={}
            for pow_puma in pow_puma_centroids:
                dist = get_haversine_distance(
                        puma_centroids[puma], pow_puma_centroids[pow_puma])
                if dist > 0:
                    dist_mat[puma][pow_puma] = dist
                else:
                    dist_mat[puma][pow_puma] = np.sqrt(puma_land_sqm[puma] / np.pi) # set inner distance to quasi-radius
        
        
        # build the PUMA aggregate data data frame
        median_income_by_puma=hh.groupby('PUMA')['HINCP'].median()
        #TODO: get more zonal attributes such as access to employment, amenities etc.
        
        puma_obj=[{'PUMA':puma,
                   'med_income':median_income_by_puma.loc[puma],
                   'puma_pop_per_sqm':float(puma_pop.loc[puma]['HD01_VD01'])/puma_land_sqm[puma]
                   } for puma in pumas_included]
        
        puma_attr_df=pd.DataFrame(puma_obj)
        
        #for poiField in poiFields:
        #    puma_attr_df[poiField] = puma_attr_df.apply(lambda row: puma_poi_dict[row['PUMA']][poiField], axis=1)
        puma_attr_df=puma_attr_df.set_index('PUMA')
        
        # create features at property level
        # normalise rent stratifying by bedroom number
        renters_recent_move.loc[renters_recent_move['BDSP']>2, 'BDSP']=3            # change [the number of bedroom] >2 to 3
        renters_recent_move.loc[renters_recent_move['BDSP']<1, 'BDSP']=1            # change [the number of bedroom] <1 to 1
        hh_vacant_for_rent.loc[hh_vacant_for_rent['BDSP']>2, 'BDSP']=3          
        hh_vacant_for_rent.loc[hh_vacant_for_rent['BDSP']<1, 'BDSP']=1
        rent_mean={}
        rent_std={}
        for beds in range(1,4):
            rent_mean[beds]=renters_recent_move.loc[renters_recent_move['BDSP']==beds, 'RNTP'].mean()
            rent_std[beds]=renters_recent_move.loc[renters_recent_move['BDSP']==beds, 'RNTP'].std()
            
        for df in [renters_recent_move, hh_vacant_for_rent]:
            df['norm_rent']=df.apply(
                lambda row: (row['RNTP']-rent_mean[row['BDSP']])/rent_std[row['BDSP']], axis=1)
            # Age of building
            df['built_since_jan2010']=df.apply(lambda row: row['YBL']>=14, axis=1)
            df['puma_pop_per_sqmeter']=df.apply(lambda row: puma_attr_df.loc[row['PUMA']]['puma_pop_per_sqm'], axis=1)
            df['med_income']=df.apply(lambda row: puma_attr_df.loc[row['PUMA']]['med_income'], axis=1)  
        all_rooms_available = pd.concat([hh_vacant_for_rent, renters_recent_move], axis=0) 
        median_norm_rent = all_rooms_available.groupby('PUMA')['norm_rent'].median()
        puma_attr_df['media_norm_rent'] =  puma_attr_df.apply(lambda row: median_norm_rent[row.name], axis=1)
        
        # num of avaiable housing units in each PUMA
        num_available_houses_in_puma = hh_vacant_for_rent.groupby('PUMA')['SERIALNO'].count()
        puma_attr_df['num_houses'] = puma_attr_df.apply(lambda row: num_available_houses_in_puma[row.name], axis=1)
        
        renters_recent_move=renters_recent_move[['SERIALNO', 'PUMA','HINCP',  'norm_rent', 'RNTP', 'built_since_jan2010', 'puma_pop_per_sqmeter', 'med_income', 'BDSP', 'NP']]
        hh_vacant_for_rent=hh_vacant_for_rent[['PUMA', 'HINCP', 'norm_rent', 'RNTP','built_since_jan2010', 'puma_pop_per_sqmeter', 'med_income', 'BDSP']]
         
        rent_normalisation={"mean": rent_mean, "std": rent_std}   
        
        home_loc_mnl = {'home_loc_mnl_PUMAs': {}, 'home_loc_mnl_hh': {}}    
        
        # =============================================================================
        # Model Estimation
        # First stage: choice model on PUMA level
        # =============================================================================
        long_data_PUMA = pd.DataFrame()
        print('\n\n[info] Preparing long data for PUMA-level choice.')
        time1 = time.time()
        numPUMAs = puma_attr_df.shape[0]
        ind = 0
        for ind_actual, row_actual in renters_recent_move.iterrows():
            if ind >= sample_size:
                break
            householdID = row_actual['SERIALNO']
            places_of_work = set(pop.loc[pop['SERIALNO']==householdID, 'POWPUMA'])
            places_of_work = [x for x in places_of_work if x in all_pow_pumas]
            if len(places_of_work):
                this_sample_puma_attr_df = puma_attr_df.copy()
                this_sample_puma_attr_df['custom_id'] = ind_actual * np.ones(numPUMAs, dtype=np.int8)
                this_sample_puma_attr_df['choice_id'] = list(puma_attr_df.index)
                this_sample_puma_attr_df['choice'] = np.zeros(numPUMAs)
                this_sample_puma_attr_df['hh_income'] = row_actual['HINCP']
                this_sample_puma_attr_df.loc[this_sample_puma_attr_df['choice_id']==row_actual['PUMA'], 'choice'] = 1
                this_sample_puma_attr_df['work_dist'] = [np.mean([dist_mat[puma][pow_puma] 
                                                      for pow_puma in places_of_work]) for puma in list(puma_attr_df.index)]
                long_data_PUMA = pd.concat([long_data_PUMA, this_sample_puma_attr_df], axis=0)
                ind += 1
        
        long_data_PUMA['income_disparity']=long_data_PUMA.apply(lambda row: np.abs(row['hh_income']-row['med_income']), axis=1)
        time2 = time.time()
        print('[info] Long data for PUMA-level choice finished. Elapsed time: {} seconds'.format(time2-time1))
#        long_data_PUMA.to_csv('./cities/'+self.city_folder+'/clean/logit_data_long_form/logit_data_PUMA.csv', index=False)
        
        choiceModelPUMA_spec = OrderedDict()
        choiceModelPUMA_names = OrderedDict()
        choiceModelPUMAsRegressors = ['puma_pop_per_sqm', 'income_disparity', 'work_dist', 'media_norm_rent', 'num_houses'] + [x for x in list(long_data_PUMA.columns) if x.endswith('_den')]
        for var in choiceModelPUMAsRegressors:
            choiceModelPUMA_spec[var] = [list(set(long_data_PUMA['choice_id']))]
            choiceModelPUMA_names[var] = [var]
            
        home_loc_mnl_PUMAs = pl.create_choice_model(data=long_data_PUMA,
                                                alt_id_col='choice_id',
                                                obs_id_col='custom_id',
                                                choice_col='choice',
                                                specification=choiceModelPUMA_spec,
                                                model_type="MNL",
                                                names=choiceModelPUMA_names)
        print('\n[info] Fitting Upper Level Model')
        numCoef=sum([len(choiceModelPUMA_spec[s]) for s in choiceModelPUMA_spec])
        
        # pylogit may encounter memory error in calculating Hessiann matrix for S.E. in this model, if so, switch to noHessian approach and only do point estimation.
        try:
            home_loc_mnl_PUMAs.fit_mle(np.zeros(numCoef))
            print(home_loc_mnl_PUMAs.get_statsmodels_summary())
            home_loc_mnl['home_loc_mnl_PUMAs'] = {'just_point': False, 'model': home_loc_mnl_PUMAs}
        except:
            home_loc_mnl_PUMAs_result = home_loc_mnl_PUMAs.fit_mle(np.zeros(numCoef), just_point=True)
            params = home_loc_mnl_PUMAs_result['x']
            print('\nLogit model parameters:\n---------------------------')
            for varname, para in zip(home_loc_mnl_PUMAs.ind_var_names, params):
                print('{}: {:4.6f}'.format(varname, para))
            home_loc_mnl['home_loc_mnl_PUMAs'] = {'just_point': True, 'model': home_loc_mnl_PUMAs, 'params': params, 'var_names': home_loc_mnl_PUMAs.ind_var_names}
        
        
        # =============================================================================
        # Model Estimation
        # Second stage: choice model on HH level
        # =============================================================================
        random.seed(1)
        print('\n\n[info] Preparing long data for HH-level choice.')
        long_data_hh_obj = []
        ind=0
        time1 = time.time()
        for ind_actual, row_actual in renters_recent_move.iterrows():
            if ind >= sample_size:
                break
            cid=1
            householdID = row_actual['SERIALNO']
            thisPUMA = row_actual['PUMA']
            places_of_work = set(pop.loc[pop['SERIALNO']==householdID, 'POWPUMA'])
            places_of_work = [x for x in places_of_work if x in all_pow_pumas]
            
            if len(places_of_work):
                # this is the real choice
                choiceObs={'custom_id':ind, # identify the individual
                           'choice_id':cid, # fake choice identifier- shouldn't matter if no ASC
                           'choice':1,
                           'rent':row_actual['RNTP'],
                           'norm_rent':row_actual['norm_rent'],
                           'puma':row_actual['PUMA'],
                           'built_since_jan2010':int(row_actual['built_since_jan2010']),
                           'bedrooms': row_actual['BDSP'],
                           'hh_income':row_actual['HINCP'],
                           'nPersons':row_actual['NP']
                           }
        
                cid+=1
                long_data_hh_obj.append(choiceObs)
                
                hh_vacant_for_rent_in_this_PUMA = hh_vacant_for_rent.loc[hh_vacant_for_rent['PUMA']==thisPUMA].copy()
                
                for i in range(NUM_ALTS):
                    selected=random.choice(range(len(hh_vacant_for_rent_in_this_PUMA)))
                    alt_obs={'custom_id':ind,# identify the individual
                             'choice_id':cid, # fake choice identifier- shouldn't matter if no ASC
                             'choice':0,
                             'rent':hh_vacant_for_rent_in_this_PUMA.iloc[selected]['RNTP'],
                             'norm_rent':hh_vacant_for_rent_in_this_PUMA.iloc[selected]['norm_rent'],
                             'puma':hh_vacant_for_rent_in_this_PUMA.iloc[selected]['PUMA'],
                             'built_since_jan2010':int(hh_vacant_for_rent_in_this_PUMA.iloc[selected]['built_since_jan2010']),
                             'bedrooms': hh_vacant_for_rent_in_this_PUMA.iloc[selected]['BDSP'],
                             'hh_income':row_actual['HINCP'],
                             'nPersons':row_actual['NP']
                             }
                    cid+=1
                    long_data_hh_obj.append(alt_obs)
                ind+=1
        
        long_data_hh=pd.DataFrame(long_data_hh_obj)  
        long_data_hh['income_norm_rent'] = long_data_hh['hh_income'] * long_data_hh['norm_rent']
        long_data_hh['income_bedrooms'] = long_data_hh['hh_income'] * long_data_hh['bedrooms']
        long_data_hh['nPerson_bedrooms'] = long_data_hh['nPersons'] * long_data_hh['bedrooms']
        time2 = time.time()
        print('\n[info] Long data for HH-level choice finished. Elapsed time: {} seconds\n'.format(time2-time1))
        long_data_hh.to_csv('./cities/'+self.city_folder+'/clean/logit_data_long_form/logit_data_hh.csv', index=False)
        
        choiceModelHH_spec = OrderedDict()
        choiceModelHH_names = OrderedDict()
        choiceModelHHRegressors = ['norm_rent', 'built_since_jan2010', 'bedrooms', 'income_norm_rent', 'income_bedrooms', 'nPerson_bedrooms']
        
        for var in choiceModelHHRegressors:
            choiceModelHH_spec[var] = [list(set(long_data_hh['choice_id']))]
            choiceModelHH_names[var] = [var]
        
        home_loc_mnl_hh = pl.create_choice_model(data=long_data_hh,
                                                alt_id_col='choice_id',
                                                obs_id_col='custom_id',
                                                choice_col='choice',
                                                specification=choiceModelHH_spec,
                                                model_type="MNL",
                                                names=choiceModelHH_names)
        
        # Specify the initial values and method for the optimization.
        print('\n[info] Fitting Model')
        numCoef=sum([len(choiceModelHH_spec[s]) for s in choiceModelHH_spec])
        
        try:
            home_loc_mnl_hh.fit_mle(np.zeros(numCoef))
            print(home_loc_mnl_hh.get_statsmodels_summary())
            home_loc_mnl['home_loc_mnl_hh'] = {'just_point': False, 'model': home_loc_mnl_hh}
        except:
            home_loc_mnl_hh_result = home_loc_mnl_hh.fit_mle(np.zeros(numCoef), just_point=True)
            params = home_loc_mnl_hh_result['x']
            print('\nLogit model parameters:\n---------------------------')
            for varname, para in zip(home_loc_mnl_hh.ind_var_names, params):
                print('{}: {:4.6f}'.format(varname, para))
            home_loc_mnl['home_loc_mnl_hh'] = {'just_point': True, 'model': home_loc_mnl_hh, 'params': params, 'var_names': home_loc_mnl_hh.ind_var_names}
                
        # save models to file
        pickle.dump(home_loc_mnl, open(self.FITTED_HOME_LOC_MODEL_PATH, 'wb'))
        json.dump(rent_normalisation, open(self.RENT_NORM_PATH, 'w'))
        puma_attr_df.to_json(self.PUMA_ATTR_PATH)