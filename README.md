# CS_Mobility_Service
A web service providing mobility simulations for CityScope projects. This project provides a mobility simulation framework which performs microsimulations of mobility behavior. It can be used to evaluate static scenarios or can be deployed as a web-service for a CityScope model which provides updated mobility simulations each time new inputs are available. The model also calculates mobility-related urban indicators based on the simulatino results. Each model is an object of the class MobilityModel.

![viz](./images/mob_sim_viz.gif)

## MobilityModel Class
The MobilityModel class creates a model of the target area and simulates the mobility behaviour of a sample of the population. The sample size is controlled by the scale_factor parameter of the MobilityModel class. The following inputs must be provided on initialisation:
- table_name: the name of the table end-point on city_IO
- city_folder: the folder where the input data for this city are located.
- seed: for the random number generator
- host: the address where [city_IO!](https://github.com/CityScope/CS_CityIO) is hosted
```
from mobility_service_model import MobilityModel
this_model=MobilityModel('corktown_dev', 'Detroit', seed=42, host=host)
```

Before the simulation can be run, the necessary prediction models must be assigned to the MobilityModel using the following methods:
- assign_mode_choice_model
- assign_home_location_choice_model
- assign_activity_scheduler()

## Mode Choice Model
A mode choice model can be created using the NhtsModeLogit class. This model predicts one of four transport modes for each trip:
| 0       | 1       | 2       | 3       |
|---------|---------|---------|---------|
| driving | cycling | walking | transit |

```
from mode_choice_nhts import NhtsModeLogit
mode_choice_model=NhtsModeLogit(table_name='corktown', city_folder='Detroit')
this_model.assign_mode_choice_model(mode_choice_model)

```
If the model has not yet been trained, the training will start on initialisation. The trained model will then be saved locally so that training will not be necessary the next time the model is created. The trained parameters of the model can also be checked or manually changed if necessary.

```
new_ASCs = {
#        'ASC for cycle': -0.9, 
            'ASC for PT': -0.9, 
#            'ASC for walk': 2.9
}
initial_ASCs= {param:mode_choice_model.logit_model['params'][param] for param in new_ASCs}
mode_choice_model.set_logit_model_params(new_ASCs)

for param in new_ASCs:
    print('Modified {} from {} to {}'.format(param, initial_ASCs[param], mode_choice_model.logit_model['params'][param]))

```

## Activity Scheduler
The activity scheduler creates a list of activities, with start times and locations for each activity. It can be created using the ActivityScheduler class. If the model has not yet been trained, the training will start on initialisation. The trained model will then be saved locally so that training will not be necessary the next time the model is created.

```
as_model=ActivityScheduler(model=this_model)
this_model.assign_activity_scheduler(as_model)
```
## Home Location Choice Model
The home location choice model is responsible for choosing home locations for agents. It can be created using the TwoStageLogitHLC class. If the model has not yet been trained, the training will start on initialisation. The trained model will then be saved locally so that training will not be necessary the next time the model is created.
```
from two_stage_logit_hlc import TwoStageLogitHLC
hlc_model=TwoStageLogitHLC(table_name='corktown', city_folder='Detroit', 
                         geogrid=this_model.geogrid, 
                         base_vacant_houses=this_model.pop.base_vacant)
this_model.assign_home_location_choice_model(hlc_model)
```

## Handler Class
The handler takes care of high-level interactions with the model such as getting data from and posting data to cityIO for live deployments. When listen_city_IO() is run, the handler will begin regularly checking the geogrid_data hash id for the specified table. Each time a change in the geogrid_data is detected, a simulation update is triggered and the results are posted back to city_IO. Example use:
```
from mobility_service_model import MobilityModel
from activity_scheduler import ActivityScheduler
from mode_logit_nhts import NhtsModeLogit
from two_stage_logit_hlc import TwoStageLogitHLC
from cs_handler import CS_Handler


this_model=MobilityModel('corktown', 'Detroit')

this_model.assign_activity_scheduler(ActivityScheduler(model=this_model))

this_model.assign_mode_choice_model(NhtsModeLogit(table_name='corktown', city_folder='Detroit'))

this_model.assign_home_location_choice_model(
        TwoStageLogitHLC(table_name='corktown', city_folder='Detroit', 
                         geogrid=this_model.geogrid, 
                         base_vacant_houses=this_model.pop.base_vacant))


handler=CS_Handler(this_model)
handler.listen_city_IO()

```


## New Transport Modes
The mode choice model may be extended to include mode options not included in the model training process. The first step is to create specifications for the new modes. The code below creates 2 new mode specifications. The "attrs" object defines the costs of the mode in relation to other exisitng modes. eg. The "active_time_minutes" for the micro-mobility mode will be equal to that of cycling + 5 minutes (walking). The "vehicle_time_minutes" of the new shuttle mode will be equal to that of driving and the "active_time_minutes" will be 5. We can also define the availability conditions of the new modes. eg. micromobility is only available for trips where the external time (to and from the simulation area) is less than 1.5 miles. The logit parameters can be copied from an existing mode using the 'copy' attribute. These can also be controlled more finely as detailed later on.
```
new_mode_specs=[
  {"name":  "micromobility", 
    "attrs":{
      "active_time_minutes": "c*1+5",
      "cost":0.2},  
    "copy": "cycle",
    "copy_route": "cycling","activity": "cycling","speed_m_s": 4.167,
    "co2_emissions_kg_met": 0,"fixed_costs": {},
    "internal_net": "active",
    "availability": "external_network_dist_mile<=1.5"},
  {"name": "shuttle", 
    "attrs":{
      "vehicle_time_minutes": "d*1",
      "active_time_minutes": 5,
      "cost":5},  
    "copy": "PT",
    "copy_route": "driving","activity": "pt","speed_m_s": 8.33,
    "co2_emissions_kg_met": 0.000066,"fixed_costs": {},
    "internal_net": "pt",
    "availability": "external_network_dist_mile<=3"
  }
]
```

The prediction of new modes is done using a generalised logit model with a nesting structure and nesting parameters specified by the user. The following code creates 2 nests: a PT-like nest containing PT, micromobility and shuttle, and a walk-like nest containing walking and micromobility. The lambda variables are independence parameters for each nest.

```
lambda_PT= 0.65
lambda_walk= 0.29
nests_spec=[{'name': 'PT_like', 'alts':['micromobility', 'PT', 'shuttle'], 'lambda':lambda_PT},
            {'name': 'walk_like', 'alts':['micromobility','walk'], 'lambda':lambda_walk}

```
We can also make changes to any of the logit parameters for the new modes. In the code below, we assign parameters values in two ways: 
- an ACS is specified directly for each new mode.
- Since micromobility is considered similar to both walking and PY, we calculate the logit parameters for micromobility as a weighted sum of the parameters for the 2 exisitng modes (apart from the ASC which was specified directly) To deice the relative weighting of PT and walk, we calibrated a parameter beta_similarity_PT which denotes the percentage similarity to PT.
Suitable values for the both types of parameter can be found using the notebook fit_new_params.ipynb.

```
new_logit_params = {}
new_logit_params['ASC for micromobility'] =  2.63
new_logit_params['ASC for shuttle'] = 2.33

beta_similarity_PT= 0.5

crt_logit_params = mode_choice_model.logit_model['params']
for g_attr in mode_choice_model.logit_generic_attrs:
    new_logit_params['{} for micromobility'.format(g_attr)] = \
        crt_logit_params['{} for PT'.format(g_attr)] * beta_similarity_PT + \
        crt_logit_params['{} for walk'.format(g_attr)] * (1-beta_similarity_PT)

```

We can then create the mobility model as before. The set_new_modes() function is used to add the modes we have just specified to the mode choice model. We must also pass the new_logit_params object to the Handler on initialisation.

```
this_model=MobilityModel(table_name, 'Detroit', seed=0, host=host)

this_model.assign_activity_scheduler(ActivityScheduler(model=this_model))

this_model.assign_mode_choice_model(mode_choice_model)

this_model.assign_home_location_choice_model(
        TwoStageLogitHLC(table_name=table_name, city_folder='Detroit', 
                         geogrid=this_model.geogrid, 
                         base_vacant_houses=this_model.pop.base_vacant))

this_model.set_prop_electric_cars(0.5, co2_emissions_kg_met_ic= 0.000272,
                                  co2_emissions_kg_met_ev=0.00011)
this_model.set_new_modes(new_mode_specs, nests_spec=nests_spec)

handler=CS_Handler(this_model, new_logit_params=new_logit_params, host_mode=host_mode)

```
## Indicator Calculations
The MobilityModel class has several methods for calculating aggregated statitical information about the mobility patterns (urban indicators). eg. mode split, CO2 emissions. These methods are called by the get_outputs() method of the Handler:

```
def get_outputs(self):
    avg_co2=self.model.get_avg_co2()
    avg_co2_norm=self.normalise_ind(avg_co2, min_value=12, max_value=5)
    live_work_prop=self.model.get_live_work_prop()
    mode_split=self.model.get_mode_split()
    delta_f_physical_activity_pp=self.model.health_impacts_pp()
    delta_f_norm=self.normalise_ind(delta_f_physical_activity_pp, min_value=0, max_value=0.004)
    output= {'CO2 Performance raw kg/day': avg_co2, 
             'Mobility Health Impacts raw mortality/year':delta_f_physical_activity_pp,
             'CO2 Performance norm': avg_co2_norm,
             'Mobility Health Impacts norm': delta_f_norm
             }
    for mode in mode_split:
        output[mode]=100*mode_split[mode]
    return output
```

These indicators can be adapted/extended by adding new methods to the MobilityModel class and calling them in the get_outputs() method of the Handler.

## Simulations

The handler can also be used to simulate random input data or run simulations with many random inputs and record the outputs.
These outputs may be used to train ML models which approximate the simulation for applications where computation time is a constraint. Example use:

```
from mobility_service_model import MobilityModel
from activity_scheduler import ActivityScheduler
from mode_logit_nhts import NhtsModeLogit
from two_stage_logit_hlc import TwoStageLogitHLC
from cs_handler import CS_Handler
  
# =============================================================================
# Create the model and add it to a handler
# =============================================================================
this_model=MobilityModel('corktown', 'Detroit')

this_model.assign_activity_scheduler(ActivityScheduler(model=this_model))

this_model.assign_mode_choice_model(NhtsModeLogit(table_name='corktown', 
	city_folder='Detroit'))

this_model.assign_home_location_choice_model(
        TwoStageLogitHLC(table_name='corktown', city_folder='Detroit', 
                         geogrid=this_model.geogrid, 
                         base_vacant_houses=this_model.pop.base_vacant))

handler=CS_Handler(this_model)

# =============================================================================
# perform an update with random input data
# =============================================================================
geogrid_data=handler.random_geogrid_data()
handler.model.update_simulation(geogrid_data)
print(handler.get_outputs())

# =============================================================================
# Perform multiple random updates, saving the inputs and outputs
# =============================================================================
X, Y = handler.generate_training_data(iterations=3)

```

## JS folder
A very simple javascript front-end prototype to demonstrate getting the simulated trip data from cityIO and displaying it with Mapbox and deck.gl

