# CS_Mobility_Service
A web service providing mobility simulations for CityScope projects.

![viz](./images/grasbrook_trips.gif)

## MobilityModel Class
The MobilityModel class creates a model of the target and simulates the mobility behaviour of a sample of the population. The following inputs ust be provided on initialisation:
- table_name: the name of the table end-point on city_IO
- city_folder: the folder where the input data for this city are located.

Before the simulation can be run, the necessary prediction models must be assigned to the MobilityModel using the following methods:
- assign_activity_scheduler()
- assign_mode_choice_model
- assign_home_location_choice_model

The MobilityModel class makes the simulation outputs available through these methods 
- get_mode_split() returns the proportion of trips made by each mode
- get_live_work_prop() takes a list of Person objects and returns the proportion of persons both living and working in the district
- get_avg_co2() takes a list of Person objects and returns the average predicted CO2 emissions.
- get_trips_layer() creates a list of simulated trips in the correct format for the DeckGL Trips visualisation layer. Each trip object has an attribute 'mode' = ['trip_mode', 'type'] where:

'type' is the type of agent where:

| 0       								| 1       					| 2       					|
|---------------------------------------|---------------------------|---------------------------|
| lives and works in simulation area 	| works in simulation area 	| lives in simulation area 	|


'trip_mode' is the mode of transport where:

| 0       | 1       | 2       | 3       |
|---------|---------|---------|---------|
| driving | cycling | walking | transit |


##  

## Handler Class
The handler takes care of getting data from and posting data to cityIO for live deployments. When listen_city_IO() is run, the handler will begin regularly checking the geogrid_data hash id for the specified table. Each time a change in the geogrid_data is detected, a simulation update is triggered and the results are posted back to city_IO. Example use:
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

