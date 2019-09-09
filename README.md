# CS_Mobility_Service
A web service providing mobility simulations for CityScope projects.

![viz](./images/grasbrook_trips.gif)

# Data Preparation
The scripts folder contains scripts for preparing the required data for each city. The scripts in './scripts' are for any US cities. For other cities, location-specific scripts can be found in the './scripts/cities/[city]' folder
### synthpop.usa.py
This script prepares the baseline synthetic population of people living and/or working in the simulation area. It makes use of the synthpop module of the [Urban Data Science Toolkit](https://github.com/UDST)
### portal_routes.py
This script creates transport networks for each transport mode using OSM and gtfs data and then finds the shortest path for each mode between each zone in the entire modelled area and each predefined 'portal' (the main entry points to the simulation area). It also creates smaller transport networks which are contained in the simulation area.
### trip_mode_rf.py
This script calibrates a random forest model for predicting transportation mode of a trip. The [NHTS](https://nhts.ornl.gov/) data are used for calibration.
### home_loc_choice.py
This script calibrates a logit model for predicting home location choices. The [PUMS](https://www.census.gov/programs-surveys/acs/data/pums.html) data are used for calibration.

# Simulation
The simulation folder contains scripts for running simulations in response to real-time user feedback from a CityScope platform.
### abm.py
1. listens for changes to the cityIO grid and in response:
2. modifies the simulated population
3. predicts changes in home and work locations of individuals 
4. predicts each individual's choice of transport mode for their commute
5. predicts the location and time of each individual's arrival to the simulation zone (if commuting from outside)
6. posts the individual trip data to the /od end-point of the cityIO server.
The data posted to the /od end-point is a list of agent objects. Each agent object has the following structure:
{
	"home_ll": [
		10.027288528782668,
		53.533127263904795
	],
	"mode": 0,
	"type": 1
	"start_time": 29070,
	"work_ll": [
		10.0282239,
		53.533328
	]
}

'home_ll' is the home location (real or 'portal').

'work_ll' is the work location.

'type' is the type of agent where:

| 0       								| 1       					| 2       					|
|---------------------------------------|---------------------------|---------------------------|
| lives and works in simulation area 	| works in simulation area 	| lives in simulation area 	|


'mode' is the mode of transport where:

| 0       | 1       | 2       | 3       |
|---------|---------|---------|---------|
| driving | cycling | walking | transit |


# JS
A very simple javascript front-end prototype to demonstrate getting the simulated trip data from cityIO and displaying it with Mapbox and deck.gl

