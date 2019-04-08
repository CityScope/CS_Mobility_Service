# CS_Mobility_Service
A web service providing mobility simulations for CityScope projects.

## python
Contains scripts for 
- preparing the required data for each city
- calibrating models (random forest and logit) to predict mobility choices
- a multi-modal transport simulation which posts geojson data representing individual agents to the cityIO API

To run run the simulation and view the results, perform the following:
- clone the repo
- run the abm_backend.py script
- nagivate to the 'js' directory and run a local web server
- open your browser and go to local host to view the simulation front-end.

## JS
A very simple javascript front-end prototype to demonstrate getting the simulated agent data from cityIO and displaying it with Mapbox

## ABM
A GAMA prototype simulation which can run and dispay the simulation locally but can also send point data to cityIO


