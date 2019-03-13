model ABM

global {
	// FILES
	string city<-"Boston";
	file geojson_zones <- file("../includes/"+city+"/zones.geojson");
	file geojson_roads <- file("../includes/"+city+"/network_C123.geojson");
	file geojson_grid <- file("../includes/"+city+"/interaction_zone.geojson");
	file geojson_amenities <- file("../includes/"+city+"/amenities.geojson");
	file job_type_1_pop <- file("../includes/"+city+"/job_type_1.csv"); // populations to sample workers  of each type from from 
	file job_type_2_pop <- file("../includes/"+city+"/job_type_2.csv");
	file job_type_3_pop <- file("../includes/"+city+"/job_type_3.csv");
	file job_type_4_pop <- file("../includes/"+city+"/job_type_4.csv");
	matrix job_type_1_mat <- matrix(job_type_1_pop);
	matrix job_type_2_mat <- matrix(job_type_2_pop);
	matrix job_type_3_mat <- matrix(job_type_3_pop);
	matrix job_type_4_mat <- matrix(job_type_4_pop);
	geometry shape <- envelope(geojson_roads);
	float step <- 5 #sec;
	date starting_date <- date("2018-7-01T06:00:00+00:00");
//	int current_hour update: (time / #hour) mod 24;
	// PARAMETERS
	//TODO  need to update logic of trip timing based on data
	int current_hour update: 6 + (time / #hour) mod 24;
	int current_minute update: 6*60 + (time / #minute) mod (24*60);
	int min_start <- 6*60; //first activity of day (apart from Home)
	int max_start <- 9*60;
//	int min_work_end <- 16; 
//	int max_work_end <- 20; 
	int occat_1<-0; // number of new workers of each type introduced in the interacion zone (due to new commercial space).
	int occat_2<-0;
	int occat_3<-0;
	int occat_4<-0;
	int occat_5<-0;
	int res_00<-0; // capacity of new residences of each type in the  interaction zone
	int res_01<-0;
	int res_02<-0;
	int res_10<-0;
	int res_11<-0;
	int res_12<-0;
	int res_20<-0;
	int res_21<-0;
	int res_22<-0;
	list res_available<-[res_00, res_01, res_02, res_10, res_11, res_12, res_20, res_21, res_22];
	// remaining capacity for each residence type in interaction zone
	
	// INDICATORS
	list res_needed<-[0,0,0,0,0,0,0,0,0];
	// unmet demand for eah residence type in the interaction zone ( for pie chart)
	map<string,int> modal_split <- map(['car', 'bike', 'walk', 'PT'] collect (each::0));
	int all_trips<-0;
	
	map<string,rgb> color_per_mobility <- ["car"::#red, "bike"::#blue, 'walk'::rgb(124,252,0), 'PT'::#yellow];
	map<string,int> speed_per_mobility <- ["car"::20, "bike"::10, 'walk'::5, 'PT'::15];
	
	list nm_occats<-[occat_1, occat_2, occat_3, occat_4, occat_5];
	list<matrix> occat_mats<-[job_type_1_mat, job_type_2_mat, job_type_3_mat, job_type_4_mat];
//	list sampled_occat_1<-sample(range(0,1000,1),occat_1, false); // should use length of file

	graph the_graph;
	
	init {
		// create graph, zones and interaction zone
		write 'init';
		create road from: geojson_roads;
		the_graph <- as_edge_graph(road);
		create interactionZone from: geojson_grid;
		create zones from: geojson_zones with: [zoneId::string(read ("id")), popsqmile::(float(read ("POP100_RE"))/(float(read("AREA_SQFT"))*0.000000035870064))];
		create amenities from: geojson_amenities with: [food::bool(int(read("food"))), groceries::bool(int(read("groceries"))), nightlife::bool(int(read("nightlife"))), osm_id::int(read("osm_id"))]{			
			}
		
		// create the new people spawned from the new workplaces
		loop o from: 0 to:length(nm_occats)-1{ // do for each occupation category
			if (nm_occats[o]>0){
				loop i from: 0 to: nm_occats[o]{ // create N people
					create people {	
						resType<-occat_mats[o][9, i]; // get nth res type from the appropriate csv file
						age<-occat_mats[o][0, i];
						male<-occat_mats[o][2, i];
						hh_income<-occat_mats[o][1, i];
//						motif<-occat_mats[o][7, i];
						motif<-occat_mats[o][8, i];
						if (motif='HWWH') or (motif='HWOWH') {work_periods<-2;}// how many times in the day the agent goes to work
						bachelor_degree<-occat_mats[o][3, i];
						if (res_available[resType]>0){
							home_location<-any_location_in (one_of(interactionZone));
//							TODO get pop_per_sqmile properly here
							pop_per_sqmile_home<-15000;
							res_available[resType]<-res_available[resType]-1;
						}
						//TODO better choice of home zone
						else {							
							zones home_zone<-one_of(zones);
							home_location<-any_location_in (home_zone);
							pop_per_sqmile_home<-home_zone.popsqmile;
							res_needed[resType]<-res_needed[resType]+1;
						}
		          		work_location<-any_location_in (one_of(interactionZone));
		          		location<-home_location;
		          		if (motif = 'H'){min_start<-25;}
						else {
							start_first <- min_start + rnd (max_start - min_start) ;
							do plan_trips();
						}
						start_next<-start_first;
//		          		objective <- motif_list[activity_ind];	          	
						objective<-motif at activity_ind;
							
					}
				}			
			}			
		}
		
		// Create the baseline population according to census data
		create people from:csv_file( "../includes/"+city+"/synth_pop.csv",true) with:
			[home_zone_num::int(get("home_geo_index")), 
			work_zone_num::int(get("work_geo_index")),
			motif::string(get("motif"))
			]{
				home_location<-any_location_in (zones[home_zone_num]);
				pop_per_sqmile_home<-zones[home_zone_num].popsqmile;
				work_location<-any_location_in (zones[work_zone_num]);
				location<-home_location;
				start_first <- min_start + rnd (max_start - min_start) ;
				start_next<-start_first;
				objective <- motif at activity_ind;
				do plan_trips();    		
			}
	}	
}

species zones {
	string zoneId; 
	float popsqmile;
	rgb color <- rgb(20,20,20)  ;
	
	aspect base {
		draw shape color: color ;
	}
}

species road  {
	rgb color <- rgb(100,100,100) ;
	
    aspect base {
	draw shape color: color ;
	}
}

species amenities{
	bool food;
	bool groceries;
	bool nightlife;
	int osm_id;
	aspect base {
		draw square(15) color: #purple;
	}
}

species interactionZone {
	string type<-nil;
	int capacity<-0;
	int available<-0;
	rgb color <- #white  ;
	
	aspect base {
		draw shape color: color ;
	}
}

species people skills:[moving] {
	rgb color <- #black ;
	int resType<-0;
	string mode<-nil;
	int hh_income<-6;
	int home_zone_num<-0;
	int work_zone_num<-0;
	// TODO- for now, dont have attributes for the basline population so use values below
	int age<-40;
	string motif;
//	list motif_list<-list(motif);
	int work_periods<-2;
	int activity_ind<-0;
	int bachelor_degree<-1;
	int male<-1;
	float pop_per_sqmile_home<-4000.0;
	
	point home_location<-nil;
	point work_location<-nil;
	int start_next;
	int start_min;
	int start_first;
	string objective ; 
	point the_target <- nil ;
	// create lists of locations and modes at initialisation so doesnt take time during simulation
	list<point> locations<- nil;
	list modes<- nil;	
	aspect base {
		draw circle(50) color: color;
	}
	
	reflex next_activity when: length(motif)>1 and current_minute > start_next and current_minute < (start_next+60){
		// using an hour window to make sure that when people finish their activities, they dont start again.
		activity_ind <- activity_ind+1;
		objective <- motif at activity_ind;	
		the_target	<- locations at activity_ind;
		mode <- modes at (activity_ind-1);
		if (objective= "W")
			{start_next <- start_next+ (10*60)/work_periods;}
		else if (objective= "H")
			{start_next <- start_first;
			activity_ind<-0;}
		else if (objective= "O")
			{start_next <- start_next+ 1*60;}
 		do set_speed_color;
	}
		
	 
	reflex move when: the_target != nil {
		do goto target: the_target on: the_graph ; 
		if the_target = location {
			the_target <- nil ;
		}
	}
	
	action plan_trips{
		int num_locs<-length(motif);
		locations<-list_with(num_locs, home_location);
		loop i from: 1 to:num_locs-2{
			if (motif at i='W'){
				locations[i]<-work_location;
			}
			else if (motif at i='O'){				
				// pick random location with 1000m in each axis and pick the closet amenity
				locations[i]<- (amenities with_min_of(each distance_to({locations[i-1].x+rnd(-500,500), locations[i-1].y+rnd(-1000,1000)}))).location;
			}
		}
		if (num_locs>1){
			modes<-list_with(num_locs-1, nil);
			loop i from: 0 to:(num_locs-2){
				float distance;
				using topology(the_graph){
				     distance <- distance_to(locations[i], locations[i+1]);
				}
				modes[i]<- choose_mode(distance/speed_per_mobility['walk'], distance/speed_per_mobility['car'], distance/speed_per_mobility['PT'], distance/speed_per_mobility['bike'], 5.0, 5.0);
			}
		}		
	}

action choose_mode(float walk_time, float drive_time, float PT_time, float cycle_time, float walk_time_PT,float drive_time_PT){ 
     if (pop_per_sqmile_home <= 23500.00) { 
         if (walk_time <= 2.93) { 
             if (walk_time <= 0.62) { 
                 if (cycle_time <= 0.64) { 
                    mode<-['car', 'bike', 'walk', 'PT'][rnd_choice([0.16, 0.02, 0.81, 0.01])];} 
                else {// if cycle_time > 0.64 
                    mode<-['car', 'bike', 'walk', 'PT'][rnd_choice([0.57, 0.0, 0.4, 0.03])];} 
                }
            else {// if walk_time > 0.62 
                 if (pop_per_sqmile_home <= 2250.00) { 
                    mode<-['car', 'bike', 'walk', 'PT'][rnd_choice([0.67, 0.03, 0.29, 0.01])];} 
                else {// if pop_per_sqmile_home > 2250.00 
                    mode<-['car', 'bike', 'walk', 'PT'][rnd_choice([0.53, 0.02, 0.43, 0.02])];} 
                }
            }
        else {// if walk_time > 2.93 
             if (drive_time <= 0.98) { 
                 if (pop_per_sqmile_home <= 2250.00) { 
                    mode<-['car', 'bike', 'walk', 'PT'][rnd_choice([0.86, 0.01, 0.13, 0.0])];} 
                else {// if pop_per_sqmile_home > 2250.00 
                    mode<-['car', 'bike', 'walk', 'PT'][rnd_choice([0.75, 0.04, 0.19, 0.02])];} 
                }
            else {// if drive_time > 0.98 
                 if (walk_time <= 2028.14) { 
                    mode<-['car', 'bike', 'walk', 'PT'][rnd_choice([0.95, 0.0, 0.02, 0.03])];} 
                else {// if walk_time > 2028.14 
                    mode<-['car', 'bike', 'walk', 'PT'][rnd_choice([0.26, 0.0, 0.06, 0.68])];} 
                }
            }
        }
    else {// if pop_per_sqmile_home > 23500.00 
         if (hh_income <= 10.50) { 
             if (PT_time <= 0.63) { 
                 if (drive_time <= 0.69) { 
                    mode<-['car', 'bike', 'walk', 'PT'][rnd_choice([0.0, 0.0, 0.95, 0.05])];} 
                else {// if drive_time > 0.69 
                    mode<-['car', 'bike', 'walk', 'PT'][rnd_choice([0.6, 0.0, 0.0, 0.4])];} 
                }
            else {// if PT_time > 0.63 
                 if (age <= 43.00) { 
                    mode<-['car', 'bike', 'walk', 'PT'][rnd_choice([0.38, 0.02, 0.1, 0.5])];} 
                else {// if age > 43.00 
                    mode<-['car', 'bike', 'walk', 'PT'][rnd_choice([0.62, 0.02, 0.06, 0.29])];} 
                }
            }
        else {// if hh_income > 10.50 
             if (drive_time <= 0.23) { 
                mode<-['car', 'bike', 'walk', 'PT'][rnd_choice([0.0, 0.0, 1.0, 0.0])];} 
            else {// if drive_time > 0.23 
                 if (drive_time_PT <= 1.32) { 
                    mode<-['car', 'bike', 'walk', 'PT'][rnd_choice([0.36, 0.14, 0.21, 0.29])];} 
                else {// if drive_time_PT > 1.32 
                    mode<-['car', 'bike', 'walk', 'PT'][rnd_choice([0.62, 0.0, 0.33, 0.05])];} 
                }
            }
        }
    }




    
    action set_speed_color{
    		modal_split[mode] <- modal_split[mode]+1;
    		all_trips<-all_trips+1;
        if mode='car'{
			speed<-20.0 #km/#h;
			color<-#red;
		}
		else if mode='bike'{
			speed<-10.0 #km/#h;
			color<-rgb(100,149,237);
		}
		else if mode='PT'{
			speed<-15.0 #km/#h;
			color<-#yellow;
		}
		else{
			speed<-5.0 #km/#h;
			color<-#green;
		}
    }
		
}


experiment mobilityAI type: gui {
	parameter "Sales jobs" var: occat_1 category: "New Jobs" min: 0 max: 50;
	parameter "Clerical jobs" var: occat_2 category: "New Jobs" min: 0 max: 50;
	parameter "Manufacturing jobs" var: occat_3 category: "New Jobs" min: 0 max: 50;
	parameter "Professional jobs" var: occat_4 category: "New Jobs" min: 0 max: 50;
	parameter "Student Enrollments" var: occat_5 category: "New Jobs" min: 0 max: 50;
	parameter "Res 1 Bed Low Rent" var: res_00 category: "New Housing" min: 0 max: 50;
	parameter "Res 1 Bed Medium Rent" var: res_01 category: "New Housing" min: 0 max: 50;
	parameter "Res 1 Bed High Rent" var: res_02 category: "New Housing" min: 0 max: 50;
	parameter "Res 2 Bed Low Rent" var: res_10 category: "New Housing" min: 0 max: 50;
	parameter "Res 2 Bed Medium Rent" var: res_11 category: "New Housing" min: 0 max: 50;
	parameter "Res 2 Bed High Rent" var: res_12 category: "New Housing" min: 0 max: 50;
	parameter "Res 3 Bed Low Rent" var: res_20 category: "New Housing" min: 0 max: 50;
	parameter "Res 3 Bed Medium Rent" var: res_21 category: "New Housing" min: 0 max: 50;
	parameter "Res 3 Bed High Rent" var: res_22 category: "New Housing" min: 0 max: 50;
	output {
		display housing autosave:false refresh:every(1000){
			chart "Housing Demand" background:#white type:pie {
				data "Res 1 Bed Low Rent" value:res_needed[0] color:rgb(166,206,227);
				data "Res 1 Bed Medium Rent" value:res_needed[1] color:rgb(178,223,138);
				data "Res 1 Bed High Rent" value:res_needed[2] color:rgb(51,160,44);
				data "Res 2 Bed Low Rent" value:res_needed[3] color:rgb(251,154,153);
				data "Res 2 Bed Medium Rent" value:res_needed[4] color:rgb(227,26,28);
				data "Res 2 Bed High Rent" value:res_needed[5] color:rgb(253,191,111);
				data "Res 3 Bed Low Rent" value:res_needed[6] color:rgb(31,120,180);
				data "Res 3 Bed Medium Rent" value:res_needed[7] color:rgb(255,127,0);
				data "Res 3 Bed High Rent" value:res_needed[8] color:rgb(202,178,214);
			}			
		}
//		display modes autosave:false refresh:every(1000){
//			chart "Modal Split" background:#white type: pie  
//				{
//					loop i from: 0 to: length(modal_split.keys)-1	{
//					  data modal_split.keys[i] value: modal_split.values[i] color:color_per_mobility[modal_split.keys[i]];
//					}
//				}			
//		}
		display city_display background:#black autosave:false type:opengl {
			species zones aspect: base ;
			species road aspect: base ;
			species amenities aspect: base ;
			species interactionZone aspect: base ;
			species people transparency:0.2 aspect: base ;
			overlay position: { 3,3 } size: { 150 #px, 170 #px } background: # gray transparency: 0.8 border: # black 
            {	
            		draw string(current_date.hour) + "h" + string(current_date.minute) +"m" at: { 20#px, 30#px } color: # white font: font("Helvetica", 25, #italic) perspective:false;
//  				draw "Mobility Modes" at: { 20#px, 60#px } color: #black font: font("Helvetica", 15, #bold) perspective:false;
  				draw "Car "+int(1000*modal_split["car"]/all_trips)/10 +"%" at: { 20#px, 60#px } color: #red font: font("Helvetica", 20, #bold ) perspective:false;
  				draw "Bike "+int(1000*modal_split["bike"]/all_trips)/10 +"%" at: { 20#px, 90#px } color: rgb(100,149,237) font: font("Helvetica", 20, #bold ) perspective:false;
  				draw "PT "+int(1000*modal_split["PT"]/all_trips)/10 +"%" at: { 20#px, 120#px } color: #yellow font: font("Helvetica", 20, #bold ) perspective:false;
  				draw "Walk " +int(1000*modal_split["walk"]/all_trips)/10 +"%" at: { 20#px, 150#px } color: rgb(124,252,0) font: font("Helvetica", 20, #bold ) perspective:false;
            }
		
				
			}
		
		
	}
}