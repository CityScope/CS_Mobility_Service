action choose_mode(float walk_time, float drive_time, float PT_time, float cycle_time, float drive_time_PT){ 
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
