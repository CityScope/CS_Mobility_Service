model choiceModel

import "MoBalance.gaml"

global{

action choose_mode_per_people(people p,float walk_time, float drive_time, float PT_time, float cycle_time, float walk_time_PT,float drive_time_PT){ 
    list probs<-[0.0,0.0,0.0,0.0];
// Tree #0
         if (p.pop_per_sqmile_home <= 23500.00) { 
             if (walk_time <= 2.04) { 
                 if (walk_time <= 0.86) { 
                     if (cycle_time <= 0.83) { 
                        list pred<-[0.21, 0.01, 0.77, 0.01]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                    else {// if cycle_time > 0.83 
                        list pred<-[0.67, 0.0, 0.29, 0.03]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                }
                else {// if walk_time > 0.86 
                     if (p.pop_per_sqmile_home <= 5000.00) { 
                        list pred<-[0.67, 0.0, 0.32, 0.01]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                    else {// if p.pop_per_sqmile_home > 5000.00 
                        list pred<-[0.39, 0.0, 0.6, 0.0]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                }
            }
            else {// if walk_time > 2.04 
                 if (drive_time <= 1.02) { 
                     if (cycle_time <= 1.99) { 
                        list pred<-[0.67, 0.04, 0.28, 0.01]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                    else {// if cycle_time > 1.99 
                        list pred<-[0.81, 0.02, 0.16, 0.01]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                }
                else {// if drive_time > 1.02 
                     if (walk_time <= 3029.82) { 
                        list pred<-[0.94, 0.01, 0.02, 0.03]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                    else {// if walk_time > 3029.82 
                        list pred<-[0.07, 0.0, 0.0, 0.93]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                }
            }
        }
        else {// if p.pop_per_sqmile_home > 23500.00 
             if (p.bachelor_degree <= 0.50) { 
                 if (PT_time <= 2.53) { 
                     if (walk_time <= 2.71) { 
                        list pred<-[0.09, 0.0, 0.91, 0.0]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                    else {// if walk_time > 2.71 
                        list pred<-[0.76, 0.0, 0.18, 0.06]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                }
                else {// if PT_time > 2.53 
                     if (p.age <= 39.00) { 
                        list pred<-[0.6, 0.0, 0.0, 0.4]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                    else {// if p.age > 39.00 
                        list pred<-[0.15, 0.0, 0.08, 0.77]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                }
            }
            else {// if p.bachelor_degree > 0.50 
                 if (drive_time <= 0.64) { 
                     if (p.male <= 0.50) { 
                        list pred<-[0.08, 0.0, 0.77, 0.15]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                    else {// if p.male > 0.50 
                        list pred<-[0.0, 0.09, 0.91, 0.0]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                }
                else {// if drive_time > 0.64 
                     if (drive_time_PT <= 1.77) { 
                        list pred<-[0.43, 0.02, 0.1, 0.45]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                    else {// if drive_time_PT > 1.77 
                        list pred<-[0.75, 0.05, 0.08, 0.12]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                }
            }
        }
// Tree #1
         if (p.bachelor_degree <= 0.50) { 
             if (walk_time <= 2.06) { 
                 if (cycle_time <= 1.81) { 
                     if (PT_time <= 0.05) { 
                        list pred<-[0.2, 0.01, 0.79, 0.01]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                    else {// if PT_time > 0.05 
                        list pred<-[0.46, 0.02, 0.53, 0.0]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                }
                else {// if cycle_time > 1.81 
                     if (drive_time <= 1.00) { 
                        list pred<-[0.96, 0.0, 0.0, 0.04]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                    else {// if drive_time > 1.00 
                        list pred<-[0.86, 0.0, 0.12, 0.02]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                }
            }
            else {// if walk_time > 2.06 
                 if (p.pop_per_sqmile_home <= 23500.00) { 
                     if (PT_time <= 0.45) { 
                        list pred<-[0.76, 0.02, 0.19, 0.02]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                    else {// if PT_time > 0.45 
                        list pred<-[0.94, 0.01, 0.02, 0.04]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                }
                else {// if p.pop_per_sqmile_home > 23500.00 
                     if (cycle_time <= 3.35) { 
                        list pred<-[0.17, 0.0, 0.83, 0.0]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                    else {// if cycle_time > 3.35 
                        list pred<-[0.43, 0.0, 0.0, 0.57]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                }
            }
        }
        else {// if p.bachelor_degree > 0.50 
             if (cycle_time <= 1.95) { 
                 if (drive_time <= 0.13) { 
                     if (cycle_time <= 0.04) { 
                        list pred<-[0.58, 0.0, 0.42, 0.0]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                    else {// if cycle_time > 0.04 
                        list pred<-[0.1, 0.02, 0.88, 0.0]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                }
                else {// if drive_time > 0.13 
                     if (p.pop_per_sqmile_home <= 23500.00) { 
                        list pred<-[0.5, 0.02, 0.48, 0.0]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                    else {// if p.pop_per_sqmile_home > 23500.00 
                        list pred<-[0.02, 0.0, 0.96, 0.02]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                }
            }
            else {// if cycle_time > 1.95 
                 if (walk_time_PT <= 2.83) { 
                     if (drive_time_PT <= 0.87) { 
                        list pred<-[0.86, 0.01, 0.05, 0.08]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                    else {// if drive_time_PT > 0.87 
                        list pred<-[0.94, 0.01, 0.04, 0.01]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                }
                else {// if walk_time_PT > 2.83 
                     if (p.pop_per_sqmile_home <= 23500.00) { 
                        list pred<-[0.86, 0.0, 0.09, 0.05]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                    else {// if p.pop_per_sqmile_home > 23500.00 
                        list pred<-[0.48, 0.0, 0.22, 0.3]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                }
            }
        }
// Tree #2
         if (cycle_time <= 1.75) { 
             if (walk_time_PT <= 2.47) { 
                 if (p.bachelor_degree <= 0.50) { 
                     if (walk_time <= 1.14) { 
                        list pred<-[0.13, 0.02, 0.85, 0.0]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                    else {// if walk_time > 1.14 
                        list pred<-[0.52, 0.02, 0.44, 0.03]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                }
                else {// if p.bachelor_degree > 0.50 
                     if (p.age <= 61.50) { 
                        list pred<-[0.25, 0.02, 0.73, 0.0]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                    else {// if p.age > 61.50 
                        list pred<-[0.5, 0.02, 0.48, 0.0]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                }
            }
            else {// if walk_time_PT > 2.47 
                 if (walk_time <= 1.66) { 
                     if (walk_time_PT <= 2.83) { 
                        list pred<-[0.42, 0.01, 0.56, 0.0]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                    else {// if walk_time_PT > 2.83 
                        list pred<-[0.05, 0.0, 0.95, 0.0]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                }
                else {// if walk_time > 1.66 
                     if (PT_time <= 0.22) { 
                        list pred<-[0.73, 0.02, 0.25, 0.0]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                    else {// if PT_time > 0.22 
                        list pred<-[0.56, 0.01, 0.43, 0.0]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                }
            }
        }
        else {// if cycle_time > 1.75 
             if (cycle_time <= 4.54) { 
                 if (p.bachelor_degree <= 0.50) { 
                     if (PT_time <= 0.32) { 
                        list pred<-[0.72, 0.04, 0.24, 0.0]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                    else {// if PT_time > 0.32 
                        list pred<-[0.86, 0.02, 0.1, 0.03]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                }
                else {// if p.bachelor_degree > 0.50 
                     if (PT_time <= 0.30) { 
                        list pred<-[0.62, 0.02, 0.35, 0.01]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                    else {// if PT_time > 0.30 
                        list pred<-[0.8, 0.02, 0.16, 0.02]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                }
            }
            else {// if cycle_time > 4.54 
                 if (walk_time <= 2380.11) { 
                     if (p.age <= 18.50) { 
                        list pred<-[0.86, 0.01, 0.01, 0.13]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                    else {// if p.age > 18.50 
                        list pred<-[0.96, 0.0, 0.01, 0.02]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                }
                else {// if walk_time > 2380.11 
                     if (cycle_time <= 1996.17) { 
                        list pred<-[0.5, 0.0, 0.0, 0.5]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                    else {// if cycle_time > 1996.17 
                        list pred<-[0.06, 0.0, 0.0, 0.94]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                }
            }
        }
// Tree #3
         if (p.age <= 18.50) { 
             if (cycle_time <= 1.53) { 
                 if (drive_time_PT <= 2.13) { 
                     if (PT_time <= 0.52) { 
                        list pred<-[0.12, 0.0, 0.88, 0.0]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                    else {// if PT_time > 0.52 
                        list pred<-[0.75, 0.0, 0.25, 0.0]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                }
                else {// if drive_time_PT > 2.13 
                     if (PT_time <= 0.00) { 
                        list pred<-[0.25, 0.75, 0.0, 0.0]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                    else {// if PT_time > 0.00 
                        list pred<-[0.38, 0.0, 0.6, 0.02]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                }
            }
            else {// if cycle_time > 1.53 
                 if (drive_time <= 10.33) { 
                     if (p.hh_income <= 1.50) { 
                        list pred<-[0.52, 0.0, 0.0, 0.48]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                    else {// if p.hh_income > 1.50 
                        list pred<-[0.79, 0.02, 0.04, 0.14]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                }
                else {// if drive_time > 10.33 
                     if (p.male <= 0.50) { 
                        list pred<-[0.99, 0.0, 0.0, 0.01]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                    else {// if p.male > 0.50 
                        list pred<-[0.95, 0.0, 0.0, 0.05]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                }
            }
        }
        else {// if p.age > 18.50 
             if (p.bachelor_degree <= 0.50) { 
                 if (drive_time <= 0.51) { 
                     if (walk_time <= 0.87) { 
                        list pred<-[0.28, 0.0, 0.72, 0.0]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                    else {// if walk_time > 0.87 
                        list pred<-[0.6, 0.01, 0.38, 0.0]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                }
                else {// if drive_time > 0.51 
                     if (walk_time_PT <= 2.83) { 
                        list pred<-[0.96, 0.0, 0.03, 0.01]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                    else {// if walk_time_PT > 2.83 
                        list pred<-[0.75, 0.0, 0.22, 0.03]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                }
            }
            else {// if p.bachelor_degree > 0.50 
                 if (p.pop_per_sqmile_home <= 23500.00) { 
                     if (cycle_time <= 2.14) { 
                        list pred<-[0.49, 0.02, 0.49, 0.0]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                    else {// if cycle_time > 2.14 
                        list pred<-[0.94, 0.01, 0.04, 0.02]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                }
                else {// if p.pop_per_sqmile_home > 23500.00 
                     if (p.age <= 22.50) { 
                        list pred<-[0.0, 0.0, 0.0, 1.0]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                    else {// if p.age > 22.50 
                        list pred<-[0.37, 0.03, 0.36, 0.24]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                }
            }
        }
// Tree #4
         if (p.bachelor_degree <= 0.50) { 
             if (walk_time <= 2.38) { 
                 if (drive_time_PT <= 4.61) { 
                     if (p.pop_per_sqmile_home <= 175.00) { 
                        list pred<-[0.67, 0.0, 0.33, 0.0]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                    else {// if p.pop_per_sqmile_home > 175.00 
                        list pred<-[0.46, 0.01, 0.53, 0.0]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                }
                else {// if drive_time_PT > 4.61 
                     if (p.male <= 0.50) { 
                        list pred<-[0.0, 0.0, 0.88, 0.12]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                    else {// if p.male > 0.50 
                        list pred<-[0.0, 0.0, 1.0, 0.0]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                }
            }
            else {// if walk_time > 2.38 
                 if (PT_time <= 0.45) { 
                     if (p.age <= 32.50) { 
                        list pred<-[0.56, 0.07, 0.31, 0.05]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                    else {// if p.age > 32.50 
                        list pred<-[0.85, 0.01, 0.14, 0.01]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                }
                else {// if PT_time > 0.45 
                     if (PT_time <= 289.20) { 
                        list pred<-[0.94, 0.01, 0.02, 0.04]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                    else {// if PT_time > 289.20 
                        list pred<-[0.0, 0.0, 0.0, 1.0]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                }
            }
        }
        else {// if p.bachelor_degree > 0.50 
             if (drive_time <= 0.61) { 
                 if (p.pop_per_sqmile_home <= 2250.00) { 
                     if (cycle_time <= 0.41) { 
                        list pred<-[0.14, 0.02, 0.85, 0.0]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                    else {// if cycle_time > 0.41 
                        list pred<-[0.64, 0.02, 0.33, 0.0]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                }
                else {// if p.pop_per_sqmile_home > 2250.00 
                     if (drive_time_PT <= 2.33) { 
                        list pred<-[0.2, 0.02, 0.78, 0.01]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                    else {// if drive_time_PT > 2.33 
                        list pred<-[0.43, 0.01, 0.56, 0.0]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                }
            }
            else {// if drive_time > 0.61 
                 if (PT_time <= 296.77) { 
                     if (drive_time <= 1.59) { 
                        list pred<-[0.83, 0.02, 0.14, 0.02]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                    else {// if drive_time > 1.59 
                        list pred<-[0.96, 0.01, 0.02, 0.02]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                }
                else {// if PT_time > 296.77 
                     if (drive_time_PT <= 1.62) { 
                        list pred<-[0.4, 0.0, 0.0, 0.6]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                    else {// if drive_time_PT > 1.62 
                        list pred<-[0.0, 0.0, 0.0, 1.0]; 
                        loop o from: 0 to:3{probs[o]<-probs[o]+pred[o]; }                    }
                }
            }
        }
    p.mode<-['car', 'bike', 'walk', 'PT'][rnd_choice(probs)];
    }
}