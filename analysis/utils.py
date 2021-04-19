import pandas as pd
import numpy as np

def compute_reward(zones_temp, T_set, officeHour, comfort_tol = 2,\
    occupancy=[1, 1, 1, 1, 1, 1, 1, 1, 1], tau=[100, 1]):
    '''
    Similar function to the _compute_reward method in medOff_env.py
      
    Input
    zones_temp: numpy array, temperature for each thermal zone
    T_set: numpy array, temperature setpoint for each thermal zone
    officeHour: binary, whether it is office hour or not
    comfort_tol: float, comfort tolerance, used to calculate the uncomfortable degree hour
    occupancy: list of binary, occupancy of each zone
    tau: weights of comfort over energy, higher weights of comfort during office hour

    Output
    cost_comfort: cost of comfort for this time step
    zone_temp_min: minimum zone temperature of the 9 zones at the time step
    zone_temp_max: maximum zone temperature of the 9 zones at the time step
    uncDegHour: uncomfortable degree hours
    '''
    
    # comfort cost is 0 if space is non-occupied
    cost_comfort_t = sum(np.multiply(
        (zones_temp-T_set), np.array(occupancy))**2)
    zone_temp_min = min(zones_temp)
    zone_temp_max = max(zones_temp)

    # the comfort/energy weight is determined on time
    if officeHour:
        tau_cost = tau[0]
        tau_udh = 1
    else:
        tau_cost = tau[1]
        zone_temp_min = np.mean(T_set)
        zone_temp_max = np.mean(T_set)
        tau_udh = 0

    cost_comfort = tau_cost * cost_comfort_t

    heaDegHour = sum(np.maximum(zones_temp-(T_set+2),np.zeros(9)))
    cooDegHour = sum(np.maximum((T_set-2)-zones_temp,np.zeros(9)))
    uncDegHour = (heaDegHour+cooDegHour) * tau_udh
    
    return cost_comfort, zone_temp_min, zone_temp_max, uncDegHour

def mapping():
    efficiency_map = {'Standard':2, 'Low':1, 'High':3}
    city_map = {'SF':2, 'Miami':3, 'Chicago':1}
    year_map = {'TMY3':0}
    return efficiency_map, city_map, year_map