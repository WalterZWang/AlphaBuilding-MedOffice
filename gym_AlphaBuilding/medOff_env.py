# -*- coding: utf-8 -*-
"""

"""

import os
from datetime import timedelta
import pandas as pd
import numpy as np
import math
import pytz
from pyfmi import load_fmu

from gym import Env
from gym import spaces
from gym.utils import seeding


class MedOffEnv(Env):
    """ AlphaResEnv is a custom Gym Environment 
    Args:
    - building_path: string, the path to the fmu simulate the building dynamics
    - sim_days: int, number of days for simulation, unit: day, example - 365
    - step_size: int, step size of simulation, unit: second, example - 900
    - sim_year: int, the year of simulation, this is to create the time index, example - 2015, 
        need to be careful about the leap year
    - tz_name: string, the name of the time zone, this is to create the time index, example - 'America/Los_Angeles',
    - occupied_hour: tuple, occupied hours, the weights of comfort will be higher to calculate the rewards for the occupied hours,
        example - (6, 20),
    - weight_reward: tuple, the weight of comfort over energy to calculate the reward, example - (0.2, 0.02)
    - eprice_path: string, the path to the electricity price, currently not used in this version,

    Action space:
    - 19 actions in total
        - act[0]: AHU supply air temperature
        - act[1:19]: VAV flow rate and reheat q, for the 2 conference rooms, 3 enclosed offices, and 4 open offices respectively

    Observation space:
    - 12 observations in total
        - obs[0:2]: time - hour of day (0-23), and day of week (0-6, 5 and 6 are weekends)
        - obs[2:5]: ambient weather - temperature, solar radiation, relative humidity
        - obs[5:12]: indoor temperature - for the 2 conference rooms, 3 enclosed offices, and 4 open offices respectively

    Rewards:
    Weighted sum of
    - Energy
        - AHU Energy
            - Fan Energy: from E+ "21 Zone PVAV Fan,Fan Electric Energy", Unit [J](TimeStep)
            - Cooling Energy: from E+ "21 ZONE PVAV,Air System DX Cooling Coil Electric Energy", Unit [J](TimeStep)
            - Heating Energy: from E+ "21 ZONE PVAV MAIN GAS HTG COIL,Heating Coil Gas Energy", Unit [J](TimeStep)
                Weight is 0.4 of heatEnergy, because conversion from Gas to Electricity is 0.4, based on gas generator
        - Terminal reheat
            - Implemented as E+ "*_ZN_Reheat" (example "EnclosedOffice_Mid_1")
                Weight is 1, as terminal reheat are usually eletrical heater
    - Comfort
        - Mean Square Error from the setpoint
            Weight of each thermal zone is based on the number of occupants in that zone
            Designed occupant counts in each zone can be found in docs/zone_summary.xlsx
    - The weight for energy and comfort is an input from the users  
    """
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 building_path=None,
                 sim_days=None,
                 step_size=None,  # unit: [s]
                 sim_year=2015,
                 tz_name='America/Los_Angeles',
                 occupied_hour=(6, 20),
                 weight_reward=(0.5, 0.05),
                 eprice_path=None):

        # load fmu models
        self.building_path = building_path
        self.building_model = load_fmu(
            self.building_path, kind='cs', log_level=0)
        self.occupied_hour = occupied_hour
        self.weight_reward = weight_reward

        # define simulations time steps
        self.tz_local = pytz.timezone(tz_name)
        self.tz_utc = pytz.timezone("UTC")
        self.start_time = 0
        self.final_time = 3600*24*sim_days
        self.step_size = step_size  # 900 == 15 min
        self.time_steps = np.arange(self.start_time,
                                    self.final_time,
                                    self.step_size)
        self.n_steps = len(self.time_steps)
        self.sim_year = sim_year
        self.time_index = pd.date_range(start='1/1/{0}'.format(self.sim_year),
                                        periods=len(self.time_steps)+1,
                                        freq='{}T'.format(self.step_size//60))

        # flag: initialize or reset, -1 means initialize
        self.episode_idx = -1

        # Initialize actions
        '''
    	ahuSAT: AHU supply air (to room) temperature, [degC]
    	*FR: terminal flow rate, [kg/s]
    	*Reheat: terminal reheat, [W]
        '''
        self.action_names = ['ahuSAT',
                             'conf1FR', 'conf1Reheat', 'conf2FR', 'conf2Reheat',
                             'enOff1FR', 'enOff1Reheat', 'enOff2FR', 'enOff2Reheat', 'enOff3FR', 'enOff3Reheat',
                             'opOff1FR', 'opOff1Reheat', 'opOff2FR', 'opOff2Reheat', 'opOff3FR', 'opOff3Reheat', 'opOff4FR', 'opOff4Reheat']
        self.action_init_raw = np.array([20,
                                         0.1, 0, 0.1, 0,
                                         0.1, 0, 0.1, 0, 0.1, 0,
                                         0.1, 0, 0.1, 0, 0.1, 0, 0.1, 0])

        # Define Action Space:
        self.action_space = spaces.Box(low=np.array([10, 0,     0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0]),
                                       high=np.array(
                                           [30, 1, 10000, 1, 10000, 1, 10000, 1, 10000, 1, 10000, 1, 10000, 1, 10000, 1, 10000, 1, 10000]),
                                       dtype=np.float32)

        #  Define Observation Space
        '''
		states_amb: ambient temperature
		states_temp: temperature of controlled rooms
		states_energy: fan, cooling and heating energy consumption [J]
        '''
        self.states_time = ['hour', 'dayOfWeek']
        self.states_amb = ['outTemp', 'outSolar', 'outRH']
        self.states_temp = ['conf1Temp', 'conf2Temp', 'enOff1Temp', 'enOff2Temp', 'enOff3Temp',
                            'opOff1Temp', 'opOff2Temp', 'opOff3Temp', 'opOff4Temp']
        # designed for interacting with the FMU
        self.states_energy = ['fanEnergy', 'coolEnergy', 'heatEnergy']

        # self.fmu_obs: interface with fmu, get obs from fmu, NOT the interface of env
        self.fmu_obs = self.states_amb + self.states_temp + self.states_energy

        # self.observation_space: interface with controller agent (act & crt network)
        # include: states_amb, states_temp, time
        # energy is not a state but a reward
        self.env_obs = self.states_time + self.states_amb + self.states_temp
        self.observation_space = spaces.Box(low=np.array(
            [0.0, 0.0, -30.0,    0.0,   0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0]),
            high=np.array(
            [23.0, 6.0,  50.0, 1500.0, 100.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0]),
            dtype=np.float32)

    def reset(self):
        self.time_step_idx = 0
        self.reward = 0.0

        self.building_model.reset()

        self.building_model.initialize(self.start_time, self.final_time)

        self.action_init_scaled = self.scale_action(self.action_init_raw)

        return self.step(self.action_init_scaled)[0]

    def step(self, action_scaled):
        '''
        input - actions_scaled
        output - obs_scaled
        '''

        if action_scaled is not None:
            action_raw = self.rescale_action(action_scaled)
            print('AHU SAT Raw: {0}'.format(action_raw[0]))
            # input the actions
            self.building_model.set(self.action_names, action_raw)
            # simulate fmu
            self.building_model.do_step(current_t=self.time_steps[self.time_step_idx],
                                        step_size=self.step_size,
                                        new_step=True)
            # extract the observations
            time_current = self.time_index[self.time_step_idx]
            obs_all_raw = np.concatenate(self.building_model.get(self.fmu_obs))
            state_raw = obs_all_raw[:-3]   # exclude energy
            hourOfDay = time_current.hour
            dayOfWeek = time_current.dayofweek    # Monday=0, Sunday=6
            state_raw = np.concatenate(
                (np.array([hourOfDay, dayOfWeek]), state_raw))

            # calculate the rewards from the observation, reheat of the current time step is in action_raw
            reward, energy, comfort, temp_min, temp_max, uncDegHour = self._compute_reward(
                obs_all_raw, time_current, action_raw, 22, self.weight_reward, self.occupied_hour)

            if self.time_step_idx < (self.n_steps - 1):
                done = False
                self.time_step_idx += 1
            else:
                done = True
                self.time_step_idx = 0

        state_scaled = self.scale_state(state_raw)
        return state_scaled, reward, done, (energy, comfort, temp_min, temp_max, uncDegHour)

    def rescale_state(self, states_scaled):
        '''
        Recover the raw states from [-1,1], energy obs not included
        Not used so far
        '''
        states_raw = self.observation_space.low + \
            (states_scaled + 1.) * 0.5 * \
            (self.observation_space.high - self.observation_space.low)
        states_raw = np.clip(
            states_raw, self.observation_space.low, self.observation_space.high)

        return states_raw

    def rescale_action(self, actions_scaled):
        '''
        Recover the raw actions from [-1,1]
        '''
        actions_raw = self.action_space.low + \
            (actions_scaled + 1.) * 0.5 * \
            (self.action_space.high - self.action_space.low)
        actions_raw = np.clip(
            actions_raw, self.action_space.low, self.action_space.high)

        return actions_raw

    def scale_state(self, states_raw):
        '''
        Scale the states to [-1,1], does not include energy obs
        '''
        states_scaled = 2.0 * (states_raw - self.observation_space.low) / \
            (self.observation_space.high - self.observation_space.low) - 1

        return states_scaled

    def scale_action(self, actions_raw):
        '''
        Scale the action to [-1,1]
        '''
        actions_scaled = 2.0 * (actions_raw - self.action_space.low) / \
            (self.action_space.high - self.action_space.low) - 1

        return actions_scaled

    def _compute_reward(self, obs, time, act, T_set, weight,
                        occupied_hour, comfort_tol=2,
                        zone_weight=[23, 23, 6, 6, 6, 11, 12, 10, 10]):
        '''
        Similar function to the compute_reward method in analysis/utils.py
        Difference is: 1. one more output - energy
                       2. T_set is a number rather than an array
        Input
        obs: numpy array, obs from the environment, raw value
        time: current time, pandas timestamp, support attribute h, and method weekday()
        act: action of previous time step, raw value
        T_set: float, temperature setpoint for each thermal zone
        weight: weights of comfort over energy, higher weights of comfort during office hour
        comfort_tol: float, comfort tolerance, used to calculate the uncomfortable degree hour
        zone_weight: comfort weight for each thermal zone, cosnidering the occupant number in each zone

        Output
        cost_comfort: cost of comfort for this time step
        zone_temp_min: minimum zone temperature of the 9 zones at the time step
        zone_temp_max: maximum zone temperature of the 9 zones at the time step
        uncDegHour: uncomfortable degree hours
        '''
        # energy cost
        ahu_energy = (obs[12] + obs[13] + obs[14]*0.4) / \
            3600000    # unit:J -> kWh
        reheat_energy = (act[2] + act[4] + act[6] + act[8] + act[10] +
                         act[12] + act[14] + act[16] + act[18])/(1000*4)   # unit:kWh, W->kW, 15min per timestep, -> h
        cost_energy = ahu_energy + reheat_energy

        # comfort cost
        zones_temp = obs[3:12]
        # the comfort/energy weight is determined on time, comfort cost weight is smaller if space is non-occupied
        officeHour = (time.weekday() < 5) & (
            time.hour < occupied_hour[1]) & (time.hour >= occupied_hour[0])
        if officeHour:
            weight_comfort = weight[0]
            weight_udh = 1
        else:
            weight_comfort = weight[1]
            weight_udh = 0

        cost_comfort_t = sum(((zones_temp-T_set)**2)*np.array(zone_weight))
        zone_temp_min = min(zones_temp)
        zone_temp_max = max(zones_temp)

        heaDegHour = sum(np.maximum(
            zones_temp-(T_set+comfort_tol), np.zeros(9)))
        cooDegHour = sum(np.maximum(
            (T_set-comfort_tol)-zones_temp, np.zeros(9)))
        uncDegHour = (heaDegHour+cooDegHour) * weight_udh

        cost_comfort = weight_comfort * cost_comfort_t

        reward_t = -1.0 * (cost_energy + cost_comfort)

        print("Energy consumption (kWh): {0}".format(cost_energy))
        print("Comfort cost: {0}".format(cost_comfort))
        print("Total reward: {0}".format(reward_t))
        print("Uncomfort degree hour: {0}".format(uncDegHour))
        print("Zone temperature range (degC): {0} ~ {1}".format(
              zone_temp_min, zone_temp_max))

        return reward_t, cost_energy, cost_comfort, zone_temp_min, zone_temp_max, uncDegHour


def time_converter(year, idx, total_timestep=35040):
    """
    Input
    -------------------
    year: int, year to be converted
    idx: int, index of timestep in the specific year, SHOULD start from zero
    total_timestep: total timestep of the specific year

    Output
    -------------------
    pandas Timestamp of the time corresponding to the idx
    """
    index = pd.date_range(
        start='1/1/{0}'.format(year), end='1/1/{0}'.format(year+1), periods=total_timestep+1)
    time = index[idx]
    return time


def get_price_data_from_csv(eprice_path, start_time=None, end_time=None):
    '''Get a DataFrame of all price variables from csv files
        Parameters
        ----------
        start_time : datetime
            Start time of timeseries
        end_time : datetime
            End time of timeseries
        -------
        df: pandas DataFrame
            DataFrame where each column is a variable in the variables section in the configuration
    '''
    df = pd.read_csv(eprice_path, index_col=0, parse_dates=True)
    #df = df.tz_localize(tz_local).tz_convert(tz_utc)
    df.index = pd.to_datetime(df.index)
    idx = df.loc[start_time:end_time].index
    df = df.loc[idx, ]

    return df


def get_price_data_from_df(df, start_time=None, end_time=None):
    '''Get a DataFrame of all price variables from csv files
        Parameters
        ----------
        start_time : datetime
            Start time of timeseries
        end_time : datetime
            End time of timeseries
        -------
        df: pandas DataFrame
            DataFrame where each column is a variable in the variables section in the configuration
    '''
    df.index = pd.to_datetime(df.index)
    idx = df.loc[start_time:end_time].index
    df = df.loc[idx, ]

    return df


class ExperienceBuffer:
    def __init__(self, env_obs, action_names):
        # initialize model observation
        self.env_obs = env_obs
        self.action_names = action_names

        obs_dict = dict()
        for obs_i in env_obs:
            obs_dict[obs_i] = [0.0]
        self.obs_df = pd.DataFrame(obs_dict)

        # initialize actions
        action_dict = dict()
        for action_i in self.action_names:
            action_dict[action_i] = [0.0]
        self.actions_df = pd.DataFrame(action_dict)

        # initialize rewards
        self.rewards_df = pd.DataFrame({'reward': [0.0]})

    def append(self, action, obs, reward):
        action_dict = dict()
        for i in range(len(self.action_names)):
            action_i = self.action_names[i]
            action_dict[action_i] = [action[i]]
        action_df_0 = pd.DataFrame(action_dict)
        self.actions_df = self.actions_df.append(
            action_df_0, ignore_index=True)

        obs_dict = dict()
        # obs does not contain energy consumption data
        for i in range(len(self.env_obs)-3):
            obs_i = self.env_obs[i]
            obs_dict[obs_i] = [obs[i]]
        obs_df_0 = pd.DataFrame(obs_dict)
        self.obs_df = self.obs_df.append(obs_df_0, ignore_index=True)

        reward_df_0 = pd.DataFrame({'reward': [reward]})
        self.rewards_df = self.rewards_df.append(
            reward_df_0, ignore_index=True)

    def last_action(self):
        return self.actions_df.iloc[len(self.actions_df)-1]

    def action_data(self):
        return self.actions_df

    def obs_data(self):
        return self.obs_df

    def reward_data(self):
        return self.rewards_df
