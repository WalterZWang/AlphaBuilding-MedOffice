'''Utility functions for distributed RL
'''

import numpy as np


class vav_scale():
    def __init__(self, env):
        # States: hour, dayOfWeek, outTemp, solar, dT (inTemp - sp), ahu sat
        vav_state_low = env.observation_space.low[range(4)].tolist() +\
            [-10] + env.action_space.low[[0]].tolist()
        self.vav_state_low = np.array(vav_state_low)
        vav_state_high = env.observation_space.high[range(4)].tolist() +\
            [10] + env.action_space.high[[0]].tolist()
        self.vav_state_high = np.array(vav_state_high)

        # Actions: flow rate and reheat
        self.vav_act_low = env.action_space.low[[1, 2]]
        self.vav_act_high = env.action_space.high[[1, 2]]

    def scale_vav_state(self, states_raw):
        '''
        Scale the states to [-1,1]
        '''
        states_scaled = 2.0 * (states_raw - self.vav_state_low) / \
            (self.vav_state_high - self.vav_state_low) - 1

        return states_scaled

    def rescale_vav_state(self, states_scaled):
        '''
        Recover the raw states from [-1,1]
        '''
        states_raw = self.vav_state_low + \
            (states_scaled + 1.) * 0.5 * \
            (self.vav_state_high - self.vav_state_low)

        return states_raw

    def scale_vav_act(self, actions_raw):
        '''
        Scale the states to [-1,1]
        '''
        actions_scaled = 2.0 * (actions_raw - self.vav_act_low) / \
            (self.vav_act_high - self.vav_act_low) - 1

        return actions_scaled

    def rescale_vav_act(self, actions_scaled):
        '''
        Recover the raw actions from [-1,1]
        '''
        actions_raw = self.vav_act_low + \
            (actions_scaled + 1.) * 0.5 * \
            (self.vav_act_high - self.vav_act_low)

        return actions_raw


def vav_obs(obs_scaled, tempSP, ahuSAT, env, comfort_schedule=True):
    '''Extract VAV states from the states output from the environment,
    Input:
        obs_scaled
    Output:
        - vav_states: A list of states [[VAV1 states], [VAV2 states], ...], scaled
        - comfort_costs: A list of comfort costs [vav1_SqErT, vav2_SqErT, ...], raw
    '''
    vav_scaler = vav_scale(env)
    obs_raw = env.rescale_state(obs_scaled)
    hourOfDay = obs_raw[0]
    dayOfWeek = obs_raw[1]
    occupied_hour = env.occupied_hour
    officeHour = (dayOfWeek < 5) & (
        hourOfDay < occupied_hour[1]) & (
        hourOfDay >= occupied_hour[0])
    weight = env.weight_reward

    weight_comfort = weight[0]
    if comfort_schedule & (not officeHour):
        weight_comfort = weight[1]

    vavs_states = []
    comfort_costs = []
    timeWeather = obs_raw[[0, 1, 2, 3]]   # an array
    for i in range(9):
        zoneTemp = obs_raw[5+i]       # a number
        deltaTemp = zoneTemp - tempSP
        vav_states_raw = np.array(timeWeather.tolist() +
                                  [deltaTemp, ahuSAT])
        vav_states_scaled = vav_scaler.scale_vav_state(vav_states_raw)
        vavs_states.append(vav_states_scaled)
        comfort_costs.append((deltaTemp**2)*weight_comfort)
    return vavs_states, comfort_costs


if __name__ == "__main__":
    '''Test the scale functions
    '''
    from gym_AlphaBuilding import medOff_env
    env = medOff_env.MedOffEnv(building_path='gym_AlphaBuilding/fmuModel/v1_fmu.fmu',
                               sim_days=365,
                               step_size=900,
                               sim_year=2015,
                               fmu_on=False)

    tempSP = 22
    ahuSAT = 12.75
    obs_raw = np.array([0, 3, 7, 0, 84, 20, 21, 22, 23, 24, 23, 22, 21, 20])
    obs_scaled = env.scale_state(obs_raw)
    vavs_states, comfort_costs = vav_obs(obs_scaled, tempSP, ahuSAT, env)
    print('=========== Raw observations ==============')
    print(env.rescale_state(obs_scaled))
    print('====== Scaled observations for VAV ========')
    print(vavs_states)
    print('============= Comfort costs ===============')
    print(comfort_costs)
