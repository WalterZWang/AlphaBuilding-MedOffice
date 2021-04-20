# -*- coding: utf-8 -*-
"""
The envelope, battery and PV are modelled through respective FMU
The Chiller and boiler is simplified with constant COP to convert load to electricity consumption

The results are stored in pandas dataframe

@author: walter
"""

from medOff_env import MedOffEnv, ExperienceBuffer
import numpy as np
import pylab as plt
import pandas as pd


def terminal_sat(T_room, T_set=22):
    '''
    A simple controller for hvac input heat

    '''
    if T_room - T_set > 1.5:
        T_sa = T_set-10
    elif T_room - T_set > 1.2:
        T_sa = T_set-6
    elif T_room - T_set > 0.8:
        T_sa = T_set-3
    elif T_room - T_set < -1.8:
        T_sa = T_set+10
    elif T_room - T_set < -1.3:
        T_sa = T_set+6
    elif T_room - T_set < -0.8:
        T_sa = T_set+3
    else:
        T_sa = T_set-1

    return T_sa


def ahu_sat(T_mean, T_set=22):
    # lower supply air to avoid overheating as no cooling in terminal
    return terminal_sat(T_mean)-2


def reheat(ahuSAT, T_sa, FR=0.1):
    reheat = 1004*max(0, T_sa-ahuSAT)*FR
    return reheat


def test_agent(obs):
    control = []
    zone_temp = obs[3:12]
    control_ahuSAT = ahu_sat(np.mean(zone_temp))
    control.append(control_ahuSAT)
    for temp in zone_temp:
        terminal_fr = 0.1
        terminal_t = terminal_sat(temp, T_set=22)
        terminal_reheat = reheat(control_ahuSAT, terminal_t, terminal_fr)
        control.append(terminal_fr)
        control.append(terminal_reheat)
    return np.array(control)


def gym_simulate(eplus_path, sim_days=365):
    '''

    '''
    env = MedOffEnv(building_path=eplus_path,
                    sim_days=sim_days,
                    step_size=900,
                    sim_year=2015,
                    tz_name='UTC')

    if env is None:
        print('Error: Failed to load the fmu')
        quit()

    buffer = ExperienceBuffer(env.obs_names, env.action_names)

    for ep in range(2):
        state = env.reset()
        for _ in range(env.n_steps-1):
            action = test_agent(state)
            new_state, reward, done, _ = env.step(action)
            buffer.append(action, state, reward)
            state = new_state
            if done:
                # actions_data = buffer.action_data()
                obs_data = buffer.obs_data()
                obs_data.to_csv('result_ep{0}.csv'.format(ep))


if __name__ == '__main__':
    envelope_path = 'fmuModel/v1_fmu.fmu'
    sim_days = 2  # number of simulation days
    gym_simulate(envelope_path, sim_days)
