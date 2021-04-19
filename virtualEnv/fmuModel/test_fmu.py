# -*- coding: utf-8 -*-
"""
The envelope, battery and PV are modelled through respective FMU
The Chiller and boiler is simplified with constant COP to convert load to electricity consumption

The results are stored in pandas dataframe

@author: walter
"""

from pyfmi import load_fmu
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


def define_IO():
    # define model inputs and outputs
    '''
    actions:
        ahuSAT: AHU supply air (to room) temperature, [degC]
        *FR: terminal flow rate, [kg/s]
        *Reheat: terminal reheat, [W]
    '''
    actions = ['ahuSAT',
               'conf1FR', 'conf1Reheat', 'conf2FR', 'conf2Reheat',
               'enOff1FR', 'enOff1Reheat', 'enOff2FR', 'enOff2Reheat', 'enOff3FR', 'enOff3Reheat',
               'opOff1FR', 'opOff1Reheat', 'opOff2FR', 'opOff2Reheat', 'opOff3FR', 'opOff3Reheat', 'opOff4FR', 'opOff4Reheat']

    '''
	states:
		states_amb: ambient temperature
		states_temp: temperature of controlled rooms
		states_energy: fan, cooling and heating energy consumption
    '''
    states_amb = ['outTemp', 'outSolar', 'outRH']
    states_temp = ['conf1Temp', 'conf2Temp', 'enOff1Temp', 'enOff2Temp', 'enOff3Temp',
                   'opOff1Temp', 'opOff2Temp', 'opOff3Temp', 'opOff4Temp']
    states_energy = ['fanEnergy', 'coolEnergy', 'heatEnergy']

    # create pandas dataframe to save the result
    res = pd.DataFrame(columns=actions+states_amb+states_temp +
                       states_energy+['aveTemp', 'totalEnergy'])
    return actions, states_amb, states_temp, states_energy, res


def FMU_simulate(eplus_path, sim_days=365, with_plots=True, save_res=True):
    '''

    '''
    # Create a simulation time grid
    tStart = 0
    tStop = 900*4*24*sim_days
    hStep = 900  # 15 min
    t = np.arange(tStart, tStop, hStep)
    n_steps = len(t)

    # Load FMUs
    envelope = load_fmu(eplus_path, kind='cs', log_level=7)

    # initialize models
    envelope.initialize(tStart, tStop)
    actions, states_amb, states_temp, states_energy, res = define_IO()
    states = states_amb+states_temp+states_energy

    # initial the counter and heat input
    i = 0
    action_ini = [20,
                  0.1, 0, 0.1, 0,
                  0.1, 0, 0.1, 0, 0.1, 0,
                  0.1, 0, 0.1, 0, 0.1, 0, 0.1, 0]
    res.loc[i, actions] = action_ini

    # Main simulation loop
    while i < n_steps:
        # model the envelope
        envelope.set(actions, res.loc[i, actions].values)
        # Do one step of simulation
        envelope.do_step(current_t=t[i], step_size=hStep, new_step=True)
        # Get the outputs of the simulation
        res.loc[i, states] = envelope.get(states)

        # net power consumption
        res.loc[i, 'aveTemp'] = res.loc[i, states_temp].mean()
        res.loc[i, 'totalEnergy'] = res.loc[i, states_energy].sum()

        # Calculate the input for the next step
        # Interface for RL controller
        control = []
        control_ahuSAT = ahu_sat(res.loc[i, 'aveTemp'])
        control.append(control_ahuSAT)

        for Temp in res.loc[i, states_temp]:
            temp = Temp[0]    # get the value from the list
            terminal_fr = 0.1
            terminal_t = terminal_sat(temp, T_set=22)
            terminal_reheat = reheat(control_ahuSAT, terminal_t, terminal_fr)
            control.append(terminal_fr)
            control.append(terminal_reheat)

        res.loc[i+1, actions] = control
        # print the result
        print('Time {0}: mean zone temp is {1}, AHU SAT is {2}, total energy consumption is {3}'.format(
              t[i], res.loc[i, 'aveTemp'], res.loc[i, 'ahuSAT'], res.loc[i, 'totalEnergy']))
        i += 1
    # delete last timestamp which only has inputs, outputs are na
    res.dropna(inplace=True)

    if save_res:
        time = pd.date_range(
            start='1/1/2015', periods=n_steps, freq='15min').values
        res.reindex(time)
        res.to_csv('result.csv')

    if with_plots:
        f, axarr = plt.subplots(3)
        axarr[0].plot(time, res.aveTemp)
        axarr[0].set_ylabel('Mean zone temp')
        axarr[1].plot(time, res.ahuSAT)
        axarr[1].set_ylabel('AHU SAT')
        axarr[2].plot(time, res.totalEnergy)
        axarr[2].set_ylabel('Total Energy')
        plt.show()


if __name__ == '__main__':
    envelope_path = 'fmuModel/v1_fmu.fmu'
    sim_days = 10  # number of simulation days
    FMU_simulate(envelope_path, sim_days, with_plots=True, save_res=True)
