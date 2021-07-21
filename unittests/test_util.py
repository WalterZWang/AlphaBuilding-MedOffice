import unittest
import numpy as np

from gym_AlphaBuilding import medOff_env

import sys
sys.path.append('/mnt/shared/RL')

from dist_util import vav_scale, vav_obs

def SSE(array1, array2):
    '''Calculate the Sum of Squared Error of two np arrays
    '''
    sse = ((array1 - array2)**2).sum()
    return sse

class TestDist(unittest.TestCase):
    
    def setUp(self):
        self.env = medOff_env.MedOffEnv(building_path='gym_AlphaBuilding/fmuModel/v1_fmu.fmu',
                            sim_days=365,
                            step_size=900,
                            sim_year=2015,
                            fmu_on=False)
        self.vav_scaler = vav_scale(self.env)
    
    def tearDown(self):
        pass

    def testVavScaleState(self):
        # States: hour, dayOfWeek, outTemp, solar, dT (inTemp - sp), ahu sat
        state1_raw = np.array([0, 0, -30, 0, 0, 10])
        state1_scaled_true = np.array([-1, -1, -1, -1, 0, -1])
        state1_scaled = self.vav_scaler.scale_vav_state(state1_raw)
        state1_raw_recon = self.vav_scaler.rescale_vav_state(state1_scaled)
        for scaled, scaled_true in zip(state1_scaled, state1_scaled_true):
            self.assertAlmostEqual(scaled, scaled_true)
        for raw, raw_recon in zip(state1_raw, state1_raw_recon):
            self.assertAlmostEqual(raw, raw_recon)

        state2_raw = np.array([23, 6, 50, 1500, 10, 30])
        state2_scaled_true = np.array([1, 1, 1, 1, 1, 1])
        state2_scaled = self.vav_scaler.scale_vav_state(state2_raw)
        state2_raw_recon = self.vav_scaler.rescale_vav_state(state2_scaled)
        for scaled, scaled_true in zip(state2_scaled, state2_scaled_true):
            self.assertAlmostEqual(scaled, scaled_true)
        for raw, raw_recon in zip(state2_raw, state2_raw_recon):
            self.assertAlmostEqual(raw, raw_recon)

    def testVavScaleAction(self):
        # Actions: flow rate and reheat
        act1_raw = np.array([0, 0])
        act1_scaled_true = np.array([-1, -1])
        act1_scaled = self.vav_scaler.scale_vav_act(act1_raw)
        act1_raw_recon = self.vav_scaler.rescale_vav_act(act1_scaled)
        for scaled, scaled_true in zip(act1_scaled, act1_scaled_true):
            self.assertAlmostEqual(scaled, scaled_true)
        for raw, raw_recon in zip(act1_raw, act1_raw_recon):
            self.assertAlmostEqual(raw, raw_recon)

        act2_raw = np.array([1, 10000])
        act2_scaled_true = np.array([1, 1])
        act2_scaled = self.vav_scaler.scale_vav_act(act2_raw)
        act2_raw_recon = self.vav_scaler.rescale_vav_act(act2_scaled)
        for scaled, scaled_true in zip(act2_scaled, act2_scaled_true):
            self.assertAlmostEqual(scaled, scaled_true)
        for raw, raw_recon in zip(act2_raw, act2_raw_recon):
            self.assertAlmostEqual(raw, raw_recon)

    def testVavObs(self):
        # obs: time + weather + zoneTemp
        time = [12, 3] # ['hour', 'dayOfWeek']
        weather = [10, 500, 80]   # ['outTemp', 'outSolar', 'outRH']
        
        zoneTemp = list(range(18, 27))
        tempSP = 22
        zoneDeltaTemp = np.array(zoneTemp) - tempSP
        obs_all_raw = time + weather + zoneTemp
        obs_all_scaled = self.env.scale_state(obs_all_raw)
        ahuSAT = 13
        weight = self.env.weight_reward[0]
        vavs_states_test, comfort_costs_test = vav_obs(obs_all_scaled, tempSP, ahuSAT, self.env)
        error_state_acc = 0
        for i, vav_state_test in enumerate(vavs_states_test):
            deltaTemp = zoneDeltaTemp[i]
            vav_state_true_raw = np.array(time + weather[:-1] + [deltaTemp, ahuSAT])
            vav_state_true_scaled = self.vav_scaler.scale_vav_state(vav_state_true_raw)
            error_state_acc += SSE(vav_state_true_scaled, vav_state_test)
        comfort_costs_true = (zoneDeltaTemp)**2*weight
        error_comfort_acc = SSE(comfort_costs_true, comfort_costs_test)
        self.assertAlmostEqual(error_state_acc, 0)
        self.assertAlmostEqual(error_comfort_acc, 0)

if __name__ == '__main__':
    unittest.main()
