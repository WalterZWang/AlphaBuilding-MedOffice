from gym_AlphaBuilding import medOff_env
from dist_util import vav_obs
import numpy as np
import os

from tensorboardX import SummaryWriter


def train(algorithm, env, seed, save_path):

    input_dims = 6  # hour, dayOfWeek, outTemp, solar, \
    # dist from sp (inTemp - sp), ahu sat
    n_actions = 2   # flow rate, reheat (sat)

    if algorithm == 'ddpg':
        from ddpg_util import Agent
        vavs = []
        for _ in range(9):
            vav = Agent(input_dims, n_actions,
                        act_lr=0.00001, crt_lr=0.0001, tau=0.001,
                        batch_size=64,  layer1_size=32, layer2_size=16)
            vavs.append(vav)
    # elif algorithm == 'sac':
    #     from sac_util import Agent
    #     agent = Agent(env, reward_scale=3)
    # elif algorithm == 'td3':
    #     from td3_util import Agent
    #     agent = Agent(env, act_lr=0.000025, crt_lr=0.00025, tau=0.001,
    #                   batch_size=64,  layer1_size=256, layer2_size=256)

    RunName = 'dist_{}_run{}'.format(algorithm, seed)

    writer = SummaryWriter(comment=RunName)
    np.random.seed(seed)

    with open(os.path.join(save_path, "{0}.log".format(RunName)), "a") as f:

        ahuSAT = 12.75
        ahuSAT_scaled = 2.0 * (ahuSAT - env.action_space.low[0]) / \
            (env.action_space.high[0] - env.action_space.low[0]) - 1
        tempSP = 22

        # For the whole HVAC system
        best_score = np.NINF
        total_reward_history = []
        total_energy_history = []
        total_comfort_history = []
        total_uncDegHour_history = []
        total_crt_loss_history = []
        total_act_loss_history = []

        # For each VAV, to determine when to save the network
        vavs_best_score = []
        vavs_reward_history = []
        for _ in range(9):
            vavs_best_score.append(np.NINF)
            vavs_reward_history.append([])

        for episode in range(21):
            obs_scaled = env.reset()
            vavs_s_old, _ = vav_obs(obs_scaled, tempSP, ahuSAT, env)

            done = False
            total_reward = 0
            total_energy = 0
            total_comfort = 0
            total_uncDegHour = 0
            total_crt_loss = 0
            total_act_loss = 0
            vavs_episode_reward = [0]*9

            while not done:
                # All the states, acts here are scaled
                # step1: Calculate actions
                vavs_a = []
                acts = [ahuSAT_scaled]
                for i, vav in enumerate(vavs):
                    vav_s = vavs_s_old[i]
                    vav_a = vav.choose_action(vav_s)
                    acts.extend(vav_a.tolist())
                    vavs_a.append(vav_a)
                acts = np.array(acts)

                # step2: One step simulation
                new_obs, reward, done, comments = env.step(acts)
                hvacEnergy, hvacComfort, temp_min, temp_max, hvacUncDegHour, fanE, coolE, heatE = comments
                total_reward += reward

                # step3: Calculate reward and save to the buffer
                ahuEnergy = fanE + coolE + heatE    # unit: kWh, already consisder gas-electricity
                vavs_s_new, vavs_c = vav_obs(
                    new_obs, tempSP, ahuSAT, env)
                acts_raw = env.rescale_action(acts)
                ahuFR = acts_raw[1::2].sum()
                for i, vav in enumerate(vavs):
                    comfort_cost = vavs_c[i]
                    reheatE = acts_raw[2+i*2]/(1000*4)         # kWh
                    ahuE = (acts_raw[1+i*2]/ahuFR)*ahuEnergy   # kWh
                    energy_cost = reheatE + ahuE               # kWh
                    vav_reward = -1 * (10*comfort_cost + energy_cost)
                    vav.remember(vavs_s_old[i], vavs_a[i],
                                 vav_reward, vavs_s_new[i], int(done))
                    vavs_episode_reward[i] += vav_reward

                    loss = vav.learn()
                    if loss:
                        crt_loss, act_loss = loss
                        total_crt_loss += crt_loss
                        total_act_loss += act_loss
                # Book keeping for the whole HVAC System
                total_energy += hvacEnergy
                total_comfort += hvacComfort
                total_uncDegHour += hvacUncDegHour

                vavs_s_old = vavs_s_new

            # Determine whether to save the vav controller or not
            for vav_index in range(9):
                vavs_reward_history[vav_index].append(
                    vavs_episode_reward[vav_index])
                avg_score = np.mean(vavs_reward_history[vav_index][-10:])
                if avg_score > vavs_best_score[vav_index]:
                    vavs_best_score[vav_index] = avg_score
                    vavs[vav_index].save_models(
                        RunName+'vav_{}'.format(vav_index))

            # Save the results of the whole HVAC system
            total_reward_history.append(total_reward)
            total_energy_history.append(total_energy)
            total_comfort_history.append(total_comfort)
            total_uncDegHour_history.append(total_uncDegHour)
            total_crt_loss_history.append(total_crt_loss)
            total_act_loss_history.append(total_act_loss)
            f.write("%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n" % (episode, total_reward, total_energy,
                                                            total_comfort, total_uncDegHour, total_crt_loss, total_act_loss))
            writer.add_scalar("reward", total_reward, episode)
            writer.add_scalar("energy", total_energy, episode)
            writer.add_scalar("comfort", total_comfort, episode)
            writer.add_scalar("uncDegHour", total_uncDegHour, episode)
            writer.add_scalar("crt_loss", total_crt_loss, episode)
            writer.add_scalar("act_loss", total_act_loss, episode)
            for vav_index in range(9):
                writer.add_scalar("VAV{}_reward".format(vav_index), vavs_episode_reward[vav_index], episode)

if __name__ == "__main__":

    # initiate log file and tensorboard writer
    save_path = os.path.join("log", "exp3-higherComfortWeight")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    algorithm = 'ddpg'
    env = medOff_env.MedOffEnv(building_path='gym_AlphaBuilding/fmuModel/v1_fmu.fmu',
                               sim_days=365,
                               step_size=900,
                               sim_year=2015)

    for seed in range(1, 2):
        train(algorithm, env, seed, save_path)
