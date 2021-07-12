from gym_AlphaBuilding import medOff_env
import numpy as np
import os

from tensorboardX import SummaryWriter


def train(algorithm, env, seed, save_path):

    input_dims = 6  # hour, dayOfWeek, outTemp, solar, \
    # dist from sp (inTemp - sp), ahu sat
    n_actions = 2   # flow rate, reheat

    if algorithm == 'ddpg':
        from ddpg_util import Agent
        vavs = []
        for _ in range(9):
            vav = Agent(input_dims, n_actions,
                        act_lr=0.000025, crt_lr=0.00025, tau=0.001,
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

        best_score = np.NINF
        total_reward_history = []
        total_energy_history = []
        total_comfort_history = []
        total_uncDegHour_history = []
        total_crt_loss_history = []
        total_act_loss_history = []

        for episode in range(21):
            obs_scaled = env.reset()
            vavs_s_old, _ = medOff_env.vav_obs(obs_scaled, tempSP, ahuSAT, env)

            done = False
            total_reward = 0
            total_energy = 0
            total_comfort = 0
            total_uncDegHour = 0
            total_crt_loss = 0
            total_act_loss = 0
            while not done:
                # All the states, acts here are scaled
                # Calculate actions
                vavs_a = []
                acts = [ahuSAT_scaled]
                for i, vav in enumerate(vavs):
                    vav_s = vavs_s_old[i]
                    vav_a = vav.choose_action(vav_s)
                    acts.extend(vav_a.tolist())
                    vavs_a.append(vav_a)
                acts = np.array(acts)

                new_obs, reward, done, comments = env.step(acts)
                total_reward += reward

                # Calculate reward and save to the buffer
                vavs_s_new, vavs_c = medOff_env.vav_obs(
                    new_obs, tempSP, ahuSAT, env)
                acts_raw = env.rescale_action(acts)
                for i, vav in enumerate(vavs):
                    comfort_cost = vavs_c[i]
                    energy_cost = acts_raw[2+i*2]/1000        # kW
                    reward = -1 * (10*comfort_cost + energy_cost)
                    vav.remember(vavs_s_old[i], vavs_a[i],
                                 reward, vavs_s_new[i], int(done))

                    loss = vav.learn()
                    if loss:
                        crt_loss, act_loss = loss
                        total_crt_loss += crt_loss
                        total_act_loss += act_loss
                # Book keeping
                energy, comfort, temp_min, temp_max, uncDegHour, fanE, coolE, heatE = comments
                total_energy += energy
                total_comfort += comfort
                total_uncDegHour += uncDegHour

                vavs_s_old = vavs_s_new

            total_reward_history.append(total_reward)
            total_energy_history.append(total_energy)
            total_comfort_history.append(total_comfort)
            total_uncDegHour_history.append(total_uncDegHour)
            total_crt_loss_history.append(total_crt_loss)
            total_act_loss_history.append(total_act_loss)

            f.write("%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n" % (episode, total_reward, total_energy,
                                                            total_comfort, total_uncDegHour, total_crt_loss, total_act_loss))
            total_uncDegHour = total_uncDegHour/(9*8760*4*(5/7)*(14/24))
            writer.add_scalar("reward", total_reward, episode)
            writer.add_scalar("energy", total_energy, episode)
            writer.add_scalar("comfort", total_comfort, episode)
            writer.add_scalar("uncDegHour", total_uncDegHour, episode)
            writer.add_scalar("crt_loss", total_crt_loss, episode)
            writer.add_scalar("act_loss", total_act_loss, episode)

            # score is the total reward of 1 episode
            avg_score = np.mean(total_reward_history[-10:])
            if avg_score > best_score:
                best_score = avg_score
                for i, vav in enumerate(vavs):
                    vav.save_models(RunName+'vav_{}'.format(i))

            print('episode ', episode, 'score %.2f' % total_reward,
                  'trailing 10 games avg %.3f' % avg_score)


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

    for seed in range(1, 3):
        train(algorithm, env, seed, save_path)
