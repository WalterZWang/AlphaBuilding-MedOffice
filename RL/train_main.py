from gym_AlphaBuilding import medOff_env
import numpy as np
import os

from tensorboardX import SummaryWriter

if __name__ == "__main__":

    env = medOff_env.MedOffEnv(building_path = 'gym_AlphaBuilding/fmuModel/v1_fmu.fmu',
                                    sim_days = 365,
                                    step_size = 900,
                                    sim_year = 2015,
                                    tz_name = 'America/Los_Angeles')

    # from ddpg_util import Agent
    # agent = Agent(env, act_lr=0.000025, crt_lr=0.00025, tau=0.001, 
    #               batch_size=64,  layer1_size=400, layer2_size=300)

    # from sac_util import Agent
    # agent = Agent(env, reward_scale=3)

    from td3_util import Agent
    agent = Agent(env, act_lr=0.000025, crt_lr=0.00025, tau=0.001, 
                  batch_size=64,  layer1_size=256, layer2_size=256)

    RunName = 'ppo_test'
    # initiate log file and tensorboard writer
    save_path = os.path.join("log")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    writer = SummaryWriter(comment = RunName)
    np.random.seed(0)

    with open(os.path.join(save_path,"{0}.log".format(RunName)), "a") as f:
        #agent.load_models()

        best_score = np.NINF
        load_checkpoint = False    # load the model and test, not learn
        total_reward_history = []
        total_energy_history = []
        total_comfort_history = []
        total_uncDegHour_history = []
        total_crt_loss_history = []
        total_act_loss_history = []

        if load_checkpoint:
            agent.load_models()

        for episode in range(22):
            obs = env.reset()
            done = False
            total_reward = 0
            total_energy = 0
            total_comfort = 0
            total_uncDegHour = 0
            total_crt_loss = 0
            total_act_loss = 0
            while not done:
                act = agent.choose_action(obs)
                new_state, reward, done, comments = env.step(act)
                total_reward += reward
                agent.remember(obs, act, reward, new_state, int(done))

                if not load_checkpoint:
                    loss = agent.learn()
                    if loss:
                        crt_loss, act_loss = loss
                        # Book keeping
                        energy, comfort, temp_min, temp_max, uncDegHour = comments
                        total_energy += energy
                        total_comfort += comfort
                        total_uncDegHour += uncDegHour
                        total_crt_loss += crt_loss
                        total_act_loss += act_loss
                
                obs = new_state

            total_reward_history.append(total_reward)
            total_energy_history.append(total_energy)
            total_comfort_history.append(total_comfort)
            total_uncDegHour_history.append(total_uncDegHour)
            total_crt_loss_history.append(total_crt_loss)
            total_act_loss_history.append(total_act_loss)

            f.write("%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n"%(episode,total_reward,total_energy,
                         total_comfort,total_uncDegHour,total_crt_loss,total_act_loss))
            writer.add_scalar("reward", total_reward, episode)
            writer.add_scalar("energy",total_energy, episode)
            writer.add_scalar("comfort",total_comfort, episode)
            writer.add_scalar("uncDegHour",total_uncDegHour/(9*8760*4), episode)
            writer.add_scalar("crt_loss",total_crt_loss, episode)
            writer.add_scalar("act_loss",total_act_loss, episode)       

            avg_score = np.mean(total_reward_history[-10:])   # score is the total reward of 1 episode
            if avg_score > best_score:
                best_score = avg_score
                if not load_checkpoint:
                    agent.save_models()

            print('episode ', episode, 'score %.2f' % total_reward,
                  'trailing 10 games avg %.3f' % avg_score)