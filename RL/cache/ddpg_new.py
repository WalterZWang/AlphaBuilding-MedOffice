from gym_AlphaBuilding import medOff_env
import os
from datetime import timedelta
import pandas as pd
import numpy as np
import pytz
import random
import time

from lib import models, utils

import torch as T
import torch.optim as optim
import torch.nn.functional as F

from tensorboardX import SummaryWriter

GAMMA = 0.98
BATCH_SIZE = 64
LR_crt = 5e-3
LR_act = LR_crt/10
LR_GAMMA = 0.95            # Multiplicative factor of learning rate decay
REPLAY_SIZE = 500000
REPLAY_INITIAL = 24*4*30*6      
DATA_COLLECT_BATCH = 24*4  # collect one day data and then train
TRAIN_ITERS = 2            # iterations of training after each day, TI
ALPHA = 0.99             # update rate of target network 
SEED = 350

TEST_ITERS = 2 # compute test evaluation every 2 episodes

CUDA = False

RunName = "setEval"

def test_net(act_net, env, episode=1, device="cpu"):
   
    rewards = 0.0
    energies = 0.0
    comforts = 0.0
    uncDegHours = 0.0
    ahuSat = []
    tempMin = []
    tempMax = []
    steps = 0
    for _ in range(episode):
        states_scaled = env.reset()   # obs include energy obs, states do not include energy obs
        
        while True:
            states_v = utils.float32_preprocessor([states_scaled]).to(device)
            mu_v = act_net(states_v)
            action_scaled = mu_v.squeeze(dim=0).data.cpu().numpy()

            states_scaled, reward, done, comments = env.step(action_scaled)

            energy, comfort, temp_min, temp_max, uncDegHour = comments
            
            action_raw = env.rescale_action(action_scaled)
            rewards += reward
            energies += energy
            comforts += comfort
            uncDegHours += uncDegHour
            ahuSat.append(action_raw[0])
            tempMin.append(temp_min)
            tempMax.append(temp_max)
            steps += 1

            if done:
                break
        tempMinNP = np.array(tempMin)
        tempMaxNP = np.array(tempMax)
    return rewards/episode, energies/episode, comforts/episode, uncDegHours/episode, ahuSat, tempMinNP, tempMaxNP

if __name__ == "__main__":
    device = T.device("cuda" if CUDA else "cpu")

    # initiate log file and tensorboard writer
    save_path = os.path.join("log", "whole-" + RunName)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    writer = SummaryWriter(comment= RunName)

    # initiate environment
    env = medOff_env.MedOffEnv(building_path = 'gym_AlphaBuilding/fmuModel/v1_fmu.fmu',
                                    sim_days = 365,
                                    step_size = 900,
                                    sim_year = 2015,
                                    tz_name = 'America/Los_Angeles')

    test_env = medOff_env.MedOffEnv(building_path = 'gym_AlphaBuilding/fmuModel/v1_fmu.fmu',
                                    sim_days = 365,
                                    step_size = 900,
                                    sim_year = 2015,
                                    tz_name = 'America/Los_Angeles')

    T.manual_seed(SEED)
    # initiate network and agent
    # The last 3 obs ('fanEnergy', 'coolEnergy', 'heatEnergy') are not included when defining observation_space 
    act_net = models.DDPGActor(env.observation_space.shape[0], env.action_space.shape[0], 400, 300).to(device)
    crt_net = models.DDPGCritic(env.observation_space.shape[0], env.action_space.shape[0], 400, 300).to(device)

    tgt_act_net = utils.TargetNet(act_net)
    tgt_crt_net = utils.TargetNet(crt_net)

    best_reward = -1
    best_uncDegHour = -1

    # initiate optimizer for the act and crt network and the agent
    act_opt = optim.Adam(act_net.parameters(), lr=LR_act)
    crt_opt = optim.Adam(crt_net.parameters(), lr=LR_crt)
    # ExponentialLR 
    # act_lr_ex_sch = optim.lr_scheduler.ExponentialLR(act_opt, LR_GAMMA)
    # crt_lr_ex_sch = optim.lr_scheduler.ExponentialLR(crt_opt, LR_GAMMA)
    # ReduceLROnPlateau
    act_lr_op_sch = optim.lr_scheduler.ReduceLROnPlateau(act_opt, 'min', factor=0.5, patience=10)
    crt_lr_op_sch = optim.lr_scheduler.ReduceLROnPlateau(crt_opt, 'min', factor=0.5, patience=10)    
    agent = models.AgentDDPG(act_net, device=device)

    # initiate rl replay buffer
    exp_source = utils.ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=1)
    buffer = utils.ExperienceReplayBuffer(exp_source, buffer_size=REPLAY_SIZE)

    exp_idx = 0
    episode = 0

    ahuSat_pd = pd.DataFrame()


    with open(os.path.join(save_path,"{0}.log".format(RunName)), "a") as f:
        with utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:
            while True:

                buffer.populate(DATA_COLLECT_BATCH)
                
                if len(buffer) < REPLAY_INITIAL:  # Do not train actor until REPLAY_INITIAL episodes
                    continue
                
                exp_idx += DATA_COLLECT_BATCH

                ## Run test on the test environment
                if exp_idx >= ((env.n_steps -1) * (TEST_ITERS + episode)):
                    episode = exp_idx // (env.n_steps -1)
                    ts = time.time()
                    rewards, energy, comfort, uncDegHour, ahuSat, tempMin, tempMax = test_net(act_net, test_env, 1, device=device)
                    ahu_sat_mean = np.array(ahuSat).mean()
                    ahuSat_pd[episode]=ahuSat
                    print("Test done in %.2f sec, reward %.3f"% (
                        time.time() - ts, rewards))
                    writer.add_scalar("test_ddpg_reward", rewards, episode)
                    writer.add_scalar("test_ddpg_energy",energy, episode)
                    writer.add_scalar("test_ddpg_comfort",comfort, episode)
                    writer.add_scalar("test_ddpg_uncDegHour",uncDegHour, episode)
                    writer.add_scalar('test_ddpg_ahuSAT',ahu_sat_mean, episode)
                    writer.add_scalar("lr_act", act_opt.param_groups[0]["lr"], episode)
                    writer.add_scalar("lr_crt", crt_opt.param_groups[0]["lr"], episode)
                    f.write("%d,%.2f,%.2f,%.2f,%.2f\n"%(episode,rewards,energy,comfort,uncDegHour))
                    if best_reward is None or best_reward < rewards:
                        if best_reward is not None:
                            print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
                            name = "best_%+.3f_%d.dat" % (rewards, exp_idx)
                            fname = os.path.join(save_path, name)
                            T.save(act_net.state_dict(), fname)
                        best_reward = rewards
                    
                    # update learning rate and save the ahuSAT
                    # act_lr_ex_sch.step()
                    # crt_lr_ex_sch.step()
                    ahuSat_pd.to_csv('analysis/ahuSAT/{0}_ddpg.csv'.format(RunName))     

                ## Train the act and crt                
                for _ in range(TRAIN_ITERS):
                    batch = buffer.sample(BATCH_SIZE)
                    states_scaled, actions_scaled, rewards, dones, last_states_scaled = utils.unpack_batch_ddqn(batch)
                    
                    states = utils.float32_preprocessor(states_scaled).to(device)
                    actions = utils.float32_preprocessor(actions_scaled).to(device)
                    rewards = utils.float32_preprocessor(rewards).to(device)
                    last_states = utils.float32_preprocessor(last_states_scaled).to(device)
                    dones_mask = T.ByteTensor(dones).to(device)

                    ## train critic
                    # calculate the target value
                    tgt_crt_net.target_model.eval()    # turn off the batch normalization
                    tgt_act_net.target_model.eval()

                    with T.no_grad():
                        last_act = tgt_act_net.target_model(last_states)
                        q_last = tgt_crt_net.target_model(last_states, last_act)
                        q_last[dones_mask] = 0.0
                        q_ref = rewards.unsqueeze(dim=-1) + q_last * GAMMA

                    # calculate loss and optimize
                    crt_net.train()
                    crt_opt.zero_grad()
                    q = crt_net(states, actions)

                    critic_loss = F.mse_loss(q, q_ref)
                    critic_loss.backward()
                    crt_opt.step()
                    tb_tracker.track("loss_critic", critic_loss, exp_idx)
                    tb_tracker.track("q_value", q_ref.mean(), exp_idx)
                    
                    ## train actor
                    # calculate the loss and optimize
                    crt_net.eval()
                    act_net.train()
                    act_opt.zero_grad()
                    cur_actions = act_net(states)
                    actor_loss = -crt_net(states, cur_actions)
                    actor_loss = actor_loss.mean()
                    actor_loss.backward()
                    act_opt.step()  # only optimize act network, but not the crt network
                    tb_tracker.track("loss_actor", actor_loss, exp_idx)
                    
                    # update the learning rate
                    crt_lr_op_sch.step(critic_loss)
                    act_lr_op_sch.step(actor_loss)

                    tgt_act_net.alpha_sync(alpha=ALPHA)
                    tgt_crt_net.alpha_sync(alpha=ALPHA)
                
        pass
