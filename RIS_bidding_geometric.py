# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 10:47:15 2023

@author: Stefan Schwarz
"""
# !pip install stable-baselines3[extra]

import numpy as np
import RIS_env_geometric_v2
# import os

import supersuit as ss
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as pyplt

import time

import pickle

from stable_baselines3 import PPO #, A2C

from wireless_fuf import Gauss_channel, RIS_alloc_response


""" Basic parameters of the environment; more are defined in the RIS_Competition module """
# seed = 1
N_UE = 10 # number of users per operator
N_RIS = 20 # number of RISs in total
N_BS = 3 # number of BSs per operator
M_RIS = 1000 # number of RIS elements
N_OP = 2 # number of operators
ROI_size = np.array([[-10,-10],[10,10]])*5 # size of ROI
budget = 1*np.ones(N_OP) # budget for bidding
# budget[0] = 8 # use this to vary the budget of operator 1
retrain = True # activate to retrain the agent
test = False # activate to test on random environments
test_MF = False # activate to test impact of microscopic fading
NN = 500 # number of microscopic fading realizations
K = np.array([1e2,3*1e0])*1e0 # K-factor under LOS/NLOS
rng1 = np.random.default_rng(1) # random number generator
Ps = 20 # power per subcarrier in dBm
ps_lin = 10**((Ps-30)/10) # power in linear scale
N0 = -174 # noise PSD dBm/Hz
F = 6 # noise figure in dB
Bs = 15e3 # subcarrier bandwidth
sigma_n2 = 10**((N0 + 10*np.log10(Bs) + F - 30)/10) # noise power


f_name = str(N_BS) + 'BS_' + str(N_UE) + 'UE_' + str(N_RIS) + 'RIS_' + str(M_RIS) + 'M' # file name

start_time = time.time()

######################################################################## 
""" Get fading channels of direct links """    
########################################################################
def direct_channel(LOS_UE_BS,pow_ue_bs,BS_UE_assoc):  # indicator for LOS between UE and BS, path gain between UE and BS, association of UEs to BSs  
    channels_BS_UE = Gauss_channel((1,1,N_UE,N_BS,N_OP,NN),rng1) # get Gaussian microscopic fading channels    
    # combine Gauss channel with LOS (non-fading) channel
    KK = np.reshape(K[0]*LOS_UE_BS + K[1]*(1-LOS_UE_BS),(1,1,N_UE,N_BS,N_OP))
    KK = np.repeat(KK[:,:,:,:,:,np.newaxis],NN,axis = 5) # add new axis and repeat NN times along it
    direct_channel_BS_UE = np.sqrt(KK/(1+KK))*np.exp(1j*rng1.uniform(0,np.pi*2,(1,1,N_UE,N_BS,N_OP,NN))) # directional channel
    for no in range(N_OP):
        for nu in range(N_UE):
            temp_chan = direct_channel_BS_UE[:,:,nu,BS_UE_assoc[nu,no],no,:]
            direct_channel_BS_UE[:,:,nu,BS_UE_assoc[nu,no],no,:] = temp_chan*np.exp(-1j*np.angle(temp_chan)) # serving BS is phase-synchronized
    direct_channel_BS_UE = direct_channel_BS_UE + np.sqrt(1/(1+KK))*channels_BS_UE # direct + scattering channel 
    pl_lin = np.sqrt(pow_ue_bs)
    direct_channel_BS_UE = direct_channel_BS_UE*pl_lin[:,:,:,np.newaxis] # channel including path loss   
    return direct_channel_BS_UE

######################################################################## 
""" Get SINR of direct-only links """    
########################################################################
def get_direct_SINR(direct_signal_BS_UE,BS_UE_assoc): # direct received signals between all BSs and all UEs, association of UEs to BSs
    # calculate SINR of direct links only
    direct_power = ps_lin*np.reshape(np.abs(direct_signal_BS_UE)**2,(N_UE,N_BS,N_OP,NN)) # include transmit power
    direct_SINR = np.zeros((N_UE,N_OP,NN))
    UE_power = np.zeros((N_UE,N_OP,NN))
    Int_power = np.zeros((N_UE,N_OP,NN))
    for no in range(N_OP):
        for nu in range(N_UE):
            UE_power[nu,no,:] = direct_power[nu,BS_UE_assoc[nu,no],no,:] # intended signal power
            Int_power_tmp = direct_power[nu,[i for i in range(N_BS) if i != BS_UE_assoc[nu,no]],no,:] # interference from other BSs 
            #Int_power = np.sum(Int_power[Int_power > UE_power]) 
            Int_power[nu,no,:] = np.sum(Int_power_tmp,0,keepdims = True) # sum interference
            #Int_power = 0
            direct_SINR[nu,no,:] = 10*np.log10(UE_power[nu,no,:]/(Int_power[nu,no,:]+sigma_n2)) # SINR
    return direct_SINR, UE_power, Int_power

######################################################################## 
""" Get effective RIS-assisted channels """    
########################################################################
def RIS_channel(LOS_RIS_UE,LOS_RIS_BS,RIS_BS_channel,channels_BS_RIS,RIS_UE_channel,channels_RIS_UE,BS_UE_assoc,opt_RIS_resp,PL_RIS_BS,PL_RIS_UE,RIS_assoc,rng1):
    # K factors for UEs -- depending on LOS/NLOS
    KK_UE = np.reshape(K[0]*LOS_RIS_UE + K[1]*(1-LOS_RIS_UE),(1,1,N_RIS,N_UE,N_OP)) # K-factor between UE and RIS
    KK_UE = np.repeat(KK_UE,M_RIS,axis = 1)
    KK_UE = np.repeat(KK_UE[:,:,:,:,:,np.newaxis],NN,axis = 5)

    RIS_channel_BS_UE = np.zeros((1,1,N_UE,N_BS,N_OP,NN)) + 1j*np.zeros((1,1,N_UE,N_BS,N_OP,NN))
    rand_RIS_resp = np.exp(1j*rng1.uniform(0,np.pi*2,(NN,M_RIS,N_RIS,N_OP))) # randomized RIS responses for uncontrolled RISs and interferences
    for no in range(N_OP):
        for nb in range(N_BS):
            for nr in range(N_RIS):
                
                ris_bs_direct = np.reshape(RIS_BS_channel[:,nr,nb,no],[M_RIS,1]) # directional channel between RIS and BS
                ris_bs_direct = np.repeat(ris_bs_direct[:,:,np.newaxis],NN,axis = 2) #*np.exp(1j*rng1.uniform(0,np.pi*2,(1,1,NN)))
                ris_bs = np.moveaxis(ris_bs_direct,2,0)

                for nu in range(N_UE):            
                    ris_ue_direct = np.reshape(RIS_UE_channel[:,nr,nu,no],[1,M_RIS]) # directional channel between RIS and UE
                    if BS_UE_assoc[nu,no] != nb: # apply a random phase shift to unintended signals (not phase-corrected)
                        ris_ue_direct = np.repeat(ris_ue_direct[:,:,np.newaxis],NN,axis = 2)*np.exp(1j*rng1.uniform(0,np.pi*2,(1,1,NN))) # random phase-shift for unsynchronized channels
                    else:
                        ris_ue_direct = np.repeat(ris_ue_direct[:,:,np.newaxis],NN,axis = 2)
                    ris_ue_scatter = channels_RIS_UE[:,:,nr,nu,no,:] # scattering channel between RIS and UE
                    kk_ue = KK_UE[:,:,nr,nu,no,:]
                    ris_ue = np.sqrt(kk_ue/(1+kk_ue))*ris_ue_direct + np.sqrt(1/(1+kk_ue))*ris_ue_scatter # channel between UE and RIS
                    ris_ue = np.moveaxis(ris_ue,2,0)
                    
                    
                    if RIS_assoc[nr] == no: # RIS is controlled by OP -- use optimized phases for each user
                        ris_chan = np.matmul(ris_ue,np.diag(opt_RIS_resp[:,nr,nu,no])) 
                    else: # RIS not controlled -- apply random phase shift at RIS
                        rand_RIS_resp_tmp = np.reshape(rand_RIS_resp[:,:,nr,no],ris_ue.shape) # different BSs und UEs see the same RIS response -- assume this changes over time (different UEs served at other OPs)
                        ris_chan = ris_ue*rand_RIS_resp_tmp
                        
                    ris_chan = np.moveaxis(np.matmul(ris_chan,ris_bs),0,2)              
                    pl_lin = 10**(-PL_RIS_BS[nr,nb,no]/20)*10**(-PL_RIS_UE[nr,nu,no]/20) # total path loss BS-to-RIS and RIS-to-UE
                    RIS_channel_BS_UE[:,:,nu,nb,no,:] += ris_chan*pl_lin                    
                        
    return RIS_channel_BS_UE

######################################################################## 
""" Get SINR of RIS-assisted transmissions """    
########################################################################
def get_SINR(direct_signal_BS_UE,RIS_signal_BS_UE,BS_UE_assoc):
     ## calculate net-effect on SINR
     total_signal_BS_UE = direct_signal_BS_UE + RIS_signal_BS_UE  # total received signal (direct + RIS-assisted) 
             
     total_power = ps_lin*np.reshape(np.abs(total_signal_BS_UE)**2,(N_UE,N_BS,N_OP,NN))
     total_SINR = np.zeros((N_UE,N_OP,NN))
     UE_power = np.zeros((N_UE,N_OP,NN))
     Int_power = np.zeros((N_UE,N_OP,NN))
     for no in range(N_OP):
         for nu in range(N_UE):
             UE_power[nu,no,:] = total_power[nu,BS_UE_assoc[nu,no],no,:] # intended signal power
             Int_power_tmp = total_power[nu,[i for i in range(N_BS) if i != BS_UE_assoc[nu,no]],no,:] # interference from other BSs 
             #Int_power = np.sum(Int_power[Int_power > UE_power]) 
             Int_power[nu,no,:] = np.sum(Int_power_tmp,0,keepdims = True)
             # Int_power = 0
             total_SINR[nu,no,:] = 10*np.log10(UE_power[nu,no,:]/(Int_power[nu,no,:]+sigma_n2))    
        
     return total_SINR, UE_power, Int_power

######################################################################## 
""" Main """    
########################################################################
if __name__ == "__main__":
    """ instantiate environment """
    env = RIS_env_geometric_v2.parallel_env(render_mode=None,N_OP = N_OP,N_UE = N_UE, N_RIS = N_RIS, N_BS = N_BS,M_RIS = M_RIS, ROI_size = ROI_size, budget = budget,K = K, N0 = N0, F=F, Bs=Bs, Ps=Ps,ps_lin=ps_lin,sigma_n2=sigma_n2)
    observations, infos = env.reset(4) # reset environment -- not important here, only for debugging below
    # registration does not seem to work for parallel_env
    # gym.register(id = 'RIS_env_v3',entry_point = 'RIS_env:parallel_env')
    # env = gym.make('RIS_env-v0',N_OP = N_OP,N_UE = N_UE, N_RIS = N_RIS, N_BS = N_BS,M_RIS = M_RIS, ROI_size = ROI_size, budget = budget)
    
    """ check if the environment does something useful -- just for debugging purposes"""
    # terminal = False
    # while ~terminal:
    #     # print(env.state['Op0'],flush = True)
    #     # print(env.state['Op1'],flush = True)
    #     # this is where you would insert your policy
    #     actions = {}
    #     for agent in env.agents:
    #         #actions[agent] = np.ones((N_RIS,1),dtype = bool)
    #         invalid = True
    #         while invalid:
    #             if env.price > env.budget[agent]: # not enough budget to continue bidding
    #                 actions[agent] = np.zeros(N_RIS,dtype = bool)    
    #                 invalid = False
    #             else:    
    #                 action = env.action_space(agent).sample() # random sample from action space 
    #                 # action = np.ones(N_RIS,dtype = bool) 
    #                 if np.sum(action*env.price) > env.budget[agent]: # allow only valid actions
    #                     continue
    #                 else:                
    #                     actions[agent] = action
    #                     invalid = False
    
    #     observations, rewards, terminated, truncated, infos = env.step(actions)
    #     term_list = []
    #     for agent in env.agents:
    #         term_list = np.append(term_list,terminated[agent])
    #     terminal = np.prod(term_list).astype(bool)
    #     # print(env.agents)
    
    """ TB logging """
    log_dir = "./RIS_tb/"  # store logs for tensorboard
    
    """ convert to vector environment """
    vec_env = ss.pettingzoo_env_to_vec_env_v1(env) # supersuit convert to vector environment
    vec_env = ss.concat_vec_envs_v1(vec_env, 4, num_cpus=0, base_class="stable_baselines3") # supersuit concatenate multiple environments -- CPUs > 1 does not seem to work in Win
    
    # check if the environment works in principle
    # check_env(env)
    # parallel_api_test(env) # check if environment works in principle
    
    
    """ perform training """
    # could be used to update the learning rate -- did not improve learning speed
    # def linear_schedule(initial_value: float) -> Callable[[float], float]:
    #     """
    #     Linear learning rate schedule.
    
    #     :param initial_value: Initial learning rate.
    #     :return: schedule that computes
    #       current learning rate depending on remaining progress
    #     """
    #     def func(progress_remaining: float) -> float:
    #         """
    #         Progress will decrease from 1 (beginning) to 0.
    
    #         :param progress_remaining:
    #         :return: current learning rate
    #         """
    #         return progress_remaining * initial_value
    
    #     return func
    
    if retrain:
    
        # proximal policy optimization model
        model = PPO(
                "MlpPolicy", # MLP
                vec_env,
                verbose=1,
                n_steps = int(2**9),  
                tensorboard_log= log_dir,
                device = "auto"
                # vf_coef = 0.5 
                # ent_coef = 0.1
                # learning_rate=linear_schedule(0.001)
            )
        # model = A2C(
        #         "MlpPolicy",
        #         vec_env,
        #         verbose=1,
        #         tensorboard_log= log_dir
        #     )
        timesteps = int(3e5) # total number of training time-steps; could be reduced - here very large to ensure that converged
        model.learn(total_timesteps = timesteps) # perform training
        
        # this can be used to start TB from PY
        # tb = program.TensorBoard()
        # tb.configure(argv=[None, '--logdir', log_dir])
        # tb.main()
        
        # plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "TD3 LunarLander")
        # plt.show()
        model_name = 'ppo_ris_' + f_name
        model.save(model_name)
        del model  # delete trained model
    
    """ test environment -- check obtained reward of agent"""
    if test:
        model_name = 'ppo_ris_' + f_name
        model = PPO.load(model_name, env=vec_env) 
        # mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
        
        # vec_env = model.get_env()
        observe, infos = env.reset(16) # initial state
        acc_rewards = np.zeros(N_OP) # accumulate rewards of each episode
        ep_rewards = np.empty(N_OP) # store episode rewards
        ep_values = np.empty(N_OP) # values of episodes
        ep_costs = np.empty(N_OP) # costs of episodes
        ep_counter = 0 # episode counter
        for i in range(1000):
            if np.mod(i,500) == 0:
                print('Iteration',i,flush = True)     
            
            obs = observe[env.possible_agents[0]]
            for agent in env.possible_agents[1:]:
                obs = np.stack([obs,observe[agent]])  
                
            temp_actions, _states = model.predict(obs, deterministic=True) # generate action
            
            actions = {agent: {} for agent in env.possible_agents}
            for ii in range(len(env.possible_agents)):
                actions[env.possible_agents[ii]] = temp_actions[ii,:]
                
            # let's overrule the action by a heuristic -- activate this to get the performance of the heuristic approach
            # for ii in range(0,len(env.possible_agents)):
            #     bet_factor = 1
            #     v_range = env.value_range[env.possible_agents[ii]] # range of values
            #     # obs[ii,0:N_RIS] = obs[ii,0:N_RIS] - env.current_value[env.possible_agents[ii]]
            #     RISs_val = v_range[0] + (v_range[1]-v_range[0])*(obs[ii,0:N_RIS]-env.delta_val)/(1-env.delta_val) # unnormalized values
            #     # interesting_RISs = RISs_val/env.improvement_value > env.price + env.acc_cost[env.possible_agents[ii]] # RISs with normalized value > current price
            #     interesting_RISs = RISs_val > env.price
            #     interesting_RISs_val = RISs_val*interesting_RISs
            #     interesting_RISs_sort_ind = interesting_RISs_val.argsort() # sorted indices of interesting RISs    
            #     mask = np.zeros(interesting_RISs.shape, bool)
            #     num_bets = int(min(np.floor(bet_factor*obs[ii,-1]/(max(obs[ii,-2],1e-6)/(1-env.delta))),N_RIS))
            #     if num_bets > 0:
            #         mask[interesting_RISs_sort_ind[-num_bets:]] = True
            #     heur_action =  interesting_RISs*mask
            #     actions[env.possible_agents[ii]] = heur_action.astype(bool)
            
            observe, rewards, dones, truncs, infos = env.step(actions) # apply action
            
            new_episode = True 
            for i in range(len(env.possible_agents)):
                acc_rewards[i] += rewards[env.possible_agents[i]]
                new_episode = np.logical_and(new_episode,dones[env.possible_agents[i]]) # check if auction has ended
                
            if new_episode:  # auction finished       
                if ep_counter == 0:
                    ep_rewards = acc_rewards
                    for no in range(N_OP):
                        ep_costs[no] = env.acc_cost[env.possible_agents[no]] # total cost of auction
                        ep_values[no] = env.current_value[env.possible_agents[no]] # achieved value/utility of allocation
                else:
                    ep_rewards = np.vstack([ep_rewards,acc_rewards])
                    temp_values = np.empty(N_OP) 
                    temp_costs = np.empty(N_OP)
                    for no in range(N_OP):
                        temp_costs[no] = env.acc_cost[env.possible_agents[no]]
                        temp_values[no] = env.current_value[env.possible_agents[no]]                        
                    ep_costs = np.vstack([ep_costs,temp_costs]) 
                    ep_values = np.vstack([ep_values,temp_values])
                acc_rewards = np.zeros(N_OP) # accumulate rewards of each episode
                ep_counter += 1
                observe, infos = env.reset(None) # reset environment
                
        # check final result: what did we pay, what did we get
        mean_reward = ep_rewards.mean(axis = 0) # mean reward
        mean_costs = ep_costs.mean(axis = 0) # mean costs
        mean_values = ep_values.mean(axis = 0) # mean utility
        pyplt.figure(1)
        ep_num = np.arange(0,ep_counter)
        pyplt.plot(ep_num,ep_rewards[:,0])
        pyplt.plot(ep_num,ep_rewards[:,1])
        
        del model  # delete trained model

        file_name = 'Store_' + f_name 
        results = [ep_rewards,ep_costs,ep_values]
        f = open(file_name,'wb')
        pickle.dump(results,f)
        f.close()
    
    """ check behavior including microscopic fading """
    if test_MF:
        model_name = 'ppo_ris_' + f_name
        model = PPO.load(model_name, env=vec_env) 
        observe, infos = env.reset(2) # reset environment
        terminated = False
        while ~terminated: # determine RIS allocation -- perform auction
            obs = observe[env.possible_agents[0]]
            for agent in env.possible_agents[1:]:
                obs = np.stack([obs,observe[agent]])  
                
            temp_actions, _states = model.predict(obs, deterministic=True)
            
            actions = {agent: {} for agent in env.possible_agents}
            for ii in range(len(env.possible_agents)):
                actions[env.possible_agents[ii]] = temp_actions[ii,:]
                
            observe, rewards, dones, truncs, infos = env.step(actions)
            
            terminated = True
            for i in range(len(env.possible_agents)):
                terminated = np.logical_and(terminated,dones[env.possible_agents[i]])
        
        # obtained RIS allocation
        RIS_alloc = np.ones((N_RIS,))*N_OP
        for no in range(N_OP):
            RIS_alloc[env.RIS_assignment[env.possible_agents[no]]] = no
        
        direct_channel_BS_UE = direct_channel(env.LOS_UE_BS,env.pow_ue_bs,env.BS_UE_assoc)    
        direct_SINR, direct_power, direct_interf = get_direct_SINR(direct_channel_BS_UE,env.BS_UE_assoc)    
        # get RIS responses
        opt_RIS_resp = RIS_alloc_response(env.RIS_UE_channel,env.RIS_BS_channel,RIS_alloc,env.BS_UE_assoc,env.np_random)    
        # get RIS-assisted channel
        RIS_channel_BS_UE = RIS_channel(env.LOS_RIS_UE,env.LOS_RIS_BS,env.RIS_BS_channel,env.channels_BS_RIS,env.RIS_UE_channel,env.channels_RIS_UE,env.BS_UE_assoc,opt_RIS_resp,env.PL_RIS_BS,env.PL_RIS_UE,RIS_alloc,env.np_random)
        # get SINR of users
        total_SINR, total_power, total_interf = get_SINR(direct_channel_BS_UE,RIS_channel_BS_UE,env.BS_UE_assoc)
        
        # ecdf_direct1 = ECDF(np.ndarray.flatten(direct_SINR[:,0,:]))
        # ecdf_direct2 = ECDF(np.ndarray.flatten(direct_SINR[:,1,:])) 
        ecdf_total1 = ECDF(np.ndarray.flatten(total_SINR[:,0,:]))
        ecdf_total2 = ECDF(np.ndarray.flatten(total_SINR[:,1,:]))
        pyplt.figure(4)
        # pyplt.plot(ecdf_direct1.x, ecdf_direct1.y,linestyle = 'solid',color = 'blue',label = 'Direct - Op1')
        # pyplt.plot(ecdf_direct2.x, ecdf_direct2.y,linestyle = 'solid',color = 'red',label = 'Direct - Op2')
        pyplt.plot(ecdf_total1.x, ecdf_total1.y,linestyle = 'solid',color = 'blue',label = 'RIS RL - Op1')
        pyplt.plot(ecdf_total2.x, ecdf_total2.y,linestyle = 'solid',color = 'red',label = 'RIS RL - Op2')
        total_SINR_RL = total_SINR
        
        # allocating everything to Operator 1
        RIS_alloc = np.ones((N_RIS,))*0
        # get RIS responses
        opt_RIS_resp = RIS_alloc_response(env.RIS_UE_channel,env.RIS_BS_channel,RIS_alloc,env.BS_UE_assoc,env.np_random)    
        # get RIS-assisted channel
        RIS_channel_BS_UE = RIS_channel(env.LOS_RIS_UE,env.LOS_RIS_BS,env.RIS_BS_channel,env.channels_BS_RIS,env.RIS_UE_channel,env.channels_RIS_UE,env.BS_UE_assoc,opt_RIS_resp,env.PL_RIS_BS,env.PL_RIS_UE,RIS_alloc,env.np_random)
        # get SINR of users
        total_SINR, total_power, total_interf = get_SINR(direct_channel_BS_UE,RIS_channel_BS_UE,env.BS_UE_assoc)
        pyplt.figure(4)
        ecdf_total1 = ECDF(np.ndarray.flatten(total_SINR[:,0,:]))
        ecdf_total2 = ECDF(np.ndarray.flatten(total_SINR[:,1,:]))
        pyplt.plot(ecdf_total1.x, ecdf_total1.y,linestyle = 'dotted',color = 'blue',label = 'RIS Op1 - Op1')
        pyplt.plot(ecdf_total2.x, ecdf_total2.y,linestyle = 'dashed',color = 'red',label = 'RIS Op1 - Op2')
        total_SINR_Op1 = total_SINR
        
        # allocating everything to Operator 2
        RIS_alloc = np.ones((N_RIS,))*1
        # get RIS responses
        opt_RIS_resp = RIS_alloc_response(env.RIS_UE_channel,env.RIS_BS_channel,RIS_alloc,env.BS_UE_assoc,env.np_random)    
        # get RIS-assisted channel
        RIS_channel_BS_UE = RIS_channel(env.LOS_RIS_UE,env.LOS_RIS_BS,env.RIS_BS_channel,env.channels_BS_RIS,env.RIS_UE_channel,env.channels_RIS_UE,env.BS_UE_assoc,opt_RIS_resp,env.PL_RIS_BS,env.PL_RIS_UE,RIS_alloc,env.np_random)
        # get SINR of users
        total_SINR, total_power, total_interf = get_SINR(direct_channel_BS_UE,RIS_channel_BS_UE,env.BS_UE_assoc)
        pyplt.figure(4)
        ecdf_total1 = ECDF(np.ndarray.flatten(total_SINR[:,0,:]))
        ecdf_total2 = ECDF(np.ndarray.flatten(total_SINR[:,1,:]))
        pyplt.plot(ecdf_total1.x, ecdf_total1.y,linestyle = 'dashed',color = 'blue',label = 'RIS Op2 - Op1')
        pyplt.plot(ecdf_total2.x, ecdf_total2.y,linestyle = 'dotted',color = 'red',label = 'RIS Op2 - Op2')
        pyplt.legend()
        pyplt.xlabel('Instantaneous SINR [dB]')
        pyplt.ylabel('Empirical cumulative distribution')
        pyplt.xlim([-10,45])
        pyplt.grid(visible = True)
        total_SINR_Op2 = total_SINR
        
        file_name = 'Store_MF_' + f_name 
        results = [total_SINR_RL,total_SINR_Op1,total_SINR_Op2,direct_SINR]
        f = open(file_name,'wb')
        pickle.dump(results,f)
        f.close()


end_time = time.time()





