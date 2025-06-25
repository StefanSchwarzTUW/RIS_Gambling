# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 09:33:52 2023

@author: Stefan Schwarz

This is the environment for multi-agent training

"""

import functools

import gymnasium # make sure to install gymnasium in your environment
from gymnasium import spaces

from pettingzoo import ParallelEnv
# from pettingzoo.utils import wrappers

# import math
import numpy as np

from RIS_Competition_Geometric import gen_POS, direct_PL, RIS_PL, RIS_channel_vectors, est_values_pos


# import matplotlib.pyplot as plt



# def env(render_mode=None):
#     """
#     The env function often wraps the environment in wrappers by default.
#     You can find full documentation for these methods
#     elsewhere in the developer documentation.
#     """
#     internal_render_mode = render_mode if render_mode != "ansi" else "human"
#     env = raw_env(render_mode=internal_render_mode)
#     # This wrapper is only for environments which print results to the terminal
#     if render_mode == "ansi":
#         env = wrappers.CaptureStdoutWrapper(env)
#     # this wrapper helps error handling for discrete action spaces
#     env = wrappers.AssertOutOfBoundsWrapper(env)
#     # Provides a wide vareity of helpful user errors
#     # Strongly recommended
#     env = wrappers.OrderEnforcingWrapper(env)
#     return env


# def raw_env(render_mode=None):
#     """
#     To support the AEC API, the raw_env() function just uses the from_parallel
#     function to convert from a ParallelEnv to an AEC env
#     """
#     env = parallel_env(render_mode=render_mode)
#     env = parallel_to_aec(env)
#     return env


class parallel_env(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "RIS_env_v3"}

    # def __init__(self,N_OP,N_UE,N_RIS,N_BS,M_RIS,ROI_size,budget,render_mode=None):
    def __init__(self,render_mode=None,**kwargs):
        """
        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        - render_mode

        Note: as of v1.18.1, the action_spaces and observation_spaces attributes are deprecated.
        Spaces should be defined in the action_space() and observation_space() methods.
        If these methods are not overridden, spaces will be inferred from self.observation_spaces/action_spaces, raising a warning.

        These attributes should not be changed after initialization.
        """
                
        # static system parameters
        self.fc = 26e9 # carrier frequency
        self.lam = 3e8/self.fc # wavelength
        self.sf = 10 # shadow fading variance 
        self.M_BS = 1 # number of antennas at Bs -- currently only single antenna support 
        self.M_UE = 1 # number of antennas at UE -- currently only single antenna supported
        self.NN = 1 # loop over random realizations -- currently not used
        self.util_alpha = 1 # alpha value of alpha-fair utility function
        self.improvement_value = 1 # 100% (improvement_value) improvement over no RIS is worth my entire budget c_V in paper
        
        # input parameters of the environment
        self.possible_agents = ["Op" + str(r) for r in range(kwargs['N_OP'])]
        self.K = kwargs['K'] # K factor (linear) for LOS and NLOS
        self.N0 = kwargs['N0'] # noise PSD dBm/Hz
        self.F = kwargs['F'] # noise figure in dB
        self.Bs = kwargs['Bs'] # subcarrier bandwidth
        self.Ps = kwargs['Ps'] # power per subcarrier in dBm
        self.ps_lin = kwargs['ps_lin'] # linear power
        self.sigma_n2 = kwargs['sigma_n2'] # noise power (linear)
        self.N_OP = kwargs['N_OP'] # number of OPs
        self.N_UE = kwargs['N_UE'] # number of UEs
        self.N_RIS = kwargs['N_RIS'] # number of RISs
        self.N_BS = kwargs['N_BS'] # number of BSs
        self.M_RIS = kwargs['M_RIS'] # number of RIS elements
        self.ROI_size = kwargs['ROI_size'] # size of ROI
        self.max_budget = {self.possible_agents[i]: kwargs["budget"][i] for i in range(kwargs['N_OP'])} # available budget for bidding (can be different for operators)
        
        self.delta = 0 # this is used as a distance between continuous ranges and "do not use" indicators        
        self.delta_val = 0
        self.start_price = 0 # price at beginning of auction
        self.increment = 0.05 # price increment from one round to the next        

        self.render_mode = render_mode        
        
        # the value_range should be chosen to approximately cover the expected range of observed values when varying the environment randomly
        # this requires some trial and error as it depends on the environment and antenna numbers
        # values below are useful for simulations in the paper
        self.improvement_max = {agents: 0.25*np.log10(self.M_RIS) for agents in self.possible_agents} # maximal considered possible improvement
        self.improvement_min = {agents: 0 for agents in self.possible_agents} # don't bid if expected improvement is below this
        self.value_range = {agents: np.array([self.improvement_min[agents],self.improvement_max[agents]]) for agents in self.possible_agents}
        
        self.acc_cost = {agents: 0 for agents in self.possible_agents} # store accumulated costs


    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        """ 
        Observation space consisting of normalized value of each RIS, normalized price and remaining budget
        """
        obs_space = spaces.Box(0,1, shape = (self.N_RIS + 2,)) # utilities of RIS, current price, remaining budget (all normalized)
        return obs_space 
        # return self.observation_spaces[agent]

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        space_dim = np.ones(self.N_RIS)*2 # accept the price for a RIS or not
        return spaces.MultiDiscrete(space_dim) 
        # return self.action_spaces[agent]

    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return
        else:
            string = ""
            for no in range(self.N_OP):
                string = np.char.add(string," OP{}: {}".format(no, self.state[self.agents[no]]))

        print(string)

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def reset(self,seed, options=None):
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.
        Here it initializes the `num_moves` variable which counts the number of
        hands that are played.
        Returns the observations for each agent
        """
        self.np_random = np.random.default_rng(seed) # random number generator
        self.agents = self.possible_agents[:]
        self.num_moves = 0
        self.price = self.start_price
        self.assigned_RIS = np.zeros(self.N_RIS,dtype = bool)
        self.RIS_assignment = {agent: np.zeros(self.N_RIS) for agent in self.agents}
        self.prior_RIS_assignment = {agent: np.zeros(self.N_RIS) for agent in self.agents} # used to check if RIS-assignment changed, such that values need to be updated
        self.budget = self.max_budget.copy() # currently available budget -- updated throughout the auction
        self.acc_cost = {agents: 0 for agents in self.possible_agents}
        
        self.punish_factor = self.improvement_max[self.agents[0]]*10 # factor used to punish invalid actions
        
        # reset environment geometry
        obs_values = self.reset_geometry()
        
        observations = {agent: np.append(np.append(obs_values[agent],self.price/self.budget[agent]*(1-self.delta)),1) for agent in self.agents}
        
        self.current_value = {agent: 0 for agent in self.agents} # initialize relative value achieved so far
                
        infos = {agent: {} for agent in self.agents}

        
        self.state = observations
                
        
        return observations, infos

    def step(self, actions):
        """
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - terminations
        - truncations
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """
        
        self.num_moves += 1       

        truncations = {agent: False for agent in self.agents}
        

        # punish invalid bids (on already assigned RISs and RISs not bid before -- activity rule)
        punish = {agent: False for agent in self.agents}
        for i in range(len(self.agents)):
            temp_action = actions[self.agents[i]]
            temp_action = temp_action.astype(bool)
            unavail_RIS = self.state[self.agents[i]][0:self.N_RIS] <= 0 # RISs with value 0 are not avail
            punish[self.agents[i]] = -sum(unavail_RIS[temp_action])*self.price*self.punish_factor/self.max_budget[self.agents[i]]
            temp_action[unavail_RIS] = False # these bids are ignored
            actions[self.agents[i]] = temp_action
            
        
        # see if there is a winner
        bid_temp = np.array(list(actions.values())) 
        bid_array = bid_temp #+ self.np_random.uniform(low = -1e-4,high = 1e-4,size = bid_temp.shape)
        auction_winner_bool = np.logical_and(~self.assigned_RIS,np.sum(bid_array,axis = 0) == 1) # if there is just one bidder --> winner
        auction_winner_ind = np.argmax(bid_array,axis = 0) # index of agent that bid        
        self.assigned_RIS = np.logical_or(self.assigned_RIS,auction_winner_bool) # RISs that are already assigned
        
        # update budget and rewards of agents
        rewards = {agent: {} for agent in self.agents}
        temp_assignment = np.zeros(self.N_RIS,dtype = bool) # perform a consistency check
        for i in range(len(self.agents)):
            self.prior_RIS_assignment[self.agents[i]] = self.RIS_assignment[self.agents[i]] # store prior RIS assignment
            won_ris = np.logical_and(auction_winner_ind == i,auction_winner_bool) # RISs won by operator            
            self.RIS_assignment[self.agents[i]] = np.logical_or(self.RIS_assignment[self.agents[i]],won_ris) # new RIS assignment
            temp_assignment = np.logical_or(temp_assignment,self.RIS_assignment[self.agents[i]])
            
            if sum(won_ris):
                costs = sum(won_ris)*self.price/self.max_budget[self.agents[i]] # relative costs we have to pay
                self.acc_cost[self.agents[i]] += costs # stores costs accumulated over auction
         
                # value of current assignment
                RIS_alloc = np.zeros((1,self.N_RIS))
                RIS_alloc[:] = self.RIS_assignment[self.agents[i]]
                # get the value of the current allocation
                sinrs, value_won_RIS = est_values_pos(False,self.M_RIS,self.M_UE,self.M_BS,self.ps_lin,self.sigma_n2,self.N_UE,self.N_BS,self.N_RIS,self.pow_ue_bs[:,:,i],self.pow_ris_bs[:,:,i],self.pow_ris_ue[:,:,i],self.util_alpha,self.BS_UE_assoc[:,i],RIS_alloc,self.IBI[:,:,:,i],self.LOS_UE_BS[:,:,i],self.LOS_RIS_BS[:,:,i],self.LOS_RIS_UE[:,:,i],self.K)
                value_won_RIS = (value_won_RIS/self.base_values[i]-1)/self.improvement_value # relative value
                value_won_RIS = value_won_RIS.item() # total value of RIS_alloc, not added value of currently won RIS!
                prior_value = self.current_value[self.agents[i]] # prior total value of RIS _alloc
                self.current_value[self.agents[i]] = value_won_RIS # update for next round
                value_won_RIS = value_won_RIS - prior_value # relative added value 
                
                if costs <= self.budget[self.agents[i]]/self.max_budget[self.agents[i]]: 
                    rewards[self.agents[i]] = value_won_RIS - costs # relative reward when budget is not overshot
                else:
                    affordable_number = np.floor(self.budget[self.agents[i]]/self.price) # affordable number of RISs
                    unaffordable_number = sum(won_ris) - affordable_number # number of RISs won too much
                    rewards[self.agents[i]] = value_won_RIS - (affordable_number*self.price + unaffordable_number*self.price*self.punish_factor)/self.max_budget[self.agents[i]]
            else: # nothing won - no costs, no reward
                costs = 0
                rewards[self.agents[i]] = 0
            rewards[self.agents[i]] += punish[self.agents[i]] # apply punishment for invalid bidding
            self.budget[self.agents[i]] = max(self.budget[self.agents[i]] - costs*self.max_budget[self.agents[i]],0) # absolute remaining budget cannot go below 0 
            
        if sum(temp_assignment != self.assigned_RIS): raise Exception("Something inconsistent with RIS assignment")
        
        # update values of remaining RISs
        obs_values = self.update_value(actions)
        
        # increase the price for the next round
        self.price += self.increment 
        
        # update observations
        observations = {agent: {} for agent in self.agents}
        for i in range(len(self.agents)):
            temp_value = obs_values[self.agents[i]]
            RIS_values = temp_value 
            norm_price = self.price/self.max_budget[self.agents[i]]*(1-self.delta) # normalized price
            if norm_price > (1-self.delta): norm_price = 1 # price higher than budget 
            observations[self.agents[i]] = np.append(RIS_values,norm_price)
            
            norm_budget = self.budget[self.agents[i]]/self.max_budget[self.agents[i]] # remaining budget (normalized)
            observations[self.agents[i]] = np.append(observations[self.agents[i]],norm_budget)
            
            # perform a consistency check -- already assigned RISs should not be used
            if sum(RIS_values[self.assigned_RIS] > 0): raise Exception("Something inconsistent with available RIS") 
        
        self.state = observations

        # If the current price is too high for an agent, the agent is out of the game
        # If the agent has no free RISs left, it is also out of the game
        terminations = {agent: {} for agent in self.agents}
        remove = np.zeros(len(self.agents), dtype=bool)
        for i in range(len(self.agents)):
            remove[i] = self.price > self.budget[self.agents[i]] # remove if price is too high
            remove[i] = remove[i] or np.sum(observations[self.agents[i]][0:self.N_RIS] > 0) == 0 # remove if no RISs available for this agent
            terminations[self.agents[i]] = remove[i]

        if self.render_mode == "human":
            self.render()            

        infos = {agent: {} for agent in self.agents}
        for agent in self.agents:
            infos[agent]["terminal_observation"] = False 
        infos['Price'] = self.price
        infos['Assignment'] = self.RIS_assignment
        
            
        return observations, rewards, terminations, truncations, infos
    
    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        # return current observation of this agent
        return np.array(self.observations[agent])
    

        
    def reset_geometry(self):
        show_plot = False
        UE_pos, BS_pos, RIS_pos = gen_POS(show_plot,self.N_OP,self.N_BS,self.N_UE,self.N_RIS,self.ROI_size,self.np_random) # generate positions
        self.PL_UE_BS, self.LOS_UE_BS, angles_UE_BS = direct_PL(show_plot,self.N_UE,self.N_BS,self.N_OP,UE_pos, BS_pos,self.np_random,self.lam,self.sf) # get pathloss of direct links
        # PL_UE_BS = np.ones(PL_UE_BS.shape)*1e3
        self.PL_RIS_BS, self.LOS_RIS_BS, angles_RIS_BS, self.PL_RIS_UE, self.LOS_RIS_UE, angles_RIS_UE = RIS_PL(show_plot,self.N_RIS,self.N_BS,self.N_OP,self.N_UE,UE_pos, BS_pos, RIS_pos,self.np_random,self.lam,self.sf) # get pathloss of RIS links
        # calculate channel gains
        self.pow_ue_bs = 10**(-self.PL_UE_BS/10)
        self.pow_ris_bs = 10**(-self.PL_RIS_BS/10)
        self.pow_ris_ue = np.swapaxes(10**(-self.PL_RIS_UE/10),0,1)
        # associate users with strongest BSs
        self.BS_UE_assoc = np.argmax(self.pow_ue_bs,axis = 1)
        # get RIS channel vectors -- needed for interference estimation of directional links
        self.channels_BS_RIS, self.channels_RIS_UE, self.RIS_BS_channel, self.RIS_UE_channel = RIS_channel_vectors(self.M_RIS,self.M_BS,self.M_UE,self.N_UE,self.N_OP,self.N_BS,self.N_RIS,self.NN,self.np_random,self.lam,angles_RIS_BS,angles_RIS_UE)
        # calculate inter-BS-interference level at RISs due to similarity of array response vectors
        self.IBI = np.zeros((self.N_BS,self.N_BS,self.N_RIS,self.N_OP))
        for no in range(self.N_OP):
            for nb1 in range(self.N_BS):
                for nb2 in range(self.N_BS):
                    for nr in range(self.N_RIS):
                        self.IBI[nb1,nb2,nr,no] = (np.abs(np.conj(self.RIS_BS_channel[:,nr,nb1,no]) @ self.RIS_BS_channel[:,nr,nb2,no]))  
        # base values when no RISs are assigned
        self.base_values = np.zeros((self.N_OP,))
        for no in range(self.N_OP):
            SINRs, value = est_values_pos(show_plot,self.M_RIS,self.M_UE,self.M_BS,self.ps_lin,self.sigma_n2,self.N_UE,self.N_BS,self.N_RIS,self.pow_ue_bs[:,:,no],self.pow_ris_bs[:,:,no],self.pow_ris_ue[:,:,no],self.util_alpha,self.BS_UE_assoc[:,no],np.zeros((1,self.N_RIS)),self.IBI[:,:,:,no],self.LOS_UE_BS[:,:,no],self.LOS_RIS_BS[:,:,no],self.LOS_RIS_UE[:,:,no],self.K)
            self.base_values[no] = value
                        
        est_SINRs = []
        est_values = []
        current_alloc = np.zeros((self.N_OP,self.N_RIS),dtype = int)    
        for no in range(self.N_OP): # estimate performance of certain RIS allocations (not all since this would be too much many)
            # currently allocated RISs for this operator -- updated during auction
            current_alloc_ind = np.nonzero(current_alloc[no,:])
            # add one RIS to the current allocation and see how it improves the value
            columns_to_set = np.arange(self.N_RIS)
            columns_to_set = np.setxor1d(columns_to_set,current_alloc_ind)        
            N_remain = self.N_RIS - sum(current_alloc[no,:])
            RIS_allocs = np.zeros((N_remain,self.N_RIS))
            RIS_allocs[:,columns_to_set] = np.identity(N_remain)        
            RIS_allocs[:,current_alloc_ind] = 1   
            # poss_RIS_allocs.append(RIS_allocs)             
            SINRs, value = est_values_pos(show_plot,self.M_RIS,self.M_UE,self.M_BS,self.ps_lin,self.sigma_n2,self.N_UE,self.N_BS,self.N_RIS,self.pow_ue_bs[:,:,no],self.pow_ris_bs[:,:,no],self.pow_ris_ue[:,:,no],self.util_alpha,self.BS_UE_assoc[:,no],RIS_allocs,self.IBI[:,:,:,no],self.LOS_UE_BS[:,:,no],self.LOS_RIS_BS[:,:,no],self.LOS_RIS_UE[:,:,no],self.K)
            est_SINRs.append(SINRs)
            est_values.append(value) # value in terms of alpha-exponentiated sum-rate
            
        obs_values = {agent: {} for agent in self.agents} # values mapped to the observation range self.delta to 1
        for no in range(self.N_OP):
            valid_ind = est_values[no] >= self.base_values[no] # values that improve over the current state
            if np.any(valid_ind):
                est_values[no] = ((est_values[no]/self.base_values[no]-1)/self.improvement_value) # relative value
                value_range_min = self.value_range[self.agents[no]][0]
                value_range_max = self.value_range[self.agents[no]][1]
                est_values[no] = self.delta_val + (1-self.delta_val)/(value_range_max-value_range_min)*(est_values[no] - value_range_min) # normalized value
                est_values[no][est_values[no] > 1] = 1 # restrict to box-range
                est_values[no][est_values[no] < self.delta_val] = 0 # value too low to bid
                obs_values[self.agents[no]] = est_values[no]
            else: # no improvement
                obs_values[self.agents[no]] = np.zeros(est_values[no].shape) 
                   
        return obs_values

    def update_value(self,actions):
        est_SINRs = []
        est_values = []
        poss_RIS_allocs = []
        for no in range(self.N_OP): # estimate performance of certain RIS allocations (not all since this would be too many)
            # if the RIS allocation changed and available RISs are still left, we have to update values
            SINRs = []
            value = []
            new_RIS_allocs = []
            if sum(self.RIS_assignment[self.agents[no]] != self.prior_RIS_assignment[self.agents[no]]) and sum(~self.assigned_RIS):
                current_alloc = self.RIS_assignment[self.agents[no]] # currently allocated RISs -- values are only required for the remaining
                current_alloc_ind = np.nonzero(current_alloc) # indices of assigned RISs
                impossible_RISs = self.assigned_RIS # already assigned RISs
                impossible_RISs = np.logical_or(impossible_RISs,~actions[self.agents[no]]) # RISs that the agent did not bid on
                impossible_RIS_inds = np.nonzero(impossible_RISs)
                # add one RIS to the current allocation and see how it improves the value
                columns_to_set = np.arange(self.N_RIS)
                columns_to_set = np.setxor1d(columns_to_set,impossible_RIS_inds)        
                if np.any(columns_to_set):
                    N_remain = len(columns_to_set) # number of possible RIS allocations
                    RIS_allocs = np.zeros((N_remain,self.N_RIS))
                    new_RIS_allocs = np.zeros((N_remain,self.N_RIS),dtype = bool)
                    new_RIS_allocs[:,columns_to_set] = np.identity(N_remain)
                    RIS_allocs = np.copy(new_RIS_allocs)        
                    RIS_allocs[:,current_alloc_ind] = 1   
                    new_RIS_allocs = np.sum(new_RIS_allocs,axis = 0,dtype = bool)
                    # poss_RIS_allocs.append(RIS_allocs)             
                    SINRs, value = est_values_pos(False,self.M_RIS,self.M_UE,self.M_BS,self.ps_lin,self.sigma_n2,self.N_UE,self.N_BS,self.N_RIS,self.pow_ue_bs[:,:,no],self.pow_ris_bs[:,:,no],self.pow_ris_ue[:,:,no],self.util_alpha,self.BS_UE_assoc[:,no],RIS_allocs,self.IBI[:,:,:,no],self.LOS_UE_BS[:,:,no],self.LOS_RIS_BS[:,:,no],self.LOS_RIS_UE[:,:,no],self.K)
            est_SINRs.append(SINRs)
            est_values.append(value) # value in terms of alpha-exponentiated sum-rate
            poss_RIS_allocs.append(new_RIS_allocs) # possible RIS allocations for this operator
        
        # calculate normalized values
        obs_values = {agent: np.zeros((self.N_RIS,)) for agent in self.agents} # values mapped to the observation range [0,1]
        for no in range(self.N_OP):
            if sum(self.RIS_assignment[self.agents[no]] != self.prior_RIS_assignment[self.agents[no]]): # if no change -- reuse prior values
                if np.any(est_values[no]): # if we have values to set
                    # self.base_values[no] = est_values[no][0] # value of current allocation
                    valid_ind = est_values[no] > self.base_values[no] # values that improve over no RIS
                    if np.any(valid_ind): # if new values improve over current state
                        est_values_tmp = (est_values[no]/self.base_values[no]-1)/self.improvement_value # percentage values
                        est_values_tmp = est_values_tmp - self.current_value[self.agents[no]] # additional value compared to current allocation
                        value_range_min = self.value_range[self.agents[no]][0]
                        value_range_max = self.value_range[self.agents[no]][1]
                        est_values_tmp = self.delta_val + (1-self.delta_val)/(value_range_max-value_range_min)*(est_values_tmp - value_range_min) # normalized value
                        est_values_tmp[est_values_tmp > 1] = 1 # restrict to box-range
                        est_values_tmp[est_values_tmp < self.delta_val] = 0                    
                        obs_values[self.agents[no]][poss_RIS_allocs[no]] = est_values_tmp # normalize value to box-range [0,1]             
            else:
                obs_values[self.agents[no]] = self.state[self.agents[no]][0:self.N_RIS]
                impossible_RISs = self.assigned_RIS # already assigned RISs
                impossible_RISs = np.logical_or(impossible_RISs,~actions[self.agents[no]],dtype = bool) # RISs that the agent did not bid on
                obs_values[self.agents[no]][impossible_RISs] = 0 # RISs that are not available any more
        
       
        return obs_values
     