import numpy as np
# import gymnasium as gym
# import matplotlib.pyplot as plt
# from statsmodels.distributions.empirical_distribution import ECDF
from itertools import product
# from itertools import product
#import math 
# import multiprocessing

# from numba import jit

import sys # import from frequently used functions

from wireless_fuf import uniform_placement, regular_noisy_placement, uma_pathloss,  Gauss_channel,  uca_response #, RIS_alloc_response
# from parfor import parfor
#import random


# helper function to generate bit vectors corresponding to RIS allocations
def generate_bit_vectors(length):
    if length <= 0:
        return []
    elif length > 20:
        print('Many combinations; should we really proceed?')
        input("Press Enter to continue...")
    possible_values = [False, True]
    all_bit_vectors = np.array(list(product(possible_values, repeat=length)))
    return all_bit_vectors


######################################################################## 
""" Positions """
########################################################################
def gen_POS(show_plot,N_OP,N_BS,N_UE,N_RIS,ROI_size,rng1):  

    # generate random user positions for each operator
    UE_pos = []
    # plt.figure(0)
    for no in range(N_OP):
        UE_pos.append(uniform_placement(2,N_UE,ROI_size,rng1))
        # plt.scatter(UE_pos[no][:,0],UE_pos[no][:,1])
    UE_pos = np.array(UE_pos)
        
    # generate random base station positions for each operator
    pos_noise_var = ((np.amax(ROI_size) - np.amin(ROI_size))/5)**2  
    BS_pos = []
    # plt.figure(1)
    for no in range(N_OP):
        BS_pos.append(regular_noisy_placement(2,N_BS,ROI_size,rng1,pos_noise_var))
        # plt.scatter(BS_pos[no][:,0],BS_pos[no][:,1])  
        
    # generate random RIS positions 
    RIS_pos = []    
    pos_noise_var = ((np.amax(ROI_size) - np.amin(ROI_size))/20)**2
    RIS_pos = regular_noisy_placement(2,N_RIS,ROI_size,rng1,25)
    # if show_plot:
    #     plt.figure(2)
    #     plt.scatter(RIS_pos[:,0],RIS_pos[:,1])
    
    return UE_pos, BS_pos, RIS_pos

######################################################################## 
""" Direct path-loss """
########################################################################
def direct_PL(show_plot,N_UE,N_BS,N_OP,UE_pos, BS_pos,rng1,lam,sf):
    # get pathloss and angles between BS and UE
    PL_UE_BS = np.empty((N_UE,N_BS,N_OP))
    LOS_UE_BS = np.empty((N_UE,N_BS,N_OP),dtype = int)
    angles_UE_BS = np.empty((N_UE,N_BS,N_OP))
    for no in range(N_OP):
        ue_pos = UE_pos[no]
        bs_pos = BS_pos[no]
        distances = np.empty((N_UE,N_BS))
        for nu in range(N_UE):
            for nb in range(N_BS):
                pos_diff = ue_pos[nu,:] - bs_pos[nb,:]
                distances[nu,nb] = np.sqrt(np.sum(np.abs(pos_diff))**2)
                angles_UE_BS[nu,nb,no] = np.arctan2(pos_diff[1],pos_diff[0])
        result = uma_pathloss(distances,rng1,lam,1,sf)   
        PL_UE_BS[:,:,no] = result[0]
        LOS_UE_BS[:,:,no] = result[1]
        # if show_plot:
        #     plt.figure(3)
        #     plt.scatter(distances,PL_UE_BS[:,:,no])
        
    return PL_UE_BS, LOS_UE_BS, angles_UE_BS
    
######################################################################## 
""" RIS path-loss """
########################################################################
def RIS_PL(show_plot,N_RIS,N_BS,N_OP,N_UE,UE_pos, BS_pos, RIS_pos,rng1,lam,sf):
    # get pathloss and angles between BS and RIS
    PL_RIS_BS = np.empty((N_RIS,N_BS,N_OP))
    LOS_RIS_BS = np.empty((N_RIS,N_BS,N_OP),dtype = int)
    angles_RIS_BS = np.empty((N_RIS,N_BS,N_OP))
    for no in range(N_OP):    
        ris_pos = RIS_pos
        bs_pos = BS_pos[no]
        distances = np.empty((N_RIS,N_BS))
        for nr in range(N_RIS):
            for nb in range(N_BS):
                pos_diff = ris_pos[nr,:] - bs_pos[nb,:]
                distances[nr,nb] = np.sqrt(np.sum(np.abs(pos_diff)**2))
                angles_RIS_BS[nr,nb,no] = np.arctan2(pos_diff[1],pos_diff[0])
        result = uma_pathloss(distances,rng1,lam,2,sf) # forced to LOS
        PL_RIS_BS[:,:,no] = result[0]
        LOS_RIS_BS[:,:,no] = result[1]
    # if show_plot:
    #     plt.figure(3)
    #     plt.scatter(distances,PL_RIS_BS[:,:,no])
       
    # get pathloss and angles between UE and RIS
    PL_RIS_UE = np.empty((N_RIS,N_UE,N_OP))
    LOS_RIS_UE = np.empty((N_RIS,N_UE,N_OP),dtype = int)
    angles_RIS_UE = np.empty((N_RIS,N_UE,N_OP))
    for no in range(N_OP):    
        ris_pos = RIS_pos
        ue_pos = UE_pos[no]
        distances = np.empty((N_RIS,N_UE))
        for nr in range(N_RIS):
            for nu in range(N_UE):
                pos_diff = ris_pos[nr,:] - ue_pos[nu,:]
                distances[nr,nu] = np.sqrt(np.sum(np.abs(pos_diff)**2))
                angles_RIS_UE[nr,nu,no] = np.arctan2(pos_diff[1],pos_diff[0])
        result = uma_pathloss(distances,rng1,lam,0,sf)
        PL_RIS_UE[:,:,no] = result[0]
        LOS_RIS_UE[:,:,no] = result[1]
    # if show_plot:
    #     plt.figure(3)
    #     plt.scatter(distances,PL_RIS_UE[:,:,no])
        
    return PL_RIS_BS, LOS_RIS_BS, angles_RIS_BS, PL_RIS_UE, LOS_RIS_UE, angles_RIS_UE


######################################################################## 
""" Estimate values of an operator only for certain RIS allocations accounting for positions and K-factors"""    
########################################################################
# @jit(nopython=True)
def est_values_pos(show_plot,M_RIS,M_UE,M_BS,ps_lin,sigma_n2,N_UE,N_BS,N_RIS,pow_ue_bs = np.array([[]]),pow_ris_bs=np.array([[]]),pow_ris_ue=np.array([[]]),util_alpha = int,BS_UE_assoc=np.array([]),RIS_allocs=np.array([[]]),IBI=np.array([[[]]]),LOS_UE_BS=np.array([[]]),LOS_RIS_BS=np.array([[]]),LOS_RIS_UE=np.array([[]]),K=np.array([[]])):
    SINR_vals = np.zeros((N_UE,RIS_allocs.shape[0]))

    kk_direct = K[1-LOS_UE_BS]
    pow_direct_Gauss = ps_lin*pow_ue_bs*(1/(1+kk_direct)) # received power over Gauss direct channels
    # magnitude of direct signals over directional channels (needed for coherent combination with RIS parts)
    mag_direct_dir = np.sqrt(pow_ue_bs)*np.sqrt(kk_direct/(1+kk_direct))
  
    # directional signals
    for rr in range(RIS_allocs.shape[0]): # loop over all considered RIS allocations
        ris_alloc = RIS_allocs[rr,:] # current RIS allocation
        
        # incoherent signals from unassigned RISs -- this includes also signals over Gauss channels
        ris_pow_incoh = M_RIS*pow_ris_bs # power enhancement for incoherent Gauss channels
        pow_ris_incoh = np.zeros((N_UE,N_BS))  
        for nu in range(N_UE): # no need for a K-factor here, since Gauss and directional channels behave the same for incoherent RISs
            ris_ue = pow_ris_ue[nu,:]*(1-ris_alloc)
            pow_ris_incoh[nu,:] = ps_lin*ris_ue @ ris_pow_incoh # received power over incoherent RIS channels
            
        # incoherent signals from assigned RISs -- these are received over Gauss channels
        ris_pow_Gauss = M_RIS*pow_ris_bs # power enhancement for incoherent Gauss channels
        pow_ris_Gauss = np.zeros((N_UE,N_BS))  
        for nu in range(N_UE): # here we need a K-factor to account for relative strength compared to coherent signals
            kk_ue = K[1-LOS_RIS_UE[:,nu]] # K-factor between UE and RIS    
            ris_ue = (1/(1+kk_ue))*pow_ris_ue[nu,:]*ris_alloc
            pow_ris_Gauss[nu,:] = ps_lin*ris_ue @ ris_pow_Gauss # received power over incoherent RIS channels

        # coherent signals from assigned RISs    
        # SINR_temp = np.zeros((N_UE,))
        for nu in range(N_UE):
            kk_ue = K[1-LOS_RIS_UE[:,nu]] # K-factor between UE and RIS
            ris_mag_coh = M_RIS*np.sqrt(pow_ris_bs[:,BS_UE_assoc[nu]]) # coherent combination increases magnitude by M
            for_range = [ris_ind for ris_ind in range(N_RIS) if ris_alloc[ris_ind] > 0]
            mag_ris_intended = np.zeros((len(for_range),))
            nrc = 0
            for nr in for_range: # coherent intended signals from RISs
                mag_ris_intended[nrc] = np.sqrt((kk_ue[nr]/(1+kk_ue[nr])))*np.sqrt(pow_ris_ue[nu,nr])*ris_mag_coh[nr]
                nrc += 1
                
            interf_ind = np.concatenate((np.arange(0, BS_UE_assoc[nu]), np.arange(BS_UE_assoc[nu]+1, N_BS)),dtype = np.int32) # interfering BSs
            mag_ris_interf = np.zeros((len(for_range),len(interf_ind)))
            nbc = 0
            for nb in interf_ind: # coherent interfering signals from RISs
                nrc = 0
                for nr in for_range:
                    mag_ris_interf[nrc,nbc] = np.sqrt((kk_ue[nr]/(1+kk_ue[nr])))*np.sqrt(pow_ris_ue[nu,nr])*np.sqrt(pow_ris_bs[nr,nb])*IBI[BS_UE_assoc[nu],nb,nr]
                    nrc += 1
                nbc += 1
                    
            power_coherent = ps_lin*(sum(mag_ris_intended) + mag_direct_dir[nu,BS_UE_assoc[nu]])**2 
            interf_coherent = ps_lin*sum(np.sum(mag_ris_interf**2,axis = 0) + (mag_direct_dir[nu,interf_ind])**2)
            power = power_coherent + pow_ris_incoh[nu,BS_UE_assoc[nu]] + pow_direct_Gauss[nu,BS_UE_assoc[nu]] + pow_ris_Gauss[nu,BS_UE_assoc[nu]]
            interference = interf_coherent + sum(pow_ris_incoh[nu,interf_ind]) + sum(pow_direct_Gauss[nu,interf_ind]) + sum(pow_ris_Gauss[nu,interf_ind]) 
            SINR_vals[nu,rr] = power/(interference + sigma_n2)    
                
    rates = np.log2(1+SINR_vals)  
    value = sum(rates**(1/util_alpha))
    return SINR_vals, value


######################################################################## 
""" Get fading channel vectors of RIS links """    
########################################################################
def RIS_channel_vectors(M_RIS,M_BS,M_UE,N_UE,N_OP,N_BS,N_RIS,NN,rng1,lam,angles_RIS_BS,angles_RIS_UE):
    # get RIS channels
    channels_BS_RIS = Gauss_channel((M_RIS,M_BS,N_BS,N_RIS,N_OP,NN),rng1) # Gauss part
    channels_RIS_UE = Gauss_channel((M_UE,M_RIS,N_RIS,N_UE,N_OP,NN),rng1) # Gauss part
    RIS_BS_channel = uca_response(M_RIS,(M_RIS*lam/2)/(2*np.pi),lam,angles_RIS_BS) # directional part
    RIS_UE_channel = uca_response(M_RIS,(M_RIS*lam/2)/(2*np.pi),lam,angles_RIS_UE) # directional part
    
    return channels_BS_RIS, channels_RIS_UE, RIS_BS_channel, RIS_UE_channel

