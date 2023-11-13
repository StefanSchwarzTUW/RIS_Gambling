"""
Frequently Used Function for Wireless Environment
"""

""" uniform placement of nodes within a region of interest """
def uniform_placement(ndim,npos,ROI,rng): # number of dimensions, number of positions, ROI borders, random number generator 
    rand_pos = rng.uniform(ROI[0],ROI[1],size=(npos, ndim))
    return rand_pos

""" regular placement within a region of interest with noise on top to observe different positions"""
def regular_noisy_placement(ndim,npos,ROI,rng,noise_var): # number of dimensions (not tested for more than 2), number of positions, ROI borders, random number generator, variance of Gaussian perturbation
    import math
    import numpy as np
    
    dim_div = math.ceil(npos**(1/ndim)) # number of positions per dimension
    dim_dist = (ROI[1]-ROI[0])/dim_div # distance between regular positions
    reg_grid = np.linspace(ROI[0]+dim_dist/2,ROI[1]-dim_dist/2,dim_div) # grid of positions
    n_pos = dim_div**ndim # total number of possible positions
    possible_pos = np.array(np.meshgrid(reg_grid[:,0],reg_grid[:,1])).T.reshape(-1,2) # all possible positions
    pos_choice = rng.choice(n_pos, npos, replace=False) # pick required number of positions randomly
    chosen_pos = possible_pos[pos_choice,:]
    rand_pos = chosen_pos + rng.normal(0,np.sqrt(noise_var),chosen_pos.shape) # randomly perturb position
    return rand_pos

""" pathloss model with LOS/NLOS pathloss coefficient """            
def uma_pathloss(distances,rng,lam,fix_los,sf): # distances between TX and RX, random number generator, wavelength, fix LOS/NLOS conditions, shadow fading variance
    import numpy as np
    
    LOS_prob = np.exp(-distances/50) # distance-dependent LOS probability
    if fix_los == 1: # LOS-condition fixed to NLOS
        LOS_prob = 0
    elif fix_los == 2:
        LOS_prob = 1 # LOS-condition fixed to LOS

    LOS_decision = rng.uniform(0,1,distances.shape) < LOS_prob  # LOS decision
    
    d0 = 2.5 # reference distance of path loss model (should be frequency dependent)
    gamma = [2,3.75] # path loss exponent for LOS and NLOS
    PL0 = 10*np.log10((4*np.pi*d0/lam)**2)  # path loss at reference distance
    PL = PL0 + 10*np.log10(distances/d0)*(gamma[0]*LOS_decision + gamma[1]*(1-LOS_decision)) + rng.normal(0,np.sqrt(sf),LOS_decision.shape) #10*np.log10(rng.lognormal(0,np.sqrt(6),LOS_decision.shape)) 
    return PL, LOS_decision            

""" array steering vector of uniform circular array (UCA) """    
def uca_response(M,R,lam,phi): # antenna number, radius of array, wavelength, angle of arrival
    import numpy as np
    
    m_vec = np.reshape(np.array(range(M)),(M,1,1,1))  # antenna elements
    phi = np.repeat(phi[np.newaxis,:,:,:],M,axis = 0) # angle of arrival
    array_response = np.exp(-1j*2*np.pi/lam*R*np.cos(phi-m_vec*2*np.pi/M))
    return array_response

""" complex Gaussian channel """    
def Gauss_channel(array_size,rng): # number of antennas, random number generator
    import numpy as np
    
    channels = 1/np.sqrt(2)*(np.array(rng.normal(0,1,array_size))+1j*np.array(rng.normal(0,1,array_size)))
    return channels

""" optimized RIS response """
def opt_RIS_response(RIS_UE,RIS_BS): # RIS-to-UE and RIS-to-BS channels
    import numpy as np
        
    RIS_BS_phase = np.angle(RIS_BS)
    RIS_UE_phase = np.angle(RIS_UE)
    N_BS = RIS_BS_phase.shape[2] # number of base stations
    N_OP = RIS_BS_phase.shape[3] # number of operators
    N_UE = RIS_UE_phase.shape[2] # number of users
    N_RIS = RIS_BS_phase.shape[1] # number of RISs
    M = RIS_BS_phase.shape[0] # number of discrete elements per RIS
    opt_RIS_resp = np.zeros((M,N_RIS,N_UE,N_BS,N_OP)) + 1j*np.zeros((M,N_RIS,N_UE,N_BS,N_OP))
    for no in range(N_OP):
        for nu in range(N_UE):
            for nr in range(N_RIS):
                temp_phase = np.reshape(RIS_UE_phase[:,nr,nu,no],(M,1))
                opt_RIS_resp[:,nr,nu,:,no] = np.exp(-1j*(RIS_BS_phase[:,nr,:,no] + np.repeat(temp_phase,N_BS,axis = 1))) # compensate phase-shift by RIS response
    return opt_RIS_resp

""" optimized RIS response consider RIS allocation amongst Operators """
def RIS_alloc_response(RIS_UE,RIS_BS,RIS_alloc,BS_UE_assoc,rng):
    import numpy as np
        
    RIS_BS_phase = np.angle(RIS_BS)
    RIS_UE_phase = np.angle(RIS_UE)
    N_BS = RIS_BS_phase.shape[2] # number of base stations
    N_OP = RIS_BS_phase.shape[3] # number of operators
    N_UE = RIS_UE_phase.shape[2] # number of users
    N_RIS = RIS_BS_phase.shape[1] # number of RISs
    M = RIS_BS_phase.shape[0] # number of discrete elements per RIS
    opt_RIS_resp = np.zeros((M,N_RIS,N_UE,N_OP)) + 1j*np.zeros((M,N_RIS,N_UE,N_OP))
    for no in range(N_OP):
        for nu in range(N_UE):
            for nr in range(N_RIS):
                if RIS_alloc[nr] == no: # RIS is controlled by this OP and UE is served by this BS
                    temp_phase = np.reshape(RIS_UE_phase[:,nr,nu,no],(M,))
                    opt_RIS_resp[:,nr,nu,no] = np.exp(-1j*(RIS_BS_phase[:,nr,BS_UE_assoc[nu,no],no] + temp_phase)) # RIS response is optimized for associated BS
                # else:
                #     opt_RIS_resp[:,nr,nu,:,no] = np.exp(-1j*2*np.pi*rng.uniform(0,1,size = (M,N_BS)))
    return opt_RIS_resp