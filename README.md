# RIS_Gambling
Gambling on Reconfigurable Intelligent Surfaces  
Please cite "Gambling on Reconfigurable Intelligent Surfaces", Stefan Schwarz, IEEE Communications Letters, 2024  

Main file is RIS_bidding_geometric.py  
Auction environment is defined in RIS_env_geometric_v2.py  
Utility/value estimation is in RIS_Competition_Geometric.py
Wireles specific functions (channels, SINRs, etc) are in wireless_fuf.py

A trained agent for 3 base stations, 2 operators, 10 users, 20 RISs and 1000 reconfigurable elements per RIS is provided in ppo_ris_3BS_10UE_20RIS_1000M.zip

Run main file to train agent; afterwards, testing in main file can be activated
