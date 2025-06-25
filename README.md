# RIS_Gambling
This code implements the techniques described in:

"Gambling on Reconfigurable Intelligent Surfaces", Stefan Schwarz, IEEE Communications Letters, 2024, Volume: 28, Issue: 4, DOI: 10.1109/LCOMM.2024.3360477  

Please cite this paper when using this code.

Main file is RIS_bidding_geometric.py  
Auction environment is defined in RIS_env_geometric_v2.py  
Utility/value estimation is in RIS_Competition_Geometric.py
Wireles specific functions (channels, SINRs, etc) are in wireless_fuf.py

A trained agent for 3 base stations, 2 operators, 10 users, 20 RISs and 1000 reconfigurable elements per RIS is provided in ppo_ris_3BS_10UE_20RIS_1000M.zip

Run main file RIS_bidding_geometric.py to test this agent
You may train new agents for other settings by activating "retrain" in RIS_bidding_geometric.py

This project is licensed under the Creative Commons Attribution 4.0 International License (CC BY-NC 4.0). See the LICENSE file for details.
