#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 12:35:55 2023

@author: dekens
"""
import tools_fig_6 as tools
import numpy as np

date = '20230927'
title = 'fig_6'
path = date + '_' + title
tools.create_directory(path, remove = False)

G = [0.7, 1.1, 1.4, 1.8] #### selection strengths

Ngen_burnin, dt = 100, 0.01
Nreplicates = [250, 1000, 1000, 1000] ### Nreplicates can vary for g in G
Ngen_sim = [350000, 200000, 100000, 50000] ### the number of generations can vary for g in G
eps, tau, theta1, theta2, K, m = np.sqrt(5e-3), 1, -1, 1, 1e4, .5  #eps, tau, theta1, theta2, K, m

count_g =0
for g in G:
    subpath = path + '/g=%4.1f'%g+'_eps=%4.2f'%eps+'_Ngen_sim=%i'%Ngen_sim[count_g]+'_dt=%4.2f'%dt+'_tau=%4.1f'%tau+'_theta1=%4.1f'%theta1+'_theta2=%4.1f'%theta2+'_K=%i'%K+'_m=%4.1f'%m
    tools.create_directory(subpath, remove = False)
    C =np.linspace(0., 2, num =10) #### vector of environmental speeds
    if (g==1.4):
        C = np.sort(np.append(C, 2 - 2/9*3/2)) ### add a speed for this parameter g
    if (g==1.8):
        C =np.linspace(0., 3, num =10)  ### larger speed range for this parameter g to capture the critical speed for persistence
    ### loop to run all the replicates for each c in C
    for c in C:
        parameters = eps, tau, g, theta1, theta2, K, m, c  #eps, tau, g, theta1, theta2, K, m, c
        parameters_time = Ngen_sim[count_g], Ngen_burnin, dt
        subpath_c = subpath + '/c=%4.2f'%c
        tools.create_directory(subpath_c, remove = False)
        #### function that runs rhe replicate IBS
        tools.run_replicate_IBS(Nreplicates[count_g], parameters, parameters_time, subpath_c)
    #### Plot the comparison between the end results of the IBS and the analytical prediction for equilibria
    tools.plot_end_size_per_c_comparison_analytical(g, C, parameters, parameters_time, subpath)
    count_g = count_g +1
