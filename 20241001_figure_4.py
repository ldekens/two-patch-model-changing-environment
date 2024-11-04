#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 10:54:36 2022

@author: dekens
"""
import numpy as np
import tools_figure_4 as tools


G = [0.7, 1.1, 1.4, 1.8]  #### Vector of selection strengths used in fig 2 - 6
m=0.5 #### Migration rate used in fig 2-6

T_max_plot_fig4 = [5, 5, 5, 2]##### Vector of max time for fig4 (zoom on early time dynamics)
count= 0 ## Just an incremental count

for g in G:
    ######## Working parameters
    
    parameters = 0.05, 1., 1., g, m, 1.
    epsilon, r, kappa, g, m, theta = parameters
    
    ######## Creating directory
    date ='20241001'
    title = "figure_4"
    subtitle = 'm =%4.2f'%m+'_g =%4.2f'%g+'_theta =%4.1f'%theta+'_epsilon = %4.2f'%epsilon
    workdir = date + '_' + title
    subworkdir = workdir +'/'+subtitle
    tools.create_directory(workdir)
    tools.create_directory(subworkdir)
    
    
    ######## Trait space discretization
    
    zmin, zmax, Nz =  -2.5, 1.2, 551
    z, dz = np.linspace(zmin, zmax, Nz, retstep=True)
    parameters_trait_space = zmin, zmax, z, Nz, dz
    
    
    #######  Vector of environmental speeds
    C =np.linspace(0., 2, num =4) #### Set environemental change speed
    if (g==1.4):
        C = np.append(C, 2 - 2/9*3/2)  ### add a data point to capture interesting phenomenon
    if (g==1.8):
        C = np.linspace(0., 3, num =4) ### explore larger speeds to capture critical speed for strongest selection
    
    ##################### Run numerical resolutions of S_0
    
    ######## Time discretization
    Tmax, Nt = T_max_plot_fig4[count], 10000
    T, dt = np.linspace(0, Tmax, Nt, retstep=True)
    parameters_time = T, Nt, dt
    #### Initial state of the limit model S_0 (Two subpop sizes and mean trait)
    N10, N20, z0 = tools.analytical_asym_eq_hab_2(parameters) ### given by Proposition 4.2 of Dekens 2022
    ### Creating a separate subdirectory to store the temporal dynamics given by S_0
    subtitle_limit = 'm =%4.2f'%m+'_g =%4.2f'%g+'_theta =%4.1f'%theta
    subworkdir_limit = workdir +'/'+subtitle_limit
    tools.create_directory(subworkdir_limit)
    ### Actually run the num. resolutions of S_0
    tools.run_limit_model(z0, N10, N20, C, parameters[1:], parameters_time, subworkdir_limit) ### No need for the first entry of parameters, which is epsilon
    
    
    ##################### Run numerical resolutions of S_eps
    ### The time parameters have to be refined for the comparison to be relevant
    Tmax_eps, Nt_eps = T_max_plot_fig4[count], int(np.floor(T_max_plot_fig4[count]*50/(epsilon**2)))   #### with eps>0, the discretization in time needs to be quite fine to capture the early transient dyanmics accurately
    T_eps, dt_eps = np.linspace(0, Tmax_eps, Nt_eps, retstep=True)
    parameters_time_eps = T_eps, Nt_eps, dt_eps
    
    ######## Initial state: local trait distributions (gaussian), subpop sizes and mean trait same as for S_0
    
    n1initial = N10*tools.Gauss(z0, epsilon, z)
    n2initial = N20*tools.Gauss(z0, epsilon, z)
    
    ### Actually run the num resolutions of S_eps
    tools.run_model(n1initial, n2initial, C, parameters, parameters_time_eps, parameters_trait_space, subworkdir)



    ################### Finally, plot the temporal trajectories of both systems S_eps and S_0    
    tools.plot_comparison_limit_model_with_model_positive_eps(C, epsilon, T_max_plot_fig4[count], parameters[1:], parameters_time, parameters_time_eps, subworkdir_limit, subworkdir)
    count = count +1
    
    
    
        
