#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 10:54:36 2022

@author: dekens
"""
import numpy as np
import tools_fig_7_to_9 as tools


G = [0.5, 1.05, 1.3, 1.4] #for m = 0.2
m = 0.2

for g in G:
    ######## Working parameters
    
    parameters = 0.05, 1., 1., g, m, 1.
    epsilon, r, kappa, g, m, theta = parameters
    
    
    date ='20230927'
    title = "fig_7_to_9"
    subtitle = 'm =%4.2f'%m+'_g =%4.2f'%g+'_theta =%4.1f'%theta+'_epsilon = %4.2f'%epsilon
    
    ######## Creating directory
    
    workdir = date + '_' + title
    subworkdir = workdir +'/'+subtitle
    tools.create_directory(workdir)
    tools.create_directory(subworkdir)
    
    ######## Time discretization
    
    Tmax, Nt = 50, 10000
    T, dt = np.linspace(0, Tmax, Nt, retstep=True)
    parameters_time = T, Nt, dt
    ######## Trait space discretization (note that a larger within-family variance requires larger maximal traits)
    
    zmin, zmax, Nz =  -2.5, 1.2, 551
    z, dz = np.linspace(zmin, zmax, Nz, retstep=True)
    parameters_trait_space = zmin, zmax, z, Nz, dz
    ######## Initial state
    
    N1star, N2star, zstar = tools.analytical_asym_eq_hab_2(parameters) ### given by Proposition 4.2 of Dekens 2022
    n1initial = N1star*tools.Gauss(zstar, epsilon, z)
    n2initial = N2star*tools.Gauss(zstar, epsilon, z)
    
    moments_initial_1 = tools.moments(n1initial, z, dz)
    moments_initial_2 = tools.moments(n2initial, z, dz)
        
    
    ### To generate figure 7
    tools.phase_lines_all_speed(parameters, parameters_trait_space, subworkdir)

    
    ###  To generate figure 8: first run numerical resolutions, next plot
    C =np.linspace(0., 2, num =10) ## set environmental speeds
    if (g>1.2):
        C =np.linspace(0., 3, num =10) ### explore larger speeds to capture critical speed for strongest selection
    ######## Run numerical resolutions of Eq. 7 for fig 4
    tools.run_model(n1initial, n2initial, C, parameters, parameters_time, parameters_trait_space, subworkdir)  
    ######## Plot results
    tools.plot_pop_size_and_mean_at_eq(C, parameters, parameters_trait_space, subworkdir)
    
        
    
#### To generate figure 9 
Gbis = np.linspace(1/5*(1+2*m), 2, 200) ### intermediate selection vector
tools.plot_critical_speed_th(Gbis, parameters, parameters_trait_space, workdir)



