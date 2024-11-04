#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 10:54:36 2022

@author: dekens
"""
import numpy as np
import tools_figures_2_to_6 as tools



G = [0.7, 1.1, 1.4, 1.8]  #### Vector of selection strengths used
m=0.5 #### Migration rate used

for g in G:
    ######## Working parameters
    
    parameters = 0.05, 1., 1., g, m, 1.
    epsilon, r, kappa, g, m, theta = parameters
    
    
    date ='20241001'
    title = "figure_5"
    subtitle = 'm =%4.2f'%m+'_g =%4.2f'%g+'_theta =%4.1f'%theta+'_epsilon = %4.2f'%epsilon
    
    ######## Creating directory
    
    workdir = date + '_' + title
    subworkdir = workdir +'/'+subtitle
    tools.create_directory(workdir)
    tools.create_directory(subworkdir)
    
    ######## Trait space discretization
    
    zmin, zmax, Nz =  -2.5, 1.2, 551
    z, dz = np.linspace(zmin, zmax, Nz, retstep=True)
    parameters_trait_space = zmin, zmax, z, Nz, dz
    
        

    
    ###  To generate figure 5, one need to first run numerical resolutions of S_eps for increasing environmental speeds c
    C =np.linspace(0., 2, num =10) #### Set environemental change speed
    if (g==1.4):
        C = np.append(C, 2 - 2/9*3/2)  ### add a data point to capture interesting phenomenon
    if (g==1.8):
        C = np.linspace(0., 3, num =10) ### explore larger speeds to capture critical speed for strongest selection
    
       
    ######## Run numerical resolutions of S_eps
    ######## Time discretization
    
    Tmax, Nt = 50, 10000
    T, dt = np.linspace(0, Tmax, Nt, retstep=True)
    parameters_time = T, Nt, dt
    
    ######## Initial state: subpop sizes and mean trait given by Dekens 2022 (specialist adapted to the native habitat), local trait distributions are Gaussian initially
    
    N1star, N2star, zstar = tools.analytical_asym_eq_hab_2(parameters) ### given by Proposition 4.2 of Dekens 2022
    n1initial = N1star*tools.Gauss(zstar, epsilon, z)
    n2initial = N2star*tools.Gauss(zstar, epsilon, z)
    
    ### Actually run the num resolutions of S_eps
    tools.run_model(n1initial, n2initial, C, parameters, parameters_time, parameters_trait_space, subworkdir)

    
    ######## Generate fig 5
    tools.plot_pop_size_and_mean_at_eq(C, parameters, parameters_trait_space, subworkdir)
    
     

