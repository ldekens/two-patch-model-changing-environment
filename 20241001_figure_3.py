#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 10:54:36 2022

@author: dekens
"""
import numpy as np
import tools_figure_3 as tools



G = [0.7, 1.1, 1.4, 1.8]  #### Vector of selection strengths used
m=0.5 #### Migration rate used in fig 2-6


for g in G:
    ######## Working parameters
    
    parameters = 1., 1., g, m, 1.
    r, kappa, g, m, theta = parameters
    
    
    date ='20241001'
    title = "figure_3"
    subtitle = 'm =%4.2f'%m+'_g =%4.2f'%g
    
    ######## Creating directory
    
    workdir = date + '_' + title
    subworkdir = workdir +'/'+subtitle
    tools.create_directory(workdir)
    tools.create_directory(subworkdir)
    
    ######## Trait space discretization
    
    zmin, zmax, Nz =  -2.5, 1.2, 551
    z, dz = np.linspace(zmin, zmax, Nz, retstep=True)
    parameters_trait_space = zmin, zmax, z, Nz, dz
    
   
    ### To generate figure 3
    tools.phase_lines_all_speed(parameters, parameters_trait_space, subworkdir)

