#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 10:54:36 2022

@author: dekens
"""
import numpy as np
import tools_figure_6 as tools



m=0.5 #### Migration rate used


######## Creating directory
date ='20241001'
title = "figure_6"
subtitle = 'm =%4.2f'%m
workdir = date + '_' + title
tools.create_directory(workdir)

######## Trait space discretization
    
zmin, zmax, Nz =  -2.5, 1.2, 5510
z, dz = np.linspace(zmin, zmax, Nz, retstep=True)
parameters_trait_space = zmin, zmax, z, Nz, dz

#### To generate figure 6  
G = np.linspace(1/5*(1+2*m), 2, 200) ### intermediate selection vector
tools.plot_critical_speed_th(G, m, parameters_trait_space, workdir)

