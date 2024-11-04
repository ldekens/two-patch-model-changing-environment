#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 10:54:36 2022

@author: dekens
"""
import numpy as np
import tools_figure_2 as tools



g =0.7  #### Selection strengths used in fig 2
m=0.5 #### Migration rate used in fig 2-6


######## Working parameters

parameters = 1., 1., g, m, 1.
r, kappa, g, m, theta = parameters


date ='20241001'
title = "figure_2"
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
    
##### Plots illustrating the methods (fig2).

C_fig2 = [0] #### environmental change speed
tools.phase_lines_dominant_trait(C_fig2, parameters, parameters_trait_space, subworkdir) ## plot Fig.2a

C_fig2 = [0, 0.25] #### environmental change speed
tools.phase_lines_dominant_trait(C_fig2, parameters, parameters_trait_space, subworkdir)## plot Fig.2b

C_fig2 = [0, 0.25, .5] #### environmental change speed
tools.phase_lines_dominant_trait(C_fig2, parameters, parameters_trait_space, subworkdir) ## plot Fig.2c


