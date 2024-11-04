 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 08:41:32 2022

@author: dekens
"""
import numpy as np
from scipy import sparse as sp
import scipy.sparse.linalg as scisplin
import scipy.signal as scsign
import os
import matplotlib as ml
import matplotlib.pyplot as plt
import multiprocessing
from itertools import repeat
from matplotlib import cm

ml.rcParams['mathtext.fontset'] = 'stix'
ml.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({
    "text.usetex": True})
ml.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #for \text command
plasma = cm.get_cmap('plasma', 300)
viridis = cm.get_cmap('viridis', 300)
inferno = cm.get_cmap('inferno', 300)


####### Create a directory of path workdir
def create_directory(workdir):
    try:
        os.mkdir(workdir)
        print("Directory " , workdir ,  " Created ") 
    except FileExistsError:
        print("Directory " , workdir ,  " already exists")
    return



################## Graphic function for the phase lines of the dominant trait


#### Auxiliary functions
def search_positive_roots(P): ## P a polynomial given by its list of coefficients
    roots = np.roots(P)
    roots = roots[np.real(np.isreal(roots))]
    roots = roots[roots>0]
    return(roots)

### Function giving the selection gradient
def F(z, rho, g):
    return(2*g*((rho**2-1)/(rho**2+1) - z))

### auxiliary function to surch the roots of a polynomial
def find_rho_star(z, parameters):
    g, m = parameters
    P = [1, 1/m - 1 - g/m*(z + 1)**2, -1/m + 1 + g/m*(z -1)**2, -1]
    roots = search_positive_roots(P)
    return(roots)

### Compute the quantities from the phase-line analysis of the selection gradient
def curve_per_parameter(parameters, parameters_trait_space): 
    g, m = parameters
    zmin, zmax, Z, Nz, dz = parameters_trait_space
    rho_star, z_star = np.empty(3*Nz), np.empty(3*Nz)
    index_current = 0
    for k in range(Nz):
        z = Z[k]
        roots = find_rho_star(z, parameters)
        index_past = index_current
        index_current = index_past + np.size(roots)
        rho_star[index_past:index_current] = roots
        z_star[index_past:index_current] = z*np.ones(np.size(roots))
    z_star = z_star[:index_current]
    rho_star = rho_star[:index_current]
    return(z_star, rho_star, np.array(list(map(F, z_star, rho_star, repeat(g))))) 
            
    

    

#### Function to build figure 6

def plot_critical_speed_th(G, m, parameters_trait_space, subworkdir):
    
    Ng = len(G)
    
    #### Initiating the different vectors that will store the values corresponding to different critical speeds and criticial thresholds for different selection strengths
    
    zswitch = np.zeros(Ng)
    z_DV = np.zeros(Ng)
    
    c_crit_switch = np.zeros(Ng)
    c_DP = np.zeros(Ng)
    c_DV = np.zeros(Ng)
    
    c_crit_persistence_P_e = np.zeros(Ng)
    c_crit_persistence_S0 = np.zeros(Ng)
    
    
    #### Loop on the vector of selection strengths to compute the different critial speeds
    for j in range(Ng):
        g = G[j]
        parameters = g, m
        z_star, rho_star, F_z_star = curve_per_parameter(parameters, parameters_trait_space) ### compute mean trait at eq, ratio at eq
        
        #### Computing the critical speed for the habitat switch c_crit_switch, and the corresponding threshold on the mean trait zswitch
        c_crit_switch[j] = max(np.array(F_z_star)[z_star>=0])
        zstarbis = z_star[[z_star>=0]]
        zswitch[j] = zstarbis[np.argmax(np.array(F_z_star)[z_star>=0])]
        
        #### Computing the critical speed for the death plain c_DP, and the corresponding threshold on the mean trait z_DP

        Delta = 4/g**2*(m**2-4*g*(m-1)) ### Requires to compute a discriminant
        z_DP = np.sqrt(1/2*(2*(1+g-m)/g + np.sqrt(Delta)))
        c_DP[j] = -2*g*z_DP*(2/(z_DP**2+1+(m-1)/g) - 1)
        
        if (g>=1):
            ### if g >1, the death valley exists and the follogin
            z_DV[j] = np.sqrt(1/2*(2*(1+g-m)/g - np.sqrt(Delta))) 
            c_DV[j] = 2*g*z_DV[j]*(2/(z_DV[j]**2+1+(m-1)/g) - 1 ) 
            
            ### if the switch occurs with the death valley
            if (zswitch[j]<=z_DV[j]):
                c_crit_persistence_S0[j] = c_DV[j]
                if (c_crit_switch[j] > c_DP[j]):
                    c_crit_persistence_P_e[j] = c_DV[j]  
                else: 
                    c_crit_persistence_P_e[j] = c_DP[j]
            else:
                c_crit_persistence_S0[j] = c_crit_switch[j]
                if (c_crit_switch[j] > c_DP[j]):
                    c_crit_persistence_P_e[j] = c_crit_switch[j]  
                else:
                    c_crit_persistence_P_e[j] = c_DP[j]
        else:#### if g<1, the death valley does not exist, and the critical rate is c_DP (death plain)
            c_crit_persistence_P_e[j] = c_DP[j]
            c_crit_persistence_S0[j] = c_DP[j]
            
            
    ##### Produce the figure     
    
    plt.figure(figsize=(15,13))
    plt.xlabel('Selection intensity $g$', fontsize = 50)
    plt.ylabel('Env. speed $c$', fontsize = 50)
    plt.xticks(fontsize = 35)
    plt.yticks(fontsize = 35)
    
    
    ##### Display the critical speed for habitat switch
    plt.plot(G, c_crit_switch, color ='Royalblue', linewidth = 5, label = 'Habitat switch')
    ##### Display the critical speed for the death plain
    plt.plot(G, c_DP, color ='darkred', linewidth = 5, label = 'Death Plain')
    ##### Display the critical speed for the death valley when it exists (G>=1). I distinguish whether sqrt(z_DV) is above or below the switch to have a smoother junction
    plt.plot(G[(z_DV<zswitch)&(G>=1)], c_DV[(z_DV<zswitch)&(G>=1)], color ='chocolate', linewidth = 5, label = r'Death Valley')
    plt.plot(G[(z_DV>zswitch)&(G>=1)], c_DV[(z_DV>zswitch)&(G>=1)], color ='chocolate', linewidth = 5 )
    
    ##### Display the critical speed for persistence according to P_eps
    plt.plot(G, c_crit_persistence_P_e, color ='black', linewidth = 6., label = r'Extinction - $P_\varepsilon$')
    ##### Display the critical speed for persistence according to S_0
    plt.plot(G, c_crit_persistence_S0, color ='black', linewidth = 6., label = 'Extinction - $S_0$', linestyle = 'dashed')
    ##### Display the critical speed for persistence for single habita models (m = 0)
    c_crit_1_habitat = 2*np.sqrt(G)
    plt.plot(G, c_crit_1_habitat, color ='grey', linewidth = 6., label = 'Extinction - $m=0$', linestyle = 'dashdot')
    
    
    #### Add vertical lines delineating the parameter region relating to the 4 different cases displayed in fig3^
    plt.vlines(1, 0, 3, linestyle = 'dashed', color ='black' )
    Gbis = G[G>1]
    plt.vlines(Gbis[np.argmin(c_crit_switch[G>1] - c_DV[G>1])], 0, 3, linestyle = 'dashed', color ='black' )
    plt.vlines(G[c_DP > c_crit_switch][-1], 0, 3, linestyle = 'dashed', color ='black' )

    plt.legend(fontsize = 30, loc = 'upper left')
    plt.savefig(subworkdir + '/critical_speed_persistence_th.png', bbox_inches='tight')

    plt.show()
    plt.close()
    