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
    r, kappa, g, m, theta = parameters
    P = [1, 1/m - 1 - g/m*(z + theta)**2, -1/m + 1 + g/m*(z -theta)**2, -1]
    roots = search_positive_roots(P)
    return(roots)

### Compute the quantities from the phase-line analysis of the selection gradient
def curve_per_parameter(parameters, parameters_trait_space): 
    r, kappa, g, m, theta = parameters
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
            
    
    
##### The next function cumputes a vector of booleans that encodes the theoretical viability of the fast equilibrium at Z (upon uniqueness, so caveat for the range of 3 equilibria)
def viability_fast_equlibria_theoric(parameters, Z):
    r, kappa, g, m, theta = parameters
    
    Delta = 4/g**2*(m**2-4*g*(m-1))
    z1 = 1/2*(2*(1+g-m)/g - np.sqrt(Delta))
    z2 = 1/2*(2*(1+g-m)/g + np.sqrt(Delta)) 
    if (g>=1):
        if (m < 2*g*(1-np.sqrt(1-1/g))):
            viability = ((z1<Z**2)&(Z**2<z2))
        else:
            viability = (Z<Z)
    else:
        if (m <= (1-g)/2):
           viability = ((Z**2<(np.sqrt((1-m)/g)-1)**2)|(z1<Z**2)&(Z**2<z2))
        else:
            if (m < (1-g)):
                viability = (Z**2 < max((np.sqrt((1-m)/g)-1)**2, z2) )
            else:
                viability = (Z**2 < z2)
    return(viability)

    
    

##### Function that displays the figure 3: long and tedious function with lots of graphic details

def phase_lines_all_speed(parameters, parameters_trait_space, subworkdir):
    zmin, zmax, Z, Nz, dz = parameters_trait_space
    plt.figure(figsize=(12,9))
    plt.xlabel('$Z$', fontsize = 40)
    plt.ylabel(r'$\mathcal{F}^{c=0}(Z)$', fontsize = 40)
    plt.xticks(fontsize = 30)
    plt.yticks(fontsize = 30)
    
    
    ### Compute and plot the theoretical equilibrium
    z_star, rho_star, F_z_star = curve_per_parameter(parameters, parameters_trait_space)
    plt.vlines(z_star[np.argmax([F_z_star<0])], -0.05, 0.05, color ='black')
    plt.vlines(0, -0.05, 0.05, linewidth =2,  color ='black')
    plt.text(-0.07, 0.1, '0', fontsize = 30)
    viability = viability_fast_equlibria_theoric(parameters, z_star)
    r, kappa, g, m, theta = parameters
    colors = [inferno(50), 'RoyalBlue', 'darkred', 'chocolate']
    count = 0
    plt.scatter(z_star[- np.argmax([F_z_star[::-1]>0])], 0, s=200., color = colors[count])

    plt.text(z_star[-np.argmax([F_z_star[::-1]>0])], 0.15, r'$ Z^*_{\text{spec}}$', fontsize = 30, color = inferno(0))
    plt.text(z_star[np.argmax([F_z_star<0])]-0.1, 0.15, r'$ -Z^*_{\text{spec}}$', fontsize = 30, color = inferno(0))
    plt.plot(z_star[viability&(z_star>=0)], F_z_star[viability&(z_star>=0)], linewidth = 3, color = colors[count])
    plt.plot(z_star[viability&(z_star<=0)], F_z_star[viability&(z_star<=0)], linewidth = 3, color = colors[count])
    plt.plot(z_star[((~viability)&(z_star>-1))], F_z_star[((~viability)&(z_star>-1))], linewidth = 3, linestyle = 'dotted', color = colors[count])
    plt.plot(z_star[((~viability)&(z_star<-1))], F_z_star[((~viability)&(z_star<-1))], linewidth = 3, linestyle = 'dotted', color = colors[count])
    
    #### Compute and plot the switch
    zstarbis = z_star[z_star>=0]
    z_max = zstarbis[np.argmax(np.array(F_z_star)[z_star>=0])]
    F_max = np.max(F_z_star[z_star>=0])
    z_post_switch = z_star[- np.argmax([F_z_star[::-1]>F_max])]
    count = count +1
    plt.scatter(z_post_switch, 0, s=200., color = colors[count])
    plt.text(z_post_switch-0.5, 0+ 0.2, r'$ Z^*_{\text{switch}}$', fontsize = 30, color = colors[count])
    plt.text(z_post_switch+0.3, F_max+ 0.05, r'Habitat switch', fontsize = 30, color = colors[count])

    plt.vlines(z_post_switch, -0.7, 2.5, linewidth = 3, color = colors[count])
    plt.text(z_max + 0.2, F_max, r'$\boldsymbol{c_{\text{switch}}}$', color = colors[count] ,fontsize = 30)
    plt.hlines(F_max, z_max - 0.2, z_max + 0.2, color = colors[count], linewidth =5)
    plt.arrow(z_max, F_max, z_post_switch-z_max, 0, color = colors[count],width = 0.025, length_includes_head=True)
    
    ### Overlay a graphical representation of the selection functions
    plt.plot(Z[(Z-1)**2<1.4], -g*(Z[(Z-1)**2<1.4]-1)**2, linewidth = 8, alpha = 0.2, color = viridis(200))
    plt.plot(Z[(Z+1)**2<1.4], -g*(Z[(Z+1)**2<1.4]+1)**2, linewidth = 8, alpha = 0.2, color = viridis(70))
    plt.plot(Z, np.zeros(np.size(Z)), color='black', linewidth=2.5)
    
    #### Compute and plot the limit of viability ie crticial speeds  death plain and valley
    F_dp = np.min(F_z_star[(~viability)&(z_star<=-1)])
    z_dp = z_star[np.argmin([F_z_star[(z_star<=-1)]>=F_dp])]
    count = count +1
    if (g!=1.4):
        plt.text(z_dp+0.16, F_dp, r'$\boldsymbol{c_{\text{death plain}}}$', color = colors[count] ,fontsize = 29)
    plt.hlines(F_dp, z_dp - 0.15, z_dp + 0.15, color = colors[count], linewidth =5)
    if (g>1):
        F_dp = np.max(F_z_star[(~viability)&(z_star>=0)])
        z_dp = z_star[z_star>=0][np.argmax(F_z_star[z_star>=0]>F_dp)]
        count = count +1
        Delta = 4/g**2*(m**2-4*g*(m-1))
        z_dv = np.sqrt(1/2*(2*(1+g-m)/g - np.sqrt(Delta)))
        c_dv = 2*g*z_dv*(-1 + 2/(z_dv**2+1+(m-1)/g) )
        z_dp = -np.sqrt(1/2*(2*(1+g-m)/g + np.sqrt(Delta)))
        plt.vlines(z_dv, 0, c_dv, color = colors[count], linewidth =2)
        if (g!=1.4):
            plt.text(z_dv+0.25, c_dv-0.04, r'$\boldsymbol{c_{\text{death valley}}}$', color = colors[count] ,fontsize = 29)
        if (g==1.4):
            plt.text(z_dv+0.25, c_dv-0.08, r'$\boldsymbol{c_{\text{death valley}}}$', color = colors[count] ,fontsize = 29)

        plt.hlines(c_dv, z_dv - 0.15, z_dv + 0.15, color = colors[count], linewidth =5)
    plt.plot(Z, np.zeros(np.size(Z)), color='black', linewidth=2.5)
    
    #### The following section is to build the visual grey area corresponding to the death plain and death valley (tailored to each set of parameters)
    Delta = 4/g**2*(m**2-4*g*(m-1))
    z_dp = -np.sqrt(1/2*(2*(1+g-m)/g + np.sqrt(Delta)))
    if (g==1.1):
        plt.vlines(z_dp, -1.4, 2.7, linewidth = 5, color = 'black')
        plt.text(z_dp-.6, 3.1, s= r'DEATH', fontsize = 30)
        plt.text(z_dp-.6, 2.8, s= r'PLAIN', fontsize = 30)
        plt.fill_betweenx(y=[-1.4, 2.7], x1= zmin, x2=z_dp, color = 'black', alpha = 0.3)
    if (g<1):
        plt.vlines(z_dp, -1.8, 2.1, linewidth = 5, color = 'black')
        plt.text(z_dp-.6, 2.37, s= r'DEATH', fontsize = 30)
        plt.text(z_dp-.6, 2.17, s= r'PLAIN', fontsize = 30)
        plt.fill_betweenx(y=[-1.8, 2.1], x1= zmin, x2=z_dp, color = 'black', alpha = 0.3)
    if (g==1.1):
        z_dv = np.sqrt(1/2*(2*(1+g-m)/g - np.sqrt(Delta)))
        c_dv = 2*g*z_dv*(-1 + 2/(z_dv**2+1+(m-1)/g) )
        plt.vlines(z_dv, -1.4, 2.7, linewidth = 5, color = 'black')
        plt.vlines(-z_dv, -1.4, 2.7, linewidth = 5, color = 'black')
        plt.text(-z_dv-0.2, 3.1, s= r'DEATH', fontsize = 30)
        plt.text(-z_dv-0.22, 2.8, s= r'VALLEY', fontsize = 30)
        plt.fill_betweenx(y=[-1.4, 2.7], x1=-z_dv, x2=z_dv, color = 'black', alpha = 0.3)
    if (g==1.4):
        plt.vlines(z_dp, -3, 4.3, linewidth = 5, color = 'black')
        plt.text(z_dp-.7, 3.9, s= r'DEATH', fontsize = 30)
        plt.text(z_dp-.7, 3.5, s= r'PLAIN', fontsize = 30)
        plt.fill_betweenx(y=[-3, 4.3], x1= zmin, x2=z_dp, color = 'black', alpha = 0.3)
    
    if (g==1.4):
        F_dp = np.min(F_z_star[(~viability)&(z_star<=-1)])
        z_dp = z_star[- np.argmax([F_z_star[::-1]>=F_dp])]
   
        plt.text(z_dp-0.4, F_dp+0.2, r'$\boldsymbol{c_{\text{death plain}}}$', color = 'darkred' ,fontsize = 29)
        plt.hlines(F_dp, z_dp - 0.15, z_dp + 0.15, color = 'darkred', linewidth =5)
        z_dv = np.sqrt(1/2*(2*(1+g-m)/g - np.sqrt(Delta)))
        c_dv = 2*g*z_dv*(-1 + 2/(z_dv**2+1+(m-1)/g) )
        plt.vlines(z_dv, -3, 4.3, linewidth = 5, color = 'black')
        plt.vlines(-z_dv, -3, 4.3, linewidth = 5, color = 'black')
        plt.text(-z_dv, 3.9, s= r'DEATH', fontsize = 30)
        plt.text(-z_dv, 3.5, s= r'VALLEY', fontsize = 30)
        plt.fill_betweenx(y=[-3, 4.3], x1=-z_dv, x2=z_dv, color = 'black', alpha = 0.3)
    if (g==1.8):
        plt.vlines(z_dp, -2.8, 5.4, linewidth = 5, color = 'black')
        plt.text(z_dp-.7, 4.9, s= r'DEATH', fontsize = 30)
        plt.text(z_dp-.7, 4.4, s= r'PLAIN', fontsize = 30)
        plt.fill_betweenx(y=[-2.8, 5.4], x1= zmin, x2=z_dp, color = 'black', alpha = 0.3)
    
    if (g==1.8):
        z_dv = np.sqrt(1/2*(2*(1+g-m)/g - np.sqrt(Delta)))
        c_dv = 2*g*z_dv*(-1 + 2/(z_dv**2+1+(m-1)/g) )
        plt.vlines(z_dv, -2.8, 5.4, linewidth = 5, color = 'black')
        plt.vlines(-z_dv, -2.8, 5.4, linewidth = 5, color = 'black')
        plt.text(-z_dv+0.13, 4.9, s= r'DEATH', fontsize = 30)
        plt.text(-z_dv+0.11, 4.4, s= r'VALLEY', fontsize = 30)
        plt.fill_betweenx(y=[-2.8, 5.4], x1=-z_dv, x2=z_dv, color = 'black', alpha = 0.3)    

    plt.savefig(subworkdir + "/phase_lines_all_speed.png", format='png', bbox_inches='tight')
    plt.show()
    plt.close()

