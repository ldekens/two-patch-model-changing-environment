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

    
    
##### Function that displays the figure 2
    
def phase_lines_dominant_trait(C, parameters, parameters_trait_space, subworkdir):
    
    # Unpack trait space parameters
    zmin, zmax, Z, Nz, dz = parameters_trait_space
    plt.figure(figsize=(12,9))
    plt.xlabel('$Z$', fontsize = 40)
    plt.ylabel(r'$\mathcal{F}^{c=0}(Z)$', fontsize = 40)
    plt.xticks(fontsize = 30)
    plt.yticks(fontsize = 30)
    
    #### Compute selection function under stable environment thanks to the function curve_per_parameter
    
    z_star, rho_star, F_z_star = curve_per_parameter(parameters, parameters_trait_space)
    plt.vlines(z_star[np.argmax([F_z_star<0])], -0.05, 0.05, color ='black')
    plt.vlines(0, -0.05, 0.05, linewidth =2,  color ='black')
    plt.text(-0.07, 0.1, '0', fontsize = 30)
    
    #### compute viability of equilibrium (does not depend on c)
    
    viability = viability_fast_equlibria_theoric(parameters, z_star)
    r, kappa, g, m, theta = parameters
    colors = [inferno(50), inferno(50), inferno(50)]
    ### Transparency parameter
    Alpha = np.ones(np.size(C))*1/3
    Alpha[-1] = 1
    # locate and plot the equilibria under stable environment
    count = 0
    plt.text(z_star[-np.argmax([F_z_star[::-1]>0])], 0.15, r'$ Z^*_{\text{spec}}$', alpha = Alpha[count], fontsize = 30, color = inferno(0))
    plt.text(z_star[np.argmax([F_z_star<0])]-0.1, 0.15, r'$ -Z^*_{\text{spec}}$', alpha = Alpha[count], fontsize = 30, color = inferno(0))
    
    #### Plot the downard translated selection gradient F_z_star (which becomes F_z_star_c)
    if (g < 1+2*m):
        for c in C:
            F_z_star_c = np.array(F_z_star - c)
            plt.plot(z_star[viability&(z_star>=0)], F_z_star_c[viability&(z_star>=0)], alpha = Alpha[count], linewidth = 3, label='$c = %4.2f'%c + '$', color = colors[count])
            plt.plot(z_star[viability&(z_star<=0)], F_z_star_c[viability&(z_star<=0)], alpha = Alpha[count], linewidth = 3, color = colors[count])
            plt.plot(z_star[((~viability)&(z_star>-1))], F_z_star_c[((~viability)&(z_star>-1))], alpha = Alpha[count], linewidth = 3, linestyle = 'dotted', color = colors[count])
            plt.plot(z_star[((~viability)&(z_star<-1))], F_z_star_c[((~viability)&(z_star<-1))], alpha = Alpha[count], linewidth = 3, linestyle = 'dotted', color = colors[count])
            zstarbis = z_star[z_star<=0]
            z_min = zstarbis[np.argmin(np.array(F_z_star)[z_star<=0])]
            F_min = F_z_star[np.argmin(np.array(F_z_star)[z_star<=0])]
            plt.scatter(z_star[- np.argmax([F_z_star_c[::-1]>0])], 0, s=200., alpha = Alpha[count], color = colors[count])
            if (count == 1):
                plt.text(z_star[-np.argmax([F_z_star_c[::-1]>0])] - 0.1, 0.15, r'$ Z^*_1$', alpha = Alpha[count], fontsize = 30, color = colors[count])
                plt.arrow(z_min+0.02, F_min, 0, -c, color = colors[count], alpha = Alpha[count],width = 0.02, length_includes_head=True)
                plt.text(z_min+0.04, F_min -c/2, '-$c_1$', alpha = Alpha[count], color = colors[count] ,fontsize = 30)
            if (count == 2):
                plt.text(z_star[-np.argmax([F_z_star_c[::-1]>0])]-0.1, 0.15, r'$ Z^*_2$', alpha = Alpha[count], fontsize = 30, color = colors[count])
                plt.arrow(z_min-0.02, F_min, 0, -c, alpha = Alpha[count], color = colors[count],width = 0.02, length_includes_head=True)
                plt.text(z_min-0.25, F_min -2*c/3, '-$c_2$', alpha = Alpha[count], color = colors[count] ,fontsize = 30)
            if (C[0]==C[-1]):
                indx = np.argmin(z_star<-1.5)
                plt.arrow(z_star[indx], F_z_star[indx], z_star[indx+1] - z_star[indx] , F_z_star[indx+1] - F_z_star[indx], width = 0.02, length_includes_head=False, color = colors[count], head_length= 0.15, head_width = 0.13)  
                indx = np.argmin(z_star<-0.6)
                plt.arrow(z_star[indx], F_z_star[indx], z_star[indx] - z_star[indx+1] , F_z_star[indx] - F_z_star[indx+1], width = 0.02, length_includes_head=False, color = colors[count], head_length= 0.15, head_width = 0.13)
                indx = np.argmin(z_star<0.6)
                plt.arrow(z_star[indx], F_z_star[indx], z_star[indx+1] - z_star[indx] , F_z_star[indx+1] - F_z_star[indx], width = 0.02, length_includes_head=False, color = colors[count], head_length= 0.15, head_width = 0.13)
                indx = np.argmin(z_star<1.1)
                plt.arrow(z_star[indx], F_z_star[indx], z_star[indx] - z_star[indx+1] , F_z_star[indx] - F_z_star[indx+1], width = 0.02, length_includes_head=False, color = colors[count], head_length= 0.15, head_width = 0.13)
            else:
                if (count ==2):
                    indx = np.argmin(z_star<-1.7)
                    plt.arrow(z_star[indx], F_z_star_c[indx], z_star[indx+1] - z_star[indx] , F_z_star_c[indx+1] - F_z_star_c[indx], width = 0.02, length_includes_head=False, color = colors[count], head_length= 0.15, head_width = 0.13)
                    indx = np.argmin(z_star<-0.6)
                    plt.arrow(z_star[indx], F_z_star_c[indx], z_star[indx] - z_star[indx+1] , F_z_star_c[indx] - F_z_star_c[indx+1], width = 0.02, length_includes_head=False, color = colors[count], head_length= 0.15, head_width = 0.13)
                    indx = np.argmin(z_star<0.1)
                    plt.arrow(z_star[indx], F_z_star_c[indx], z_star[indx] - z_star[indx+1] , F_z_star_c[indx] - F_z_star_c[indx+1], width = 0.02, length_includes_head=False, color = colors[count], head_length= 0.15, head_width = 0.13)
                    indx = np.argmin(z_star<1.1)
                    plt.arrow(z_star[indx], F_z_star_c[indx], z_star[indx] - z_star[indx+1] , F_z_star_c[indx] - F_z_star_c[indx+1], width = 0.02, length_includes_head=False, color = colors[count], head_length= 0.15, head_width = 0.13)
            count = count +1
        
    if (g > 1+2*m):
        for c in C:
            F_z_star_c = np.array(F_z_star - c)
            plt.plot(z_star[viability&(z_star>0)], F_z_star_c[viability&(z_star>0)], linewidth = 2, label='$c = %4.2f'%c + '$')
            plt.plot(z_star[viability&(z_star<=0)], F_z_star_c[viability&(z_star<=0)], linewidth = 2)
            plt.plot(z_star[((~viability)&(z_star>-1))], F_z_star_c[((~viability)&(z_star>-1))],  linewidth = 2, linestyle = 'dotted')
            plt.plot(z_star[((~viability)&(z_star<-1))], F_z_star_c[((~viability)&(z_star<-1))], linewidth = 2, linestyle = 'dotted')
            plt.scatter(z_star[- np.argmax([F_z_star_c[::-1]>0])], 0, s=200.)
            
            plt.text(z_star[- np.argmax([F_z_star_c[::-1]>0])], 0.2, r'$ Z^*_{\text{spec}}$', fontsize = 30)
            
            count = count +1
    
    #### Overlay a visual representation of the selection functions
    plt.plot(Z[(Z-1)**2<1.4], -g*(Z[(Z-1)**2<1.4]-1)**2, linewidth = 8, alpha = 0.2, color = viridis(200))
    plt.plot(Z[(Z+1)**2<1.4], -g*(Z[(Z+1)**2<1.4]+1)**2, linewidth = 8, alpha = 0.2, color = viridis(70))
    
    
    #### The following section is to build the visual grey area corresponding to the death plain
    Delta = 4/g**2*(m**2-4*g*(m-1))
    z_dp = -np.sqrt(1/2*(2*(1+g-m)/g + np.sqrt(Delta)))

    plt.vlines(z_dp, -1.8, 2.1, linewidth = 5, color = 'black')
    plt.text(z_dp-.6, 2.37, s= r'DEATH', fontsize = 30)
    plt.text(z_dp-.6, 2.17, s= r'PLAIN', fontsize = 30)
    plt.fill_betweenx(y=[-1.8, 2.1], x1= zmin, x2=z_dp, color = 'black', alpha = 0.3)

    plt.plot(Z, np.zeros(np.size(Z)), color='black', linewidth=2.5)
    
    plt.savefig(subworkdir + "/phase_lines_%i"%count+".png", format='png', bbox_inches='tight')
    plt.show()
    plt.close()
    return()
