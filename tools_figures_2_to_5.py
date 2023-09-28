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



#### Function that returns the subpopulations sizes and the metapopulation mean trait of a specialist species under stable environment, according to Proposition 4.2 of [Dekens 2022]
def analytical_asym_eq_hab_2(parameters):
    epsilon, r, kappa, g, m, theta = parameters
    ### assuming r, theta and kappa are normalized
    if (1+2*m > 5*g):
        raise Exception('No specialist equilibria can exist under these parameters.')
    else:
        S = [1., (1-4*g)/m, -4*g/m, 4*g/m]
        y = np.max(np.roots(S)[np.isreal(np.roots(S))])
        rho = (y + np.sqrt(y**2-4))/2
        N1, N2, z = (1 - m) + m*rho -4*g*rho**4/(rho**2 + 1)**2, (1 - m) + m/rho -4*g/(rho**2 + 1)**2, (rho**2 - 1)/(rho**2 + 1)
        return(N1, N2, z)



### Auxiliary function that splits the work between parallel processes handling their own environmental speed c of C
def run_model(n1, n2, C, parameters, parameters_time, parameters_trait_space, subworkdir):
    pool = multiprocessing.Pool(processes = 6)
    inputs = [*zip(C, repeat(n1), repeat(n2), repeat(parameters), repeat(parameters_time), repeat(parameters_trait_space), repeat(subworkdir))]
    pool.starmap(run_per_c, inputs)
    
######## Function that runs the numerical resolution of Eq 7 for a given environmental speed c
def run_per_c(c, n1, n2, parameters, parameters_time, parameters_trait_space, subworkdir):
    epsilon, r, kappa, g, m, theta = parameters
    zmin, zmax, z, Nz, dz = parameters_trait_space
    T, Nt, dt = parameters_time
    subworkdir_per_c = subworkdir + '/c=%4.2f'%c
    create_directory(subworkdir_per_c)
    
    moments_1 = np.zeros((4,np.size(T)))
    moments_2 = np.zeros((4,np.size(T)))
    
    #### Auxiliary matrices
    #Migration
    Mmigration = sp.coo_matrix(np.block([[-np.eye(Nz)*m, np.eye(Nz)*m],[np.eye(Nz)*m,-np.eye(Nz)*m]]))
    
    #selection
    Vselection1 = -g*(z+theta)*(z+theta)
    Vselection2 = -g*(z-theta)*(z-theta)
    Vselection = np.concatenate((Vselection1, Vselection2))
    Mselection = sp.spdiags(Vselection, 0, 2*Nz, 2*Nz)
    
    #Advection (environmental change effect of distributions)
    if (c<=0):
        Advect_aux = c*epsilon**2/dz*(np.diag(np.ones(Nz), k = 0) - np.diag(np.ones(Nz-1), k = -1))
        Advect_aux[0, 0] = 0 # boundary conditions
    else:
        Advect_aux = c*epsilon**2/dz*(-np.diag(np.ones(Nz), k = 0) + np.diag(np.ones(Nz-1), k = 1))
        Advect_aux[-1, -1] = 0

    Advect = sp.block_diag((Advect_aux, Advect_aux))
    
    Id = sp.spdiags(np.ones(2*Nz), 0, 2*Nz, 2*Nz)
    ### Build the matrix used in the semi implicit scheme from the preivous ones
    M_semi_implicit_scheme = Id - dt/(epsilon**2)*(Mmigration + Mselection + Advect)
    
    ### Loop across time range to update the numerical scheme at each time step
    for t in range(Nt):
        moments_1[:, t] = moments(n1, z, dz)
        moments_2[:, t] = moments(n2, z, dz)
        n1, n2 = update(n1, n2, parameters, M_semi_implicit_scheme, dt, parameters_trait_space)

    ### Save the temporal series of the subpopulation sizes and local mean traits
    np.save(subworkdir_per_c +'/moments_1.npy', moments_1)
    np.save(subworkdir_per_c +'/moments_2.npy', moments_2)

    

####### Function that implements the scheme iterations necessary for the numerical resolution of Eq 7
def update(n1, n2, parameters, M_semi_implicit_scheme, dt, parameters_trait_space):
    epsilon, r, kappa, g, m, theta = parameters
    zmin, zmax, z, Nz, dz = parameters_trait_space

    
    N1, N2 = sum(n1*dz), sum(n2*dz)
    # Reproduction terms
    B1, B2 = reproduction_conv(n1, n1, N1, epsilon, zmin, zmax, Nz), reproduction_conv(n2, n2, N2, epsilon, zmin, zmax, Nz)
    B12 = np.concatenate((B1, B2))
    ### Build a single vector from the two distributions
    n12aux = np.concatenate((n1, n2))
    N12 = np.concatenate((np.ones(Nz)*N1, np.ones(Nz)*N2))
    
    #### Solve the numerical scheme equation
    n12 = scisplin.spsolve((M_semi_implicit_scheme + dt/(epsilon**2)*sp.spdiags(kappa*N12, 0, 2*Nz, 2*Nz)),(n12aux + dt/(epsilon**2)*r*B12))
    
    #### Split the concatenated solution to the two distributions and return them
    n1 = n12[:Nz]
    n2 = n12[Nz:]
    return(n1, n2)


############# Auxiliary functions necessary to run the update function
####### Creates grids used in the reproduction operator
def grid_double_conv(zmin, zmax, Nz):
    z, dz = np.linspace(zmin, zmax, Nz, retstep=True)
    zGauss, dzGauss = np.linspace((zmin-zmax)/2, (zmax-zmin)/2, 2*Nz-1, retstep=True)
    z2, dz2 = np.linspace(zmin + (zmin-zmax)/2, zmax + (zmax - zmin)/2, 4*Nz-3, retstep=True)
    
    return(Nz, z, dz, zGauss, dzGauss, z2, dz2)

####### Creates a discritized Gaussian distribution with mean m and variance s**2 on the grid z
def Gauss(m, s, z):
    Nz = np.size(z)
    G = np.zeros(Nz)
    for k in range(Nz):
        G[k] = 1/( np.sqrt(2*np.pi)*s )* np.exp( - (z[k]-m)**2 / (2*s**2) )
    return(G)
    
####### Encodes the reproduction operator as a double convolution
def reproduction_conv(n1, n2, N, epsilon, zmin, zmax, Nz):
    Nz, z, dz, zGauss, dzGauss, zaux, dzaux = grid_double_conv(zmin, zmax, Nz)

    Gsex = Gauss(0, epsilon/np.sqrt(2), zGauss)
    Bconv_aux = scsign.convolve(scsign.convolve(n1, Gsex*dz)*dz, n2)
    if (N>0):
        Bconv = np.interp(z, zaux, Bconv_aux)/N
    else:
        Bconv = np.zeros(np.size(z))
    return(Bconv)

####### Compute the first four moments of a given a discrete distribution n on grid z and step dz
def moments(n, z, dz):
    N = sum(n)*dz
    if (N>0):
        m = sum(z*n)*dz/N
        v = sum((z - m)**2*n)*dz/N
        s = sum((z - m)**3/np.sqrt(v)**3*n)*dz/N
    else:
        m, v, s = np.empty(3)
    return(N, m , v, s)

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
    epsilon, r, kappa, g, m, theta = parameters
    P = [1, 1/m - 1 - g/m*(z + theta)**2, -1/m + 1 + g/m*(z -theta)**2, -1]
    roots = search_positive_roots(P)
    return(roots)

### Compute the quantities from the phase-line analysis of the selection gradient
def curve_per_parameter(parameters, parameters_trait_space): 
    epsilon, r, kappa, g, m, theta = parameters
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
    epsilon, r, kappa, g, m, theta = parameters
    
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
    epsilon, r, kappa, g, m, theta = parameters
    colors = [inferno(50), inferno(50), inferno(50)]
    count=0
    Alpha = np.ones(np.size(C))*1/3
    Alpha[-1] = 1
    # locate and plot the equilibria under stable environment
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
    
    
    #### The following section is to build the visual grey area corresponding to the death plain and death valley (tailored to each set of parameters)
    Delta = 4/g**2*(m**2-4*g*(m-1))
    z_dp = -np.sqrt(1/2*(2*(1+g-m)/g + np.sqrt(Delta)))
    if (g>1):
        if (count==1):
            plt.scatter(0, -1, s = 200, marker = 'X', color = 'black')
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
        z_dv = np.sqrt(1/2*(2*(1+g-m)/g - np.sqrt(Delta)))
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
        plt.vlines(z_dv, -2.8, 5.4, linewidth = 5, color = 'black')
        plt.vlines(-z_dv, -2.8, 5.4, linewidth = 5, color = 'black')
        plt.text(-z_dv+0.13, 4.9, s= r'DEATH', fontsize = 30)
        plt.text(-z_dv+0.11, 4.4, s= r'VALLEY', fontsize = 30)
        plt.fill_betweenx(y=[-2.8, 5.4], x1=-z_dv, x2=z_dv, color = 'black', alpha = 0.3)
    plt.plot(Z, np.zeros(np.size(Z)), color='black', linewidth=2.5)
    
    plt.savefig(subworkdir + "/phase_lines_%i"%count+".png", format='png', bbox_inches='tight')
    plt.show()
    plt.close()
    return()
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
    epsilon, r, kappa, g, m, theta = parameters
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



#### Function to build fig 4
def plot_pop_size_and_mean_at_eq(C, parameters, parameters_trait_space, subworkdir):
    zmin, zmax, Z, Nz, dz = parameters_trait_space
    parameters_trait_space_bis = zmin, zmax, np.linspace(zmin, zmax, Nz*10), Nz*10, dz/10
    epsilon, r, kappa, g, m, theta = parameters
    
    ####### Start of code section to compute the theoretical equilibrium 
    Ncbis = 300
    Cbis = np.linspace(0, np.max(C), num = Ncbis) # compute over more points over the same range of environmental speeds
    z_c, rho_c, size_c, f1 = np.zeros(Ncbis), np.zeros(Ncbis), np.zeros(Ncbis), np.zeros(Ncbis)
    
    ### Call the function curve_per_parameter, which basically computes the selection gradient
    z_star, rho_star, F_z_star = curve_per_parameter(parameters, parameters_trait_space_bis)
    
    #### Loop over the environmental speeds to get the new equilibrium z_c and new metapop size size_c given the speed c
    for i in range(Ncbis):
        c = Cbis[i]
        z_c[i] = z_star[-np.argmin(F_z_star[::-1]<c)]
        rho_c[i] = rho_star[-np.argmin(F_z_star[::-1]<c)]
        f1[i] = 1 + g/m*(z_c[i]+1)**2 - 1/m
        size_c[i] = max(m*(rho_c[i] - f1[i])*(1 + rho_c[i]), 0)
    
    #### Compute the quantities related to the habitat switch
    c_switch = np.max(F_z_star[z_star>=0])  ### critical speed of switch
    z_star_bis = z_star[z_star >0] 
    z_switch = z_star_bis[np.argmax(F_z_star[z_star>=0])] ##### mean trait just before the switch
    z_post_switch = z_star[- np.argmax([F_z_star[::-1]>c_switch])] ##### mean trait just after the switch
    
    ####### End of code section to compute the theoretical prediction for the equilibrium
    
    ####### Read numerical equlibrium for speeds c in C from previously run simulations (from run_model call in the main code)

    Nc = len(C)
    size_end_num = np.zeros(Nc)
    mean_trait_end_num = np.zeros(Nc)
    
    ### Loop over C to get the end state of the numerical resolution for a given speed c
    for i in range(Nc):
        c = C[i]
        subworkdir_per_c = subworkdir + '/c=%4.2f'%c
        N1, N2 =np.load(subworkdir_per_c +'/moments_1.npy')[0, :], np.load(subworkdir_per_c +'/moments_2.npy')[0, :]
        size_end_num[i] = np.load(subworkdir_per_c +'/moments_1.npy')[0, -1] + np.load(subworkdir_per_c +'/moments_2.npy')[0, -1]
        cutoff = 1e-100  #### artificial cut-off of pop size to be able to compute non degenerate mean traits
        if (np.argmin(N1+N2>cutoff)>0): ### if the cut-off is reached (ie. pop si going extinct), take the mean trait at cutoff time
            mean_trait_end_num[i] = np.load(subworkdir_per_c +'/moments_1.npy')[1, np.argmin(N1+N2>cutoff)]/2 + np.load(subworkdir_per_c +'/moments_2.npy')[1, np.argmin(N1+N2>cutoff)]/2
        else: #### else just take the mean from the final state
            mean_trait_end_num[i] = np.load(subworkdir_per_c +'/moments_1.npy')[1, -1]/2 + np.load(subworkdir_per_c +'/moments_2.npy')[1, -1]/2
    
    
    #### Compare the theoretical equilibium mean trait and numerical one on the same first figure    plt.figure(figsize=(11,6))
    plt.xlabel('Environmental speed $c$', fontsize = 40)
    plt.ylabel('$Z^*$', fontsize =40)
    plt.xticks([], fontsize = 40)
    plt.yticks([-1, 1], [r'$\theta_1(t)$', r'$\theta_2(t)$'], fontsize = 30)
    color = (z_c - min(z_c))/(np.max(z_c) - np.min(z_c))
    z_c_bis = mean_trait_end_num
    colorbis = (z_c_bis - min(z_c))/(np.max(z_c) - np.min(z_c))

    plt.set_cmap(viridis)
    ### Plot the mean trait : theoretical z_c and numerical mean_trait_end_num
    plt.scatter(Cbis, z_c, c = color, vmin = -10/19, vmax = 1+3/19, s = 30)
    plt.scatter(C, mean_trait_end_num,  c=colorbis, vmin = -10/19, vmax = 1+3/19, s = 200)
    ### Add the visual for the habitat switch (quite long and specific)
    plt.text(c_switch +0.1, 0.6, 'Native',fontsize = 30)
    plt.text(c_switch +0.1, 0.4, 'habitat',fontsize = 30)

    plt.vlines(c_switch + 0.07, max(z_c), z_switch, color = 'black')
    plt.hlines(max(z_c), c_switch, c_switch+ 0.07, color = 'black')
    plt.hlines(z_switch, c_switch, c_switch+ 0.07, color = 'black')
    
    if (g==1.8):
        plt.text(c_switch - 0.5, z_switch/2 + z_post_switch/2 , 'Switch',fontsize = 30)
        plt.text(c_switch - 0.75, z_post_switch/2 + (max(z_c)-3)/2 , 'Refugium',fontsize = 30)
    else:
        plt.text(c_switch - 0.35, z_switch/2 + z_post_switch/2 , 'Switch',fontsize = 30)
        plt.text(c_switch - 0.55, z_post_switch/2 + (max(z_c)-3)/2 , 'Refugium',fontsize = 30)

    plt.arrow(c_switch, z_switch, 0, z_post_switch-z_switch, color = 'black', width = 0.01, length_includes_head = True, head_length= 0.12, head_width = 0.07)

    plt.vlines(c_switch - 0.07, z_post_switch, max(z_c)-3, color = 'black')
    plt.hlines(z_post_switch, c_switch, c_switch- 0.07, color = 'black')
    plt.hlines(max(z_c)- 3, c_switch, c_switch- 0.07, color = 'black')
    plt.vlines(c_switch, - 0.05,0.05, color = 'RoyalBlue', linewidth = 3)
    if (g==1.8):
        plt.xticks([0, c_switch, 3],[0, r'$c_{\text{switch}}$', 3], fontsize = 45)
        my_colors =  ['black', 'RoyalBlue','black']
        for ticklabel, tickcolor in zip(plt.gca().get_xticklabels(), my_colors):
            ticklabel.set_color(tickcolor)
    else:
        plt.xticks([0, c_switch, 2],[0, r'$c_{\text{switch}}$', 2], fontsize = 45)
        my_colors =  ['black', 'RoyalBlue','black']
        for ticklabel, tickcolor in zip(plt.gca().get_xticklabels(), my_colors):
            ticklabel.set_color(tickcolor)

    plt.savefig(subworkdir + '/mean_at_eq.png', bbox_inches='tight')
    plt.show()
    plt.close()
    
    #### End of the figure comparing theoretical and numerical equilibrium mean trait
    
    #### Compare the theoretical equilibium metapop size and numerical one on the same second figure

    plt.figure(figsize=(11,6))
    
    plt.set_cmap(viridis)
    plt.scatter(Cbis, size_c, c = color, vmin = -10/19, vmax = 1+3/19, s=40) #### plot theoretical equilibrium
    plt.scatter(C, size_end_num,  c=colorbis, vmin = -10/19, vmax = 1+3/19, s = 200) #### plot numerical equilibrium
    plt.hlines(0, 0, max(C), color = 'black') ### plot horizontal line for 0

    plt.xlabel('Environmental speed $c$', fontsize = 40)
    plt.ylabel('$N^*$', fontsize = 40)
    plt.vlines(c_switch, - 0.05,0.05, color = 'RoyalBlue', linewidth = 6)
    
    ### Auxiliary quantities to compute and visualize the critical speed for persistence
    Delta = 4/g**2*(m**2-4*g*(m-1))
    z_dp = -np.sqrt(1/2*(2*(1+g-m)/g + np.sqrt(Delta)))
    c_dp = -2*g*z_dp*(1 - 2/(z_dp**2+1+(m-1)/g) )
    c_crit = Cbis[-np.argmax(size_c[::-1]>0)] #### theoretical critical speed for persistence


    plt.vlines(c_dp, - 0.05,0.05, color = 'darkred', linewidth = 6)
    if (g>1):
        z_dv = np.sqrt(1/2*(2*(1+g-m)/g - np.sqrt(Delta)))
        c_dv = 2*g*z_dv*(-1 + 2/(z_dv**2+1+(m-1)/g) )
        plt.vlines(c_dv, - 0.05,0.05, color = 'chocolate', linewidth = 6)

    if (g==1.8):
        plt.xticks([0, c_crit, 3],[0, r'$c_{\text{death valley}}$', 3], fontsize = 45)
        my_colors =  ['black', 'chocolate','black']
        for ticklabel, tickcolor in zip(plt.gca().get_xticklabels(), my_colors):
            ticklabel.set_color(tickcolor)
    else:
        plt.xticks([0, c_crit, 2],[0, r'$c_{\text{death plain}}$', 2], fontsize = 45)
        my_colors =  ['black', 'darkred','black']
        for ticklabel, tickcolor in zip(plt.gca().get_xticklabels(), my_colors):
            ticklabel.set_color(tickcolor)
    
    plt.locator_params(axis='both', nbins=4)
    plt.yticks(fontsize = 35)
    plt.set_cmap(viridis)


    plt.savefig(subworkdir + '/size_at_eq.png', bbox_inches='tight')
    plt.show()
    plt.close()
   

    zmin, zmax, Z, Nz, dz = parameters_trait_space
    parameters_trait_space_bis = zmin, zmax, np.linspace(zmin, zmax, Nz*10), Nz*10, dz/10
    epsilon, r, kappa, g, m, theta = parameters
    Ng = len(G)
    #### Theoretical size at equilibrium depending on speed in Cbis
    Ncbis = 100
    Cbis = np.linspace(0, np.max(C), num = Ncbis)
    z_c, z_c_bis, f1_bis, size_c = np.zeros((Ncbis, Ng)), np.zeros((Ncbis, Ng)), np.zeros((Ncbis, Ng)), np.zeros((Ncbis, Ng))
    
    plt.figure(figsize=(9,6))
    plt.xlabel('Environmental speed $c$', fontsize = 40)
    plt.ylabel('Metapopulation size', fontsize = 40)
    plt.xticks(fontsize = 25)
    plt.yticks(fontsize = 25)
    for j in range(Ng):
        g = G[j]
        parameters = epsilon, r, kappa, g, m, theta
        z_star, rho_star, F_z_star = curve_per_parameter(parameters, parameters_trait_space_bis) ### compute mean trait at eq, ratio at eq
        for i in range(Ncbis):
            c = Cbis[i]
            z_c[i, j] = z_star[-np.argmin(F_z_star[::-1]<c)]
            z_c_bis[i, j] = z_star[np.argmin(F_z_star>c)]
            rho_c = rho_star[-np.argmin(F_z_star[::-1]<c)]
            f1 = 1 + g/m*(z_c[i, j]+1)**2 - 1/m
            f1_bis[i, j] = 1 + g/m*(z_c_bis[i,j ]+1)**2 - 1/m
            size_c[i, j] = max(m*(rho_c - f1)*(1 + rho_c), 0)
        Delta = 4/g**2*(m**2-4*g*(m-1))
        z2 = 1/2*(2*(1+g-m)/g + np.sqrt(Delta))
        c2 = 2*g*np.sqrt(z2)*(1 - 2/(z2+1+(m-1)/g) )
        c = np.max(F_z_star[np.argmin(z_star<0):])
        if (c > c2):

            if (g>1):
                z1 = 1/2*(2*(1+g-m)/g - np.sqrt(Delta))
                c1 = 2*g*np.sqrt(z1)*(2/(z1+1+(m-1)/g) - 1 )
                clim = c1
            else:
                clim = c
        else:
            clim = c2
        
        color_g = viridis(int(250*j/Ng))
        plt.plot(Cbis[1:-1], (size_c[2:, j]-size_c[:-2, j])/2, color = color_g, label = '$g=%4.2f'%g + '$')
        #plt.vlines(clim, 0, 0.05, linewidth= 2, color = color_g, linestyle = "dashed")
    plt.legend()
    plt.savefig(workdir + '/size_at_eq_th.png', bbox_inches='tight')

    plt.show()
    plt.close()
    plt.figure(figsize=(9,6))
    plt.xlabel('Environmental speed $c$', fontsize = 40)
    plt.ylabel('$z$ after jump', fontsize = 40)
    plt.xticks(fontsize = 25)
    plt.yticks(fontsize = 25)
    for j in range(Ng):
        g = G[j]
        color_g = viridis(int(250*j/Ng))
        plt.plot(Cbis, z_c_bis[:, j], color = color_g, label = '$g=%4.2f'%g + '$')
        plt.plot(Cbis, -1- Cbis/(2*g), linestyle = 'dashed', color = color_g)
    plt.legend()
    plt.savefig(workdir + '/z_after_jump.png', bbox_inches='tight')

    plt.show()
    plt.close()
    plt.figure(figsize=(9,6))
    plt.xlabel('Environmental speed $c$', fontsize = 40)
    plt.ylabel('$z$ before jump', fontsize = 40)
    plt.xticks(fontsize = 25)
    plt.yticks(fontsize = 25)
    for j in range(Ng):
        g = G[j]
        color_g = viridis(int(250*j/Ng))
        plt.plot(Cbis, z_c[:, j], color = color_g, label = '$g=%4.2f'%g + '$')
        plt.plot(Cbis, 1- Cbis/(2*g), linestyle = 'dashed', color = color_g)
        plt.plot(Cbis, -1- Cbis/(2*g), linestyle = 'dashed', color = color_g)
    plt.legend()
    plt.savefig(workdir + '/z_before_jump.png', bbox_inches='tight')

    plt.show()
    plt.close()
    zmin, zmax, Z, Nz, dz = parameters_trait_space
    parameters_trait_space_bis = zmin, zmax, np.linspace(zmin, zmax, Nz*10), Nz*10, dz/10
    epsilon, r, kappa, g, m, theta = parameters
    Ng = len(G)
    #### Theoretical size at equilibrium depending on speed in Cbis
    Ncbis = 300
    Cbis = np.linspace(0, np.max(C), num = Ncbis)
    z_c = np.zeros((Ncbis, Ng))
    
    plt.figure(figsize=(9,6))
    plt.xlabel('Environmental speed $c$', fontsize = 40)
    plt.ylabel('Mean trait', fontsize = 40)
    plt.xticks(fontsize = 25)
    plt.yticks(fontsize = 25)
    for j in range(Ng):
        g = G[j]
        parameters = epsilon, r, kappa, g, m, theta
        z_star, rho_star, F_z_star = curve_per_parameter(parameters, parameters_trait_space_bis) ### compute mean trait at eq, ratio at eq
        for i in range(Ncbis):
            c = Cbis[i]
            z_c_bis = z_star[-np.argmin(F_z_star[::-1]<c)]
            rho_c = rho_star[-np.argmin(F_z_star[::-1]<c)]
            f1 = 1 + g/m*(z_c_bis+1)**2 - 1/m
            if (rho_c > f1):
                z_c[i, j] = z_c_bis
            else:
                z_c[i, j] = np.nan
        color_g = viridis(int(250*j/Ng))
        print(z_c)
        plt.plot(Cbis, z_c[:, j], color = color_g, label = '$g=%4.2f'%g + '$')
    plt.legend()
    plt.savefig(subworkdir + '/mean_at_eq_th.png', bbox_inches='tight')

    plt.show()
    plt.close()    
#### Function to build figure 5

def plot_critical_speed_th(G, parameters, parameters_trait_space, subworkdir):
    zmin, zmax, Z, Nz, dz = parameters_trait_space
    z = np.linspace(zmin, zmax, Nz*10)
    parameters_trait_space_bis = zmin, zmax, z, Nz*10, dz/10
    epsilon, r, kappa, g, m, theta = parameters
    Ng = len(G)
    c_crit_switch = np.zeros(Ng)
    c_crit_persistence = np.zeros(Ng)
    c_crit_viab_2 = np.zeros(Ng)
    c_crit_viab_1 = np.zeros(Ng)
    Zstar = np.zeros(Ng)
    z1 = np.zeros(Ng)
    
    #### Computation of the different critical speeds for a given selection strenght g
    for j in range(Ng):
        g = G[j]
        parameters = epsilon, r, kappa, g, m, theta
        z_star, rho_star, F_z_star = curve_per_parameter(parameters, parameters_trait_space_bis) ### compute mean trait at eq, ratio at eq
        c_crit_switch[j] = max(np.array(F_z_star)[z_star>=0]) ### critical speed for switch
        zstarbis = z_star[z_star>=0]
        Zstar[j] = zstarbis[np.argmax(np.array(F_z_star)[z_star>=0])] ### mean trait at switch
        
        ### Computation of critical speed for persistence from to potential critical speeds c_crit_viab_1 and c_crit_viab_2 (c1 and c2 in main text)
        Delta = 4/g**2*(m**2-4*g*(m-1))
        z2 = 1/2*(2*(1+g-m)/g + np.sqrt(Delta))
        c_crit_viab_2[j] = -2*g*np.sqrt(z2)*(2/(z2+1+(m-1)/g) - 1)
        if (g>=1):
            z1[j] = 1/2*(2*(1+g-m)/g - np.sqrt(Delta))
            c_crit_viab_1[j] = 2*g*np.sqrt(z1[j])*(2/(z1[j]+1+(m-1)/g) - 1 )
        if (c_crit_switch[j] > c_crit_viab_2[j]):
            if (g>1):
                c_crit_persistence[j] = c_crit_viab_1[j]
            else:
                c_crit_persistence[j] = c_crit_switch[j]
        else:
            c_crit_persistence[j] = c_crit_viab_2[j]
            
    #### Display the critical speeds previously computed
    plt.figure(figsize=(15,11))
    plt.xlabel('Selection intensity $g$', fontsize = 50)
    plt.ylabel('Env. speed $c$', fontsize = 50)
    plt.xticks(fontsize = 35)
    plt.yticks(fontsize = 35)
    
    plt.plot(G, c_crit_switch, color ='Royalblue', linewidth = 5, label = 'Habitat switch')
    plt.plot(G, c_crit_viab_2, color ='darkred', linewidth = 5, label = 'Death Plain')
    plt.plot(G[(np.sqrt(z1)<Zstar)&(G>=1)], c_crit_viab_1[(np.sqrt(z1)<Zstar)&(G>=1)], color ='chocolate', linewidth = 5, label = r'Death Valley')
    plt.plot(G[(np.sqrt(z1)>Zstar)&(G>=1)], c_crit_viab_1[(np.sqrt(z1)>Zstar)&(G>=1)], color ='chocolate', linewidth = 5 )
    plt.plot(G, c_crit_persistence, color ='black', linewidth = 6., label = 'Extinction')
    plt.vlines(1, 0, 3, linestyle = 'dashed', color ='black' )
    Gbis = G[G>1]
    plt.vlines(Gbis[np.argmin(c_crit_switch[G>1] - c_crit_viab_1[G>1])], 0, 3, linestyle = 'dashed', color ='black' )
    plt.vlines(G[c_crit_viab_2 > c_crit_switch][-1], 0, 3, linestyle = 'dashed', color ='black' )

    plt.legend(fontsize = 35)
    plt.savefig(subworkdir + '/critical_speed_habitat_switch_persistence_th.png', bbox_inches='tight')

    plt.show()
    plt.close()
    return()
    zmin, zmax, Z, Nz, dz = parameters_trait_space
    z = np.linspace(zmin, zmax, Nz*10)
    print(z[z>0])
    parameters_trait_space_bis = zmin, zmax, z, Nz*10, dz/10
    epsilon, r, kappa, g, m, theta = parameters
    Ng = len(G)
    #### Theoretical size at equilibrium depending on speed in Cbis
    c_crit_switch = np.zeros(Ng)
    c_crit_persistence = np.zeros(Ng)
    c_crit_viab_2 = np.zeros(Ng)
    c1 = np.zeros(Ng)
    Zstar = np.zeros(Ng)
    z1 = np.zeros(Ng)

    plt.figure(figsize=(15,11))
    plt.xlabel('Selection intensity $g$', fontsize = 50)
    plt.ylabel('Env. speed $c$', fontsize = 50)
    plt.xticks(fontsize = 35)
    plt.yticks(fontsize = 35)
    for j in range(Ng):
        g = G[j]
        parameters = epsilon, r, kappa, g, m, theta
        z_star, rho_star, F_z_star = curve_per_parameter(parameters, parameters_trait_space_bis) ### compute mean trait at eq, ratio at eq
        c_crit_switch[j] = max(np.array(F_z_star)[z_star>=0])
        zstarbis = z_star[z_star>=0]
        Zstar[j] = zstarbis[np.argmax(np.array(F_z_star)[z_star>=0])]
        Delta = 4/g**2*(m**2-4*g*(m-1))
        z2 = 1/2*(2*(1+g-m)/g + np.sqrt(Delta))
        c_crit_viab_2[j] = -2*g*np.sqrt(z2)*(2/(z2+1+(m-1)/g) - 1)
        if (g>=1):
            z1[j] = 1/2*(2*(1+g-m)/g - np.sqrt(Delta))
            c1[j] = 2*g*np.sqrt(z1[j])*(2/(z1[j]+1+(m-1)/g) - 1 )
        if (c_crit_switch[j] > c_crit_viab_2[j]):
            if (g>1):
                c_crit_persistence[j] = c1[j]
            else:
                c_crit_persistence[j] = c_crit_switch[j]
        else:
            c_crit_persistence[j] = c_crit_viab_2[j]
    plt.plot(G, c_crit_switch, color ='Royalblue', linewidth = 5, label = 'Habitat switch', alpha = 0.5)
    plt.plot(G, c_crit_viab_2, color ='darkred', linewidth = 5, label = 'Death Plain', alpha = 0.5)
    plt.plot(G[(np.sqrt(z1)<Zstar)&(G>=1)], c1[(np.sqrt(z1)<Zstar)&(G>=1)], color ='chocolate', linewidth = 5, label = r'Death Valley', alpha = 0.5)
    plt.plot(G[(np.sqrt(z1)>Zstar)&(G>=1)], c1[(np.sqrt(z1)>Zstar)&(G>=1)], color ='chocolate', linewidth = 5, alpha = 0.5 )
    plt.plot(G, c_crit_persistence, color ='black', linewidth = 6., label = 'Extinction')
    #plt.vlines(1, 0, 3, linestyle = 'dashed', color ='black' )
    Gbis = G[G>1]
    #plt.vlines(Gbis[np.argmin(c_crit_switch[G>1] - c1[G>1])], 0, 3, linestyle = 'dashed', color ='black' )
    #plt.vlines(G[c_crit_viab_2 > c_crit_switch][-1], 0, 3, linestyle = 'dashed', color ='black' )

    plt.legend(fontsize = 35)
    plt.savefig(subworkdir + '/critical_speed_habitat_switch_persistence_th_short.png', bbox_inches='tight')

    plt.show()
    plt.close()
