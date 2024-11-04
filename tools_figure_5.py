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
        if (t%100 ==0):
            print(t)
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


#### Function to build fig 5

def plot_pop_size_and_mean_at_eq(C, parameters, parameters_trait_space, subworkdir):
    zmin, zmax, Z, Nz, dz = parameters_trait_space
    parameters_trait_space_bis = zmin, zmax, np.linspace(zmin, zmax, Nz*10), Nz*10, dz/10
    epsilon, r, kappa, g, m, theta = parameters
    
    ####### Start of code section to compute the theoretical equilibrium 
    Ncbis = 300
    Cbis = np.linspace(0, np.max(C), num = Ncbis) # compute over more points over the same range of environmental speeds
    z_c, rho_c, size_c, f1 = np.zeros(Ncbis), np.zeros(Ncbis), np.zeros(Ncbis), np.zeros(Ncbis)
    
    Ncter = 30 # different speed grid for displaying the potential refugium's equilibrium (c \leq c_switch) 
    Cter = np.linspace(0, np.max(C), num = Ncter)
    z_c_refuge = np.zeros(Ncter)
    
    ### Call the function curve_per_parameter, which basically computes the selection gradient
    z_star, rho_star, F_z_star = curve_per_parameter(parameters, parameters_trait_space_bis)
    
    #### Loop over the environmental speeds to get the new equilibrium z_c and new metapop size size_c given the speed c
    for i in range(Ncbis):
        c = Cbis[i]
        z_c[i] = z_star[-np.argmin(F_z_star[::-1]<c)]
        rho_c[i] = rho_star[-np.argmin(F_z_star[::-1]<c)]
        f1[i] = 1 + g/m*(z_c[i]+1)**2 - 1/m
        size_c[i] = max(m*(rho_c[i] - f1[i])*(1 + rho_c[i]), 0)
    ## Same for refugium's equilibrium (notice the change of argument in np.argmax, scanning from the left)
    for i in range(Ncter):
        c = Cter[i]
        z_c_refuge[i] = z_star[np.argmax(F_z_star<c)]
        
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
    
    
    #### Compare the theoretical equilibium mean trait and numerical one on the same first figure    
    plt.figure(figsize=(11,6))
    plt.xlabel('Environmental speed $c$', fontsize = 40)
    plt.ylabel('$Z^*$', fontsize =40)
    plt.xticks([], fontsize = 40)
    plt.yticks([-1, 1], [r'$\theta_1(t)$', r'$\theta_2(t)$'], fontsize = 30)
    color = (z_c - min(z_c))/(np.max(z_c) - np.min(z_c))
    z_c_bis = mean_trait_end_num
    colorbis = (z_c_bis - min(z_c))/(np.max(z_c) - np.min(z_c))
    color_refuge = (z_c_refuge - min(z_c))/(np.max(z_c) - np.min(z_c))

    plt.set_cmap(viridis)
    
    ### Plot the mean trait : theoretical z_c and numerical mean_trait_end_num
    plt.scatter(Cbis, z_c, c = color, vmin = -10/19, vmax = 1+3/19, s = 30)
    plt.scatter(C, mean_trait_end_num,  c=colorbis, vmin = -10/19, vmax = 1+3/19, s = 200)
    plt.scatter(Cter, z_c_refuge, c = color_refuge, vmin = -10/19, vmax = 1+3/19, s = 30) ## Plot potential refugium's eq (c\leq c_switch)

    plt.plot(C[::3], mean_trait_end_num[::3], 'ks',markerfacecolor='none', color='black', ms = 30) # Plot black squares to relate to fig. 4
    if (g==1.4):
        plt.plot(C[-1], mean_trait_end_num[-1], 'ks',markerfacecolor='none', color='black', ms = 30)
        
        
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
    plt.plot(C[::3], size_end_num[::3], 'ks',markerfacecolor='none', color='black', ms = 30) ## Black squares to relate to fig 4 simulations
    if (g==1.4):
        plt.plot(C[-1], size_end_num[-1], 'ks',markerfacecolor='none', color='black', ms = 30)
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
    return()


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
            
    
