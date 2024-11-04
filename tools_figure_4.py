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
    pool = multiprocessing.Pool(processes = 5)
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


### Auxiliary function that splits the work between parallel processes handling their own environmental speed c of C

def run_limit_model(z0, N10, N20, C, parameters, parameters_time, subworkdir):
    pool = multiprocessing.Pool(processes = 5)
    inputs = [*zip(C, repeat(z0), repeat(N10), repeat(N20), repeat(parameters), repeat(parameters_time), repeat(subworkdir))]
    pool.starmap(run_limit_model_per_c, inputs)
    
######## Function that runs the numerical resolution of S_0 for a given environmental speed c
def run_limit_model_per_c(c, z0, N10, N20, parameters, parameters_time, subworkdir):
    r, kappa, g, m, theta = parameters
    
    # Time parameters
    T, Nt, dt = parameters_time
    # Directory where to store the outputs
    subworkdir_per_c = subworkdir + '/c=%4.2f'%c
    create_directory(subworkdir_per_c)
    
    ## Time series to be updated
    Z, Rho = np.zeros(np.size(T)), np.zeros(np.size(T))
    N1, N2 = np.zeros(np.size(T)), np.zeros(np.size(T))
    ## Initial state
    Z[0], Rho[0], N1[0], N2[0] = z0, N20/N10, N10, N20
    
    ### Update the values of the mean trait Z, the ratio of pop size rho via the function update_limit_model (numerical resolution of S_0) next the subpop size N1 and N2
    for t in range(Nt-1):
        Z[t+1], Rho[t+1]  = update_limit_model(Z[t], Rho[t], c, parameters, dt)
        f1 = 1+g/m*(Z[t+1] + 1)**2 - 1/m
        N1[t+1] = m*(Rho[t+1] - f1)
        N2[t+1] = Rho[t+1]*N1[t+1]

    ## Store the time series
    np.save(subworkdir_per_c +'/N1.npy', N1)
    np.save(subworkdir_per_c +'/N2.npy', N2)
    np.save(subworkdir_per_c +'/Z.npy', Z)
    np.save(subworkdir_per_c +'/Rho.npy', Rho)

####### Function that implements the scheme iterations for the limit model according to a discretization of S_0
def update_limit_model(z, rho, c, parameters, dt):
    r, kappa, g, m, theta = parameters

    #### Scheme of our limit model S_0
    znew = z + dt*(-c + 2*g*((rho - 1/rho)/(rho + 1/rho) - z))
    f1new, f2new = 1 + g/m*(znew + 1)**2 - 1/m, 1 + g/m*(znew - 1)**2 - 1/m 
    P = [1., -f1new, f2new, -1]

    rhonew = np.max(np.roots(P)[np.isreal(np.roots(P))])
    return(znew, rhonew)



#### Function to build fig 4 ; assumes that the code run_model (main code) has been ran to produce the simulations whose outputs are used here.

def plot_comparison_limit_model_with_model_positive_eps(C, epsilon, Tmax_plot, parameters, parameters_time, parameters_time_eps, subworkdir, subworkdir_eps):
    #### Two different set of time parameters for simulations with eps>0 and limit eps = 0
    T, Nt, dt = parameters_time 
    T_eps, Nt_eps, dt_eps = parameters_time_eps   
    ## SYnchronize final time for both time grids
    idx_f = np.argmax(T[T<=Tmax_plot])
    idx_f_eps = np.argmax(T_eps[T_eps<=Tmax_plot])
    T = T[:idx_f]
    T_eps = T_eps[:idx_f_eps]
    
    ### Fixed parameters for the simulations
    r, kappa, g, m, theta = parameters # reproduction rate, conpetition rate, selection strength, migration rate, optimal trait
    
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (12, 18)) ## ax1 for Pop size plot, ax2 for mean trait plot
    
    ## Loop over all environmental speeds. This part assumes that the code run_model (main code) has been ran to produce the simulations whose outputs are used here.
    for i in range(np.size(C)):
        c = C[i]
        ### Load the dynamics given by the limit system (eps = 0)
        subworkdir_per_c = subworkdir + '/c=%4.2f'%c
        N1, N2, Z = np.load(subworkdir_per_c +'/N1.npy')[:idx_f], np.load(subworkdir_per_c +'/N2.npy')[:idx_f], np.load(subworkdir_per_c +'/Z.npy')[:idx_f]
        ### Load the dynamics given by the system with eps>0
        subworkdir_eps_per_c = subworkdir_eps + '/c=%4.2f'%c
        N1eps, N2eps, Zeps = np.load(subworkdir_eps_per_c +'/moments_1.npy')[0, :idx_f_eps], np.load(subworkdir_eps_per_c +'/moments_2.npy')[0, :idx_f_eps], np.load(subworkdir_eps_per_c +'/moments_1.npy')[1, :idx_f_eps]/2 + np.load(subworkdir_eps_per_c +'/moments_2.npy')[1, :idx_f_eps]/2
        
        
        color_c = plasma(int(250/np.size(C)*i))
        # Compute some additional necessary quantities related to the death valley when g>1
        if (g>=1):
            Delta = 4/g**2*(m**2-4*g*(m-1))
            z_DV = np.sqrt(1/2*(2*(1+g-m)/g - np.sqrt(Delta)))
            c_DV = 2*g*z_DV*(2/(z_DV**2+1+(m-1)/g) - 1 )
            # If the env speed is such that the pop attempts to cross the death valley, record the time of the crossing (t_DV) and display it by a horizontal line on both subplots
            if (c>=c_DV):
                idx_DV = np.argmin(Z[Z>z_DV])
                t_DV = T[idx_DV]
                ax1.axvline(t_DV, color = color_c, linestyle = 'dotted', linewidth =3)
                ax2.axvline(t_DV, color = color_c, linestyle = 'dotted', linewidth =3)
                labelc = r'$c = %4.2f'%c+'\geq c_{DV}$' # update the label for the speed to explicit the fact that it is above the threshold speed for the crossing of the valley
            else:
                labelc = '$c = %4.2f'%c+'$'
        else:
            labelc = '$c = %4.2f'%c+'$'
        
        
        ######## Display the dynamics of the metatpop size for the limit system and the system with eps >0.
        ### The limit system is constrained to have the metapopulation size positive. Record the time of extinction (t_death) and plot N1 and N2 until this time.
        t_death = T[np.argmin(N1+N2>0)]
        ax1.plot(T[T<t_death], N1[T<t_death] + N2[T<t_death], color=color_c, linewidth = 6, linestyle = 'dashed')
        ax1.plot(T, np.zeros(idx_f), color = 'black', linestyle = 'dashed') ### Display the x-axis
        
        ### The system with eps>0 does not need outside control for extinction.
        ax1.plot(T_eps, N1eps + N2eps, color=color_c, linewidth = 5, label = labelc)
        
        ## Grapĥic settings for labels and legend
        ax1.set_ylabel(r'$N(t)$', fontsize = 40)
        ax1.legend(fontsize = 30, loc = 'upper right')
        plt.setp(ax1.get_xticklabels(), fontsize=30)
        plt.setp(ax1.get_yticklabels(), fontsize=30)
        
        ######## Display the dynamics of the meant trait for the limit system and the system with eps >0.
        
        ### The limit system is constrained to have the metapopulation size positive. Plot the mean trait Z until this time.
        ax2.plot(T[T<t_death], Z[T<t_death], color=color_c, linewidth = 6, linestyle = 'dashed')
        ### The system with eps>0 can get reall yclose to extinction without getting extinct per say. However, for such values, the mean trait is ill defined.
        ### Therefore I define a cutoff on the size of the population to avoid displaying the mean trait Zeps for times where the metapop is effectively extinct.
        cutoff_size_display = 1e-100
        #If the metapop size N1eps + N2eps reaches the cut-off threshold before the end time, stop the display of Zeps at the time where the threshold is reached
        if (np.argmin(N1eps+N2eps>cutoff_size_display)>0):
            ax2.plot(T_eps[:np.argmin(N1eps+N2eps>cutoff_size_display)], Zeps[:np.argmin(N1eps+N2eps>cutoff_size_display)], color=color_c, linewidth = 3, label = '$c = %4.2f'%c+'$')
        else: ## Otherwise display Zeps for all times
            ax2.plot(T_eps, Zeps, color=color_c, linewidth = 5, label = '$c = %4.2f'%c+'$')

    ### Labels and ticks for the mean trait plot. If the death valley exists (g > 1), add the value of the mean trait corresponding to the entering edge of the valley (z_DV) tp to the ticks.
    if (g>=1):
        ax2.axhline(z_DV, color = 'chocolate', linewidth =3)
        tick_DV = [z_DV]
        ticks_old = [-2., -1.,0, 1.]
        ticklabels_old = [-2., -1., 0,  1.]
        ax2.set_yticks(ticks_old + tick_DV)
        ax2.set_yticklabels(ticklabels_old + [r'$Z_{DV}$'])
    else:
        ticks_old = [-2., -1.,0, 1.]
        ticklabels_old = [-2., -1., 0,  1.]
        ax2.set_yticks(ticks_old)
        ax2.set_yticklabels(ticklabels_old)
    ax2.set_ylabel(r'$Z(t)$', fontsize = 40)
    
    plt.xticks(fontsize = 30)
    plt.yticks(fontsize = 30)
    plt.xlabel('Time $t$', fontsize = 40)
    plt.savefig(subworkdir + "/size_and_mean_trait_comparison_Tmax=%4.1f"%Tmax_plot+".png", format='png', bbox_inches='tight') ## save the resulting figure
    plt.close()


