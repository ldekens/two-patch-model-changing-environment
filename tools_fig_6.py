#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 16:40:31 2023

@author: dekens
"""


import numpy as np
import numpy.random as nprand
import shutil
import os
import multiprocessing
from itertools import repeat
import matplotlib.pyplot as plt
import matplotlib as ml
from matplotlib import cm

ml.rcParams['mathtext.fontset'] = 'stix'
ml.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({
    "text.usetex": True})
ml.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}' #for \text command
plasma = cm.get_cmap('plasma', 300)
viridis = cm.get_cmap('viridis', 300)
inferno = cm.get_cmap('inferno', 300)

def reproduction(P, eps, tau, dt): ### P is the trait vector of the focal population, tau is the reproduction rate, dt is the generation time and eps**2/2 is the segregational within family-varaince
    N = np.size(P) ### number of individuals of the population
    U = nprand.uniform(size = N)
    Mates1 = P[np.nonzero(U <= tau*dt)] #### draw approx tau * N individuals that will choose a mate to reproduce
    Mates2 = nprand.choice(P, size = np.size(Mates1), replace = False)
    Offspring = nprand.normal(loc = (Mates1 + Mates2)/2, scale = eps/np.sqrt(2))
    P_post_reprod = np.append(P, Offspring)
    return(P_post_reprod)

def selection_competition(P, g, theta, K, dt): ### g*dt is the selection rate, theta is the optimal trait, K is the carrying capacity
    N = np.size(P) ### number of individuals of the population
    Prob_life = np.array(np.exp(-g*dt*(P - theta)**2) * np.exp( - N*dt/K))  ### probability of surviving with trait p in population of size N
    U = np.array(nprand.uniform(size = N))
    P_survivors = P[np.nonzero(U<=Prob_life)]  ### survivors are those who succeeded the trial according to their trait
    return(P_survivors)

def migration(P1, P2, m, dt): ### exchange approx m*dt migrants between P1 and P2
    N1, N2 = np.size(P1), np.size(P2)
    nMigrants1, nMigrants2 = min(nprand.poisson(m*dt*N1), N1), min(nprand.poisson(m*dt*N2), N2) ### draw number of migrants according to Poisson of paramter m*dt (independently per pop)
    idx_Migrants1, idx_Migrants2 = nprand.choice(N1, size = nMigrants1, replace = False), nprand.choice(N2, size = nMigrants2, replace = False)
    mask1, mask2  = np.ones(len(P1), dtype = bool), np.ones(len(P2), dtype = bool)
    mask1[idx_Migrants1], mask2[idx_Migrants2] = False, False
    P1_stay, P2_stay = P1[mask1], P2[mask2]
    #print(nMigrants1)
    P1_post_mig = np.append(P1_stay, P2[idx_Migrants2]) ### replace migrants 1 by migrants 2 in pop1
    P2_post_mig = np.append(P2_stay, P1[idx_Migrants1]) ### replace migrants 2 by migrants 1 in pop2
    return(P1_post_mig, P2_post_mig)

def generation(P1, P2, parameters, count_gen, dt): #### update the state of the two populations through one generational life cycle of time duration dt
    eps, tau, g, theta1, theta2, K, m, c = parameters
    P1_post_reprod = reproduction(P1, eps, tau, dt)
    P2_post_reprod = reproduction(P2, eps, tau, dt)
    P1_post_selection_comp = selection_competition(P1_post_reprod, g, theta1 + eps**2*c*dt*count_gen, K, dt)
    P2_post_selection_comp = selection_competition(P2_post_reprod, g, theta2 + eps**2*c*dt*count_gen, K, dt)
    P1_post_mig, P2_post_mig = migration(P1_post_selection_comp, P2_post_selection_comp, m, dt)
    return(P1_post_mig, P2_post_mig)
    

def initial_specialist_state(parameters): ### this function returns the initial state of the system
#### To do so, it computes the analytical initial specialist state from Proposition 4.2 in [Dekens 2022]
#### It assumes that the parameters are rescaled, so that r = kappa = 1.
    eps, tau, g, theta1, theta2, K, m, c = parameters
    if (1+2*m > 5*g):
        raise Exception('No specialist equilibira can exist under these parameters.')
    else:
        S = [1., (1-4*g)/m, -4*g/m, 4*g/m]
        y = np.max(np.roots(S)[np.isreal(np.roots(S))])
        rho = (y + np.sqrt(y**2-4))/2
        N10, N20, Z0 = (1 - m) + m*rho -4*g*rho**4/(rho**2 + 1)**2, (1 - m) + m/rho -4*g/(rho**2 + 1)**2, (rho**2 - 1)/(rho**2 + 1)
    #### Note that N10 and N20 are between 0 and 1 (proportion of carrying capacity)
    #### generate two populations (Gaussian vectors of trait) of size N10*K et N20*K and mean Z0, variance eps**2 
    P10, P20 = nprand.normal(loc = Z0, scale = eps, size = int(np.floor(N10*K))), nprand.normal(loc = Z0, scale = eps, size = int(np.floor(N20*K)))
    return(P10, P20)

def run_IBS(parameters, parameters_time, workdir):
    
    Ngen_sim, Ngen_burnin, dt = parameters_time
    #### 1. Generate initial specialist state
    P10, P20 = initial_specialist_state(parameters)
    ### 2. Run Ngen_burnin generations of burn-in with stable environment and tore end of burn-in state
    P1, P2 = P10, P20
    count_gen = 0 ## count the number of generation of the simulation (not burn-in) to update the optimal traits
    count_ext = 0 ## count the number of generation of the simulation (not burn-in) to update the optimal traits

    for gen in range(Ngen_burnin):
        P1, P2 = generation(P1, P2, parameters, count_gen, dt)
    np.save(workdir +'/P1_0.npy', P1)
    np.save(workdir +'/P2_0.npy', P2)
    
    ### 3. Run Ngen_sim generations with changing environement and save each generation
    
    moments = np.empty((Ngen_sim, 4))
    while (count_gen<Ngen_sim)&(count_ext<2):
        moments[count_gen, :] = [np.size(P1), np.size(P2), np.mean(P1), np.mean(P2)]
        count_gen = count_gen + 1
        P1, P2 = generation(P1, P2, parameters, count_gen, dt)
        if (np.size(P1) + np.size(P2)==0):
            count_ext = count_ext +1
            moments[count_gen:, 0] = np.ones(Ngen_sim - count_gen)*np.size(P1)
            moments[count_gen:, 1] = np.ones(Ngen_sim - count_gen)*np.size(P2)
            moments[count_gen:, 2] = np.ones(Ngen_sim - count_gen)*np.mean(P1)
            moments[count_gen:, 3] = np.ones(Ngen_sim - count_gen)*np.mean(P2)
            
    ### Save final state
    np.save(workdir +'/P1_final.npy', P1)
    np.save(workdir +'/P2_final.npy', P2)
    np.save(workdir + '/moments_all_times.npy', moments)
    N1f, N2f = np.size(P1), np.size(P2)
    z1f, z2f = np.mean(P1), np.mean(P2)
    np.save(workdir +'/moments_final.npy', np.array([N1f, N2f, z1f, z2f]))
    print([N1f, N2f, z1f, z2f])
    return()

def run_replicate_IBS(Nreplicates, parameters, parameters_time, workdir):
    ### Run replicates in parallel
    if __name__ == 'tools_test_IBS_implicit_architecture':
        pool = multiprocessing.Pool(processes = 15)
        print(Nreplicates)
        inputs = [*zip(range(Nreplicates), repeat(parameters), repeat(parameters_time), repeat(workdir))]
        pool.starmap(run_per_replicate, inputs)
        pool.close()
    Ngen_sim, Ngen_burnin, dt = parameters_time
    moments_all = np.zeros((Nreplicates, 4))
    moments_all_all_times = np.zeros((Nreplicates, Ngen_sim, 4))
    ##### collect and save the outcomes of each replicates in a single file
    for i in range(Nreplicates):
        #run_per_replicate(i, parameters, parameters_time, workdir) ## alternative loop instead of parallel run
        moments_all[i, :] = collect_final_moments_per_simulation(i, workdir)
        moments_all_all_times[i, :, :] = collect_moments_all_times_per_simulation(i, workdir)
    np.save(workdir +'/moments_all.npy', moments_all)
    np.save(workdir +'/moments_all_all_times.npy', moments_all_all_times)
    
    return()

#### auxiliary function to run per replicate
def run_per_replicate(i, parameters, parameters_time, workdir):
    nprand.seed()
    workdir_current = workdir + '/simulation_%i'%i
    create_directory(workdir_current, False)
    run_IBS(parameters, parameters_time, workdir_current)
    return()


def collect_final_moments_per_simulation(i, workdir):
    workdir_current = workdir + '/simulation_%i'%i

    moments = np.load(workdir_current +'/moments_final.npy')
    return(moments)

def collect_moments_all_times_per_simulation(i, workdir):
    workdir_current = workdir + '/simulation_%i'%i

    moments = np.load(workdir_current +'/moments_all_times.npy')
    return(moments)


### Selectio gradient under stable environment
def F(z, rho, g):
    return(2*g*((rho**2-1)/(rho**2+1) - z))
#### Auxiliary functions
def search_positive_roots(P): ## P a polynomial given by its list of coefficients
    roots = np.roots(P)
    roots = roots[np.real(np.isreal(roots))]
    roots = roots[roots>0]
    return(roots)

def find_rho_star(z, parameters):
    eps, tau, g, theta1, theta2, K, m, c = parameters
    theta = (theta2 - theta1)/2
    P = [1, 1/m - 1 - g/m*(z + theta)**2, -1/m + 1 + g/m*(z -theta)**2, -1]
    roots = search_positive_roots(P)
    return(roots)

### Build the graph of the selection gradient
def curve_per_parameter(parameters, parameters_trait_space): 
    eps, tau, g, theta1, theta2, K, m, c  = parameters
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

def plot_end_size_per_c_comparison_analytical(g, C, parameters, parameters_time, subworkdir):
    eps, tau, g, theta1, theta2, K, m, c = parameters
    
    ##### Get the end metapop size and local mean traits from replicates IBS
    Ngen_sim, Ngen_burnin, dt = parameters_time
    N1, N2 = np.zeros(np.size(C)), np.zeros(np.size(C))
    N1_persist, N2_persist = np.zeros(np.size(C)), np.zeros(np.size(C))

    N1_8, N2_8 = np.zeros(np.size(C)), np.zeros(np.size(C))
    N1_2, N2_2 = np.zeros(np.size(C)), np.zeros(np.size(C))
    mean1, mean2 = np.zeros(np.size(C)), np.zeros(np.size(C))

    mean1_8, mean2_8 = np.zeros(np.size(C)), np.zeros(np.size(C))
    mean1_2, mean2_2 = np.zeros(np.size(C)), np.zeros(np.size(C))
    for i in range(np.size(C)):
        c = C[i]
        Lag = eps**2*c*dt*Ngen_sim
        moments_all = np.load(subworkdir +'/c=%4.2f'%c+'/moments_all.npy')
        N1_all, N2_all, mean1_all, mean2_all = moments_all[:, 0], moments_all[:, 1], moments_all[:, 2], moments_all[:, 3]
        N1[i], N2[i] = np.quantile(N1_all, 0.5, axis = 0)/K, np.quantile(N2_all, 0.5, axis = 0)/K
        if (np.size(N1_all[N1_all>0])>0):
            N1_persist[i] = np.quantile(N1_all[N1_all>0], 0.5, axis = 0)/K
        if (np.size(N2_all[N2_all>0])>0):
            N2_persist[i] = np.quantile(N2_all[N2_all>0], 0.5, axis = 0)/K


        N1_8[i], N2_8[i] = np.quantile(N1_all, .99, axis = 0)/K, np.quantile(N2_all, .99, axis = 0)/K
        N1_2[i], N2_2[i] = np.nanquantile(N1_all, 0.01, axis = 0)/K, np.nanquantile(N2_all, 0.01, axis = 0)/K
        mean1[i], mean2[i] = np.nanquantile(mean1_all- Lag, 0.5, axis = 0), np.nanquantile(mean2_all- Lag, 0.5, axis = 0)
        mean1_8[i], mean2_8[i] = np.nanquantile(mean1_all- Lag, .99, axis = 0), np.nanquantile(mean2_all- Lag, .99, axis = 0)
        mean1_2[i], mean2_2[i] = np.nanquantile(mean1_all- Lag, 0.01, axis = 0), np.nanquantile(mean2_all- Lag, 0.01, axis = 0)

        
    #### Compute the analytical predictions for metapop size and mean trait at equilibrium
    ## trait space
    zmin, zmax, Nz =  -2.5, 1.2, 551
    z, dz = np.linspace(zmin, zmax, Nz, retstep=True)
    parameters_trait_space_bis = zmin, zmax, np.linspace(zmin, zmax, Nz*10), Nz*10, dz/10
    
    #### Theoretical size at equilibrium depending on speed in Cbis
    Ncbis = 300
    Cbis = np.linspace(0, np.max(C), num = Ncbis)
    z_c, rho_c, size_c, f1 = np.zeros(Ncbis), np.zeros(Ncbis), np.zeros(Ncbis), np.zeros(Ncbis)
    z_star, rho_star, F_z_star = curve_per_parameter(parameters, parameters_trait_space_bis)
    z_star_C, rho_star, F_z_star = curve_per_parameter(parameters, parameters_trait_space_bis)
    for i in range(Ncbis):
        c = Cbis[i]
        z_c[i] = z_star[-np.argmin(F_z_star[::-1]<c)]
        rho_c[i] = rho_star[-np.argmin(F_z_star[::-1]<c)]
        f1[i] = 1 + g/m*(z_c[i]+1)**2 - 1/m
        size_c[i] = max(m*(rho_c[i] - f1[i])*(1 + rho_c[i]), 0)

    c_switch = np.max(F_z_star[z_star>=0])
    z_star_bis = z_star[z_star >0]
    z_switch = z_star_bis[np.argmax(F_z_star[z_star>=0])]
    z_post_switch = z_star[- np.argmax([F_z_star[::-1]>c_switch])]
    
    ##### Plot of mean trait
    plt.figure(figsize=(11,6))
    plt.xlabel('Environmental speed $c$', fontsize = 40)
    plt.ylabel('$Z^*$', fontsize =40)
    
    color = (z_c - min(z_c))/(np.max(z_c) - np.min(z_c))
    plt.set_cmap(viridis)
    
    ## Plot of analytical eq
    plt.scatter(Cbis, z_c, c = color, vmin = -10/19, vmax = 1+3/19, s = 40)
    ## Plot of IBS eq
    CI_Z = np.zeros((2, np.size(C)))
    CI_Z[0, :] = mean1/2 + mean2/2 -(mean1_2/2 + mean2_2/2) ### lower endof confidence interval
    CI_Z[1, :] = (mean1_8/2 + mean2_8/2) - (mean1/2 + mean2/2)
    Nc = np.size(C)
    z_C = np.zeros(Nc)
    z_star_C, rho_star_C, F_z_star_C = curve_per_parameter(parameters, parameters_trait_space_bis)
    for i in range(Nc):
        c = C[i]
        z_C[i] = z_star[-np.argmin(F_z_star_C[::-1]<c)]
    plt.scatter(C, (mean1 + mean2)/2, c = 'black', marker = 's', s= 250)
    plt.errorbar(C, (mean1 + mean2)/2, ecolor = 'black', yerr= CI_Z, elinewidth=3, marker = 'o', fmt ='none')
    ## Additional features of the plot to make it palatable
    plt.xticks([], fontsize = 40)
    plt.yticks([-1, 1], [r'$\theta_1(t)$', r'$\theta_2(t)$'], fontsize = 30)
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
        plt.text(c_switch - 0.5, z_post_switch/2 + (max(z_c)-3)/2 , 'Refugium',fontsize = 30)

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
    ## save plot
    plt.savefig(subworkdir + '/mean_at_eq_g=%4.2f'%g+'.png', bbox_inches='tight')
    #plt.show()
    plt.close()
    
    ######## Plot of metapopulation size at equilibrium
    plt.figure(figsize=(11,6))
    plt.xlabel('Environmental speed $c$', fontsize = 40)
    plt.ylabel('$N^*$', fontsize = 40)
    plt.set_cmap(viridis)
    
    ## Analytical eq
    plt.scatter(Cbis, size_c, c = color, vmin = -10/19, vmax = 1+3/19, s=40)
    
    ## IBS eq
    CI_N = np.zeros((2, np.size(C)))
    CI_N[0, :] = N1 + N2 -(N1_2 + N2_2) ### lower endof confidence interval
    CI_N[1, :] = (N1_8 + N2_8) - (N1 + N2)
    print(N1 + N2)
    plt.scatter(C, N1 + N2, c = 'black', marker = 's', s= 250)
    plt.scatter(C, N1_persist + N2_persist, c = 'black',  marker = 'o', s= 250)

    plt.errorbar(C, N1 + N2, ecolor = 'black', yerr= CI_N, elinewidth=3, marker = 'o', fmt ='none')
    plt.hlines(0, 0, max(C), color = 'black')
    
    ## Additional features to make the plot more palatable
    plt.vlines(c_switch, - 0.05,0.05, color = 'RoyalBlue', linewidth = 6)
    Delta = 4/g**2*(m**2-4*g*(m-1))
    z_dp = -np.sqrt(1/2*(2*(1+g-m)/g + np.sqrt(Delta)))
    c_dp = -2*g*z_dp*(1 - 2/(z_dp**2+1+(m-1)/g) )
    plt.vlines(c_dp, - 0.05,0.05, color = 'darkred', linewidth = 6)
    if (g>1):
        z_dv = np.sqrt(1/2*(2*(1+g-m)/g - np.sqrt(Delta)))
        c_dv = 2*g*z_dv*(-1 + 2/(z_dv**2+1+(m-1)/g) )
        plt.vlines(c_dv, - 0.05,0.05, color = 'chocolate', linewidth = 6)
    c_crit = Cbis[-np.argmax(size_c[::-1]>0)]
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
    ## save plot
    plt.savefig(subworkdir + '/size_at_eq_g=%4.2f'%g+'.png', bbox_inches='tight')
    #plt.show()
    plt.close()
    return()
    

def create_directory(workdir, remove):
    if (os.path.exists(workdir))&remove:
        shutil.rmtree(workdir)
    try:
    # Create target Directory
        os.mkdir(workdir)
    except FileExistsError:
        print("Directory " , workdir ,  " already exists")
    return
