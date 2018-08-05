import time
import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches
from functions import *
from constants import *

def start():
    unitdimension = (4/density)**(1/3)*sigma          #Dimension of the unit cell
    a = M*unitdimension                               #Box-size
    N = 4*M**3                                        #Number of particles
    starttime = time.time()
    Tarray = np.linspace(T_sim-0.1*T_sim,T_sim+0.1*T_sim,NT)
    
    #Initialisation of time-independent variables
    U_final = np.zeros((NT,))
    T_final = np.zeros((NT,))
    pressure = np.zeros((NT,2))
    cv = np.zeros((NT,2))

    for q in range(NT):
        if NT == 1:
            T = T_sim
        else:
            T = Tarray[q]
            
        #Initialising matrices
        F_res = np.zeros((N,3,Ntimesteps),dtype = float)
        r_min = np.zeros((N,N),dtype = float)
        x = np.zeros((N,3,Ntimesteps),dtype=float)
        U_pot = np.zeros((Ntimesteps-1,),dtype = float)
        E_kin = np.zeros((Ntimesteps-1,),dtype=float)
        T_var = np.zeros((Ntimesteps-1,),dtype=float)
        v = np.zeros((N,3,Ntimesteps),dtype=float)
        virial = np.zeros((Ntimesteps-1,),dtype=float)
        dist_histogram = np.zeros((Ntimesteps,nbins-1),dtype=int)
        v[:,:,0] = particle_velocity_initialise(0,kb*T,N,3)
        x[:,:,0] = structure_function(structure,N,M,a,unitdimension)
       
        #determine wait between temperature rescaling
        avgspeed = np.sum(v[:,:,0]**2)**(1/2)/N
        freepathlength = 1/(np.pi*(2*sigma)**2*density)
        wait = 1*freepathlength/avgspeed
        k_stable = 0                                      #timestep for which the system reached equilibrium
        stabilitycheck = 0                   #Variable that turns to 1 if the temperature is stable, such that the temperature correction stops

        for k in range(Ntimesteps-1):  #loop over timesteps
            x[:,:,k+1] = new_particle_positions(x[:,:,k], v[:,:,k], delta_t, F_res[:,:,k], a)
            xmin, xdiffvec = nearest_neighbour_calculation(x[:,:,k+1],a,'x')
            ymin, ydiffvec = nearest_neighbour_calculation(x[:,:,k+1],a,'y')
            zmin, zdiffvec = nearest_neighbour_calculation(x[:,:,k+1],a,'z')
            r_min = minimum_distance(xmin,ymin,zmin)           
            
            U_pot[k], dUdr,Umatrix = energy_calculation(r_min)            
            F_res[:,:,k+1] = force_calculation(xdiffvec,ydiffvec,zdiffvec,r_min,dUdr)
            v[:,:,k+1] = new_particle_velocities(v[:,:,k], delta_t, F_res[:,:,k+1], F_res[:,:,k])
            E_kin[k],T_var[k],virial[k] = kinetic_energy_temperature(x[:,:,k],v[:,:,k],F_res[:,:,k],N,m=1,kb=1)
            
            dist_histogram[k,:] = histogram_function(r_min,a,N,nbins)
            v,stabilitycheck,k_stable = temperature_rescaling(stabilitycheck,delta_t,T_var,k,T,N,v,wait,k_stable,E_kin)

        #calculation of thermodynamic quantities
        pressure[q,:] = pressure_function(density,N,T_var[k_stable:Ntimesteps],virial[k_stable:Ntimesteps],kb=1)
        cv[q,:] = specific_heat_function(N, E_kin[k_stable:Ntimesteps])
        realised_temp = bootstrap_function(T_var[k_stable:Ntimesteps], np.mean)
        energy_per_particle = bootstrap_function(U_pot[k_stable:Ntimesteps]/N, np.mean)
        paircor = paircor_from_histograms(dist_histogram[k_stable:Ntimesteps,:], a, N)
        U_final[q] = np.mean(U_pot[Ntimesteps-int(wait/delta_t):Ntimesteps]+E_kin[Ntimesteps-int(wait/delta_t):Ntimesteps])
        T_final[q] = np.mean(T_var[Ntimesteps-int(wait/delta_t):Ntimesteps])
        
        
    fitCv, errCvfit = cv_direct_calculation(U_final,T_final,N,NT)

    endtime = time.time()
    print("Time elapsed: ", endtime - starttime)
    
    return Ntimesteps, delta_t, N, a, nbins, U_pot, E_kin, x, pressure, k_stable, T_var, cv, T_final, U_final, paircor, realised_temp, energy_per_particle, fitCv, errCvfit, M, T_sim, density 
