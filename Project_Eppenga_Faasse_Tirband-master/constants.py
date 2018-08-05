import numpy as np

#To make the simulation dimensionless, these 4 constants are 1. This can be changed to SI-units by filling in the correct values for these constants
m = 1                                #Mass
kb = 1                               #scaled Boltzman constant to epsilon
epsilon = 1                          #Energy scale for the system
sigma = 1                            #Characteristic length scale of the system

#Input parameters:
#-----------------
M = 3                                #number of unit cells fcc in one dimension
density = 1                          #number of particles per unit volume of sigma^3 
T_sim = 1                            #Temperature
NT = 1                               #Number of loops (for different temperatures)
Ntimesteps = 3000                    #Number of timesteps
nbins = 100                          #bins for histogram for pair correlation
delta_t = np.sqrt(m/epsilon)*4e-3    #length of one timestep in dimensionless units
structure = "fcc"                    #only fcc is supported 

# Saving of results:
#-------------------
save_image = 'off'                   #Enter 'on' or 'off' depending on if you want to save images
save_thermodynamic_values = 'off'    #Enter 'on' or 'off' depending on if you want to save the data
directory = 'Images_and_data/number_of_unit_cells_'+str(M)+'/temperature_'+str(T_sim)+'/density'+str(density)



