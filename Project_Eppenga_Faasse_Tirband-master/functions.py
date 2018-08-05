import time
import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches
from scipy.optimize import curve_fit
import os

def bootstrap_function(data, function, *func_args, n=1000):
    """bootstrap_function(data, function *func_args, n): 
    Description:
    ------------
    This function takes an array of data, and a function (such as np.mean, if     
    the quantity is a mean of the data), together with its function arguments to produce 
    The bootstrap function then returns the quantity and its standard deviation
    
    Parameters:
    ----------
    data: array (can be any size)
        Data from which the quantity needs to be calculated
    function: function
        Function which calculates the quantity
    func_args: can be any format, depend on the parameters of the function
        Arguments that are needed for the input function
    n: float
        Number of random sample that are picked from the data
        Default: n = 1000
    Results:
    --------
    quantity: can be anything
        The quantity calculated with the function and data
    std_quantity: float
        Standard deviation of the quantity
    """
    
    N = len(data)
    sample_ind = np.floor(np.random.rand(N,n)*N).astype(int)
    
    #take random samples of the data
    data_resample = data[sample_ind]
    
    #compute the quantity and its standard deviation
    quantity_resample = function(data_resample, *func_args, axis=0)
    std_quantity = np.std(quantity_resample)
    quantity = function(data, *func_args, axis=0)
    return quantity, std_quantity

def nearest_neighbour_calculation(x,a,axis):
    """nearest_neighbour_calculation(x,a,axis): 
    Description:
    ------------
    This function calculates the closest distance between one particle to all the other particles along one axis. The function does 
    this calculation for each particle.
    The function assumes that the particles are within a box with periodic boundary conditions. It determines the distance between one
    particle and the closet copy of another particle, along one axis (which may be a particle outside the box)
    It returns a N by N matrix (where N is the number of particles). The matrix is designed  such that element i,j is the distance
    between particle i and the closest copy of particle j. 

    
    Parameters:
    ----------
    x: array of size (N,3)
        Array containing the coordinates of the N particles.
    a: float
        Length of the box where the particles move in
    axis: string
        Axis along the nearest neighbor particles is determine (x, y, or z)
    
    Results:
    --------
    xmin: array of size(N,N)
        Array with the absolute relative distance between two particles. The i,j element is the absolute distance between particle i,j 
        (note it should be that element i,j = j,i) 
    xdiffvec: Array (N,N)
         Array with the relative distance between two particles. The i,j element is the distance between particle i,j 
        (note it should be that element i,j = -j,i)  
    """

    if axis=='x':
        g=0
    elif axis=='y':
        g=1
    elif axis=='z':
        g=2
    xdiffvec = (x[:,g] - np.transpose([x[:,g]]) + a/2)%a - a/2 #matrices containing closest relative 
    xmin = np.absolute(xdiffvec) #absolute relative x-y-z positions of other types of particles
    return xmin,xdiffvec

def minimum_distance(xmin,ymin,zmin):
    """minimum_distance(xmin,ymin,zmin): 
    Description:
    ------------
    This function calculates the minimum distance between one particles and all the other particles. The function does this calculation
    for all the particles in the system.
    
    Parameters:
    xmin: array of size(N,N)
        Array with the absolute relative distance between two particles along the x axis. The i,j element is the absolute distance 
        between particle i,j (note it should be that element i,j = j,i) 
    ymin: array of size(N,N)
        Array with the absolute relative distance between two particles along the y axis. The i,j element is the absolute distance 
        between particle i,j (note it should be that element i,j = j,i) 
    zmin: array of size(N,N)
        Array with the absolute relative distance between two particles along the z axis. The i,j element is the absolute distance 
        between particle i,j (note it should be that element i,j = j,i) 
        
    Results:
    --------
    r_min: array of size (N,N)
         Array with the minimum distance between two particles, where element [i,j] is the distance between particle i  and particle
          j
          """
    
    r_min = np.sqrt(xmin**2 + ymin**2 + zmin**2)
    N = xmin.shape                     #number of particles
    r_min = r_min + np.identity(N[1])  #add identity matrix for relative position to itself to prevent division by zero
    return r_min    

def energy_calculation(r_min,sigma=1,epsilon=1):
    """energy_calculation(r_min,sigma=1,epsilon=1): 
    Descripion:
    -----------
    This function calculates the potential energy, its derivatives to r and the total energy of the system. It calulates the potential
    energy with the lenard jones potential.
    
    Lenard-Jones potential: U(r) = 4*epsilon*((sigma/r)^6-(sigma/r)^12)
    
    Parameters:
    r_min: array of size (N,N)
        Array with the minimum distance between two particles where element i,j is the distance between particle i and particle j
    sigma: float
        Characteristic length scale for the interaction (parameter of the lenard jones potential)
        Default: sigma = 1) 
    epsilon: float
        Measure for the energy (parameter of the Lenard-Jones potential)
        Default: epsilon = 1) 
    
    Results:
    U_potk: float
        Total potential energy of the system
    dUdr: array of size (N,N)
         Array with containin the derivative of the potential with respect to r. The i,j-th element is the derivative of the potential
         energy of particle i due to particle j
    Umatrix: array of size (N,N)
         Array with i,j-th element the potential energy of particle i due to particle j
    """
    Umatrix = 4*epsilon*((sigma/r_min)**12-(sigma/r_min)**6)         
    dUdr = 4*epsilon*(-6*sigma**6/r_min**7 + 12*sigma**12/r_min**13)    
    U_potk = np.sum(Umatrix)/2          
    return U_potk, dUdr, Umatrix

def temperature_rescaling(check,delta_t,T_var,k,T,N,v,wait,k_stable,E_kin,kb=1):
    """temperature_rescaling(check,delta_t,T_var,k,T,N,v,wait,k_stable,E_kin,kb=1): 
    Description:
    ------------
    This function rescales the velocity of the particles. 
    This function is designed to force the velocity of the particles in a system to another velocity such that the system has a 
    specific energy. The function only rescales the velocity when the timestep k is an integer multiple of wait/delta_t.
    The function stops with rescaling when the mean of the temperature deviates only 2 percent of the desired temperature
    during a time interval of wait/delta_t
    
    Parameter:
    ----------
    check: float
        parameter that determines of rescaling is necessary. Only rescales the velocity if check not equal to 1
    delta_t: float
        
    T_var: float
        temperature of the system
    k: float
        Indices of the for loop over the time
    T: float
        Desired temperature
    N: float
        Number of particles
    v: array of (N,3,Ntimesteps)
        array containig the velocity in the x, y, z-direction of the N particles at all the ntimesteps
    wait: float
        
    k_stable: float
        The time step when the function stops with rescaling the energy (can be any number, the function will return the correct 
             value) 
    E_kin: float
        The kinetic energy of the system
    kb: float
        Boltzmann constant
        Default: kb=1) 
        
    Results:
    --------
    v: array of (N,3,Ntimesteps)
        Array containing the rescaled velocities in the x, y, z-direction of the N particles at all the ntimesteps
    check: float 
        Parameter that determines if further rescaling is necessary.
    k_stable: float
        The time step when the system stops with rescaling
    
    """
    if check != 1:
        if ((k % int((wait/delta_t))) == 0):
            if k > 0 :
                #checks if the average is within 2% of the desired temperature
                if np.absolute(np.mean(T_var[k-int(wait/2/delta_t):k])-T)>0.02*T:
                    labda = np.sqrt((N-1)*3*kb*T/(2*np.mean(E_kin[k-int(wait/2/delta_t):k]))) 
                    v[:,:,k+1] = labda*v[:,:,k+1]
                elif np.absolute(np.mean(T_var[k-int(wait/2/delta_t):k])-T)<=0.02*T:
                    check = check+1
                    k_stable = k
    return v,check,k_stable

def structure_function(structure,N,M,a,unitdimension):
    """structure_function(structure,N,M,a,unitdimension):
    Description:
    ------------
    This function puts the particles in the desired starting position.
    Only fcc structure is supported!
    
    Parameters:
    ------------
    structure: string
        The structure of the lattice. Enter "fcc". 
    N: float
        The number of particles in the system
    M: float
        The number of unit cells in the system along one axis
    a: float
        The length of the box
    unitdimension: float
        Length of one unit cell
   
    
    Result:
    -------
    xinit: array of size (N,N)
         Array with the initial positions of the particles the i,1 element is the x coordinate of particle i,
           i,2 element is the y coordinate of particle i, i,3 element is the y coordinate of particle i."""
    
    xinit = np.zeros((N,3),dtype=float)

    if structure == "fcc":
        #Creating the fcc lattice
        test = unitdimension*np.array([[0,0,0],[0, 0.5, 0.5], [0.5,0.5,0],[0.5,0,0.5]])
        tmp = int(np.ceil(N**(1/3)))
        tel=-4
        for i in range(M):
            for j in range(M):
                for k in range(M):
                    tel = tel+4
                    xinit[tel:tel+4,:] = test + np.array([i*unitdimension*np.ones(4,),j*unitdimension*np.ones(4,),k*unitdimension*np.ones(4,)]).T
    else:
        sys.exit("Enter valid structure")
    return xinit

def histogram_function(r_min, a, N, nbins):
    """"histogram_function(r_min, a, N, nbins) 
    Description:
    ------------
    This function makes a histogram from the relative distance matrix.
    
    Parameters:
    -----------
    r_min: array of size (N,N)
        Array with the minimum distance between two particles where element i,j is the distance between particle i and particle j
    a: float
        The length of the box
    N: float
        Number of particles
    nbins: float
        Number of bins for making the histogram.
        
    Results:
    --------
    histogram: Array of size (nbins,)
        Histogram contains the number of times a particle was between a range r and r+dr of a different particle, with dr the bin size
    """
    r_min_no_selfdistance = r_min - np.identity(len(r_min)) #remove identitity matrix so own distance will be treated as 0
    histarray = np.array(r_min_no_selfdistance).flatten()
    histogram = np.histogram(histarray,bins=np.linspace(0,a,nbins))
    histogram[0][0] = histogram[0][0] - N # remove N from the first bin to prevent it counting particle distance to itself
    return histogram[0]

def pressure_function(density,N,T_var,virial,kb=1):
    """"pressure_function(density,N,T_var,virial,kb=1):
    Description:
    ------------
    This function calculates the pressure of a system. The pressure is approximate with the viral theorem.
    The bootstrap function is used to determine the standard deviation of the pressure.
    
    The pressure is calculated with the following formula: Pressure = n/beta-n/(3N)*time_average(sum_i r_i*nabla_i*V_N(R)) (Jos Thijssen 
    computational physics equation 7.36)
    
    Parameters:
    --------------
    density: float
        The density of the system 
    N: float
        Number of particles
    T_var: array of size (1,timestep)
        Temperature of the system for each timestep
    virial: array of size (1,timestep)
        Inproduct of the particle positions with the force on the particle for each timestep
    kb: float
        Boltzmann constant
       default: kb = 1
    
    Results:
    --------
    pressure: float
        Pressure of the system
    std_pressure: float
        Standard deviation of the pressure
    """
    T = np.mean(T_var)
    pressureVar = (1+1/(3*N*kb*T)*virial)
    pressure, std_pressure = bootstrap_function(pressureVar, np.mean)
    return pressure, std_pressure

def specific_heat_dependence(E_kin, N, axis=0):
    """specific_heat_dependence(E_kin, N, axis=0):
    Description:
    -----------
    This function is used for computation of the specific heat whilst still allowing use of the bootstrap function.
    
    Parameters:
    -----------
    E_kin: array of size (1,Ntimesteps)
        Kinetic energy of the system for each time step
    N: float
        Number of particles
    Axis: float
        axis along which the mean and variance of the kinetic energy is calculated
        Default: axis =  0
    
    Results:
    ---------
    cv: Array of size (1,ntimestep)
        Specific heat of the system at each time step
    
    """
    E_kin_fluct = np.var(E_kin, axis=0)/(np.mean(E_kin, axis=0)**2)
    cv = 3/(2-3*N*E_kin_fluct)
    return cv

def specific_heat_function(N, E_kin):
    """"specific_heat_function(N,E_kin):
    Description:
    ------------
    This function calculates the specific heat of the system and its standard deviation from the fluctuations
    in the kinetic energy as in Jos Thijssen's computational physics equation 7.37. 
    
    Parameter:
    ----------
    N: float
        Number of particles
    E_kin: array of size (1,number of timestep) 
        kinetic energy of the system at each timestep
    
    Results:
    --------
    cv: float
        Specific heat of the system
    std_cv: float
        Standard deviation of the specific heat
    """
    
    cv,std_cv = bootstrap_function(E_kin,specific_heat_dependence, N)
    return cv, std_cv

def new_particle_positions(old_x, v, delta_t, F_res, a):
    """new_particle_positions(x,v,delta_t, F_res, a):
    Description:
    This function calculates the new coordinates of a particle from old coordinate, the velocity and the force on the particle
    The Verlet algorithm is used to do this calculation
    
    Parameters:
    old_x: array of size (N,3)
        Array containing the current coordinates of the N particles
    v: array of size (N,3) 
        Array containing the current velocities in the x, y, and z-direction of the N particles
    delta_t: float
        Timestep (difference in time between new coordinates and the old coordinates)
    F_res: array of size (N,3) 
        Array containing the total force in the x, y, and z-direction for the N particles
    a: float
        Length of the box where the particles move in
        
    Results:
    --------
    new_x: array of size (N,3)
        Array containing the new particle coordinates (coordinates of the particles at the current time + delta_t)
    """
    new_x = old_x + v*delta_t + delta_t**2/2*F_res     #update positions of particles
    new_x = new_x%a                                    #check if particles are inside of bounds
    return new_x

def new_particle_velocities(v_old, delta_t, F_res_new, F_res_prev):
    """new_particle_velocities(v_old, delta_t, F_res_new, F_res_prev):
    Description:
    ------------
    This function  calculates the new velocity of the particles form the old and new force on the particles
    The Verlet algorithm is used to do this calculation
    
    Parameters:
    -----------
    v_old: array of size (N,3)
        Array containing the current velocities in the x, y, and z-direction of the N particles
    delta_t: float
        Timestep (difference in time between new coordinates and the old coordinates)
    F_res_new: array of size (N,3)    
        Array containing the total force on the N particles in the x, y, and z-direction of the N particles at the 
        current time plus delta_t
    F_res_old: array of size (N,3)    
        Array containing the total force on the N particles in the x, y, and z-direction of the N particles at the 
        current time
        
    Results:
    --------
    v_new: array of size (N,3)
        Array containing the new velocities in the x, y, and z-direction of the N particles (velocities at current time + delta_t)
    """
    v_new = v_old + delta_t/2*((F_res_new+F_res_prev))
    return v_new

def force_calculation(xdiffvec,ydiffvec,zdiffvec,r_min,dUdr):
    """force_calculation(xdiffvec, ydiffvec, zdiffvec, r_min, dUdr):
    Description:
    ------------
    This function calculates the total force on the particles with the Lenard-Jones potential.
    
    Parameters:
    -----------
    xdiffvec: array of size (N,N)
        Array with the relative distance between particles in the x-direction element i,j should represent the relative distance between 
        particle i and j. (note that commonly element i,j = -j,i)
    ydiffvec: array of size (N,N)
        Array with the relative distance between particles in the y-direction element i,j should represent the relative distance between 
        particle i and j.  Element i,i should be zero! (note that ccommonly element i,j = -j,i) 
    zdiffvec: array of size (N,N)
        Array with the relative distance between particles in the z-direction element i,j should represent the relative distance between 
        particle i and j. (note that commonly element i,j = -j,i)
    r_min: array of size (N,N)
        Array containing the absolute distance between two particles. The i,j element should represent the distance between particle i
        and j. The i,i element can be any number EXCEPT ZERO! (note that commonly element i,j = j,i) 
    dUdr: array of size (N,N)
        Array containing the derivative of the potential energy with respect to r. The i,j element is the dervivative of the potential of 
        particle j has on particle i. The i,i element can be any number. (note that commonly element i,j = j,i)
        
    Results:
    --------
    F_res: array of size (N,3)
        Array containing the total force in the x, y, z-direction on the N particles
    """
    F = -dUdr/r_min*np.array([xdiffvec, ydiffvec, zdiffvec]) #NxNx3 matrix with i,j,k-th elements the force acting on particle i due to particle j in the k-th dimension
    F_res = np.sum(F,axis = 2).T 
    return F_res

def kinetic_energy_temperature(x,v,F_res,N,m=1,kb=1):
    """"kinetic_energy_temperature(x,v,F_res,N,m=1,kb=1):
    Description:
    ------------
    This function calculates the kinetic energy of system from the velocity of the particles. It also determines the temperature
    of the system. And it determines  the virial of the system (inproduct of particle coordinates with the force on the particle)
    The temperature of the system is calculated with the equipartition theorem
    
    Parameters:
    -----------
    x: array of size (N,3)
        Array with the coordinates of the N particles
    v: array of size (N,3)
        Array with the velocity in x, y, z-direction of the N particles
    N: float
        The number of particles
    m: float
        mass of the particles
        default m = 1
    kb: float
        Boltzmann constant
        default kb = 1
    
    Results:
    --------
    E_kin: float
        Kinetic energy of the system
    T_var: float
        Temperature of the system
    virial: float
        Virial of the system
    
    """
    E_kin = 1/2*m*np.sum(v**2)           #total kinetic energy of the system     
    T_var = 2/3*E_kin/(N-1)/kb           #temperature per timestep
    virial = sum(sum(x*F_res))
    return E_kin,T_var,virial

def paircor_from_histograms(dist_histogram, a, N):
    """paircor_from_histograms(dist_histogram, a, N, k_stable,Ntimesteps): 
    Description:
    ------------
    This function calculates the paircorrelation using the relative distance histograms of each timestep
    
    Parameters:
    ----------
    dist_histogram: Array of size (timesteps,nbins) 
        Here, timesteps is the number of timesteps provided, and nbins is the number of bins in the histogram  
    a: float
        Length of the box
    N: float
        the number of particles in the simulation
    
    Results:
    --------
    paircor: the pair correlation function of the system.
    """
    hist_average = np.mean(dist_histogram,axis=0)
    bins=np.linspace(0,a,len(hist_average))
    binspacing = bins[1]-bins[0]
    bins[0] = 1                          #prevent division by zero
    paircor = a**3/(N*(N-1))*hist_average/(4*np.pi*bins**2*binspacing)
    return paircor

def cv_direct_calculation(U_final,T_final,N,NT):
    """"cv_direct_calculation(U_final,T_final,N,Nt):
    Description:
    ------------
    This function calculates the specific heat out of the potential and the temperature of the system.
    
    Parameters:
    -----------
    U_final: array of size ()
        Total potential energy of the system
    T_final: array of size ()
        Temperature of the system
    N: float
        Number of particles
    NT: float
        Number of timesteps
    
    Results:
    --------
    fitCV:
    errCvfit:
    
    
    U_final is an array with the total energy of the system at a certain T
    T_final is an array with the corresponding T
    N is the number of particles in the system"""  
    
    #Define the linear dependence
    def func(x,a,b):
        return a*x+b
    
    #Only fit if NT>2
    if NT>2:
        fitCv, covCv = curve_fit(func, T_final, U_final/N)
        errCvfit = np.sqrt(np.diag(covCv))
    else:
        fitCv = np.array([0,0])
        errCvfit = 0
    return fitCv, errCvfit

def particle_velocity_initialise(mu,sigma,N,dimensions):
    """particle_velocity(mu,sigma,number_of_particles,dimensions): 
    Description:
    ------------
    This function calculates the initial velocity of particles. The velocities of the particles are distributed according to the maxwell
    distribution. The velocities are scaled such that the mean of the velocity array is equal to mu (the mean velocity)
    
    Parameters:
    -----------
    mu: float
        mean velocity of the particles
    sigma: float
        The variance of the velocities of the particles
    N: float
        The total number of particles
    dimensions: float
        The number of dimension (i.e for x, y, z velocities the dimension is 3
    
    Output:
    v: the velocities of the particles. This is an matrix of number_of_particles by dimensions"""
    
    v = np.zeros((N,dimensions))
    for tel in range(dimensions):
        v[:,tel] = np.random.normal(loc=mu, scale=np.sqrt(sigma), size=(N,))
    totalmomentum = np.sum(v,axis = 0)     
    v = v - (totalmomentum-mu)/N
    return v


def text_writer(directory,date,data_name,data,filename,save_text='on'):
    """text_writer(directory,date,data_name,data,filename):
    Description:
    ------------
    This function makes a .txt file of the data_name string and date array.
    
    Parameters:
    ------------
    directory: string
        Contains the path to where the file should be saved. When the directory does not exist the function makes a new directory.
    date: string
         Contains the date of today
    data_name: string
        Contains the name of the data 
    data: array (1,N)
            Contains the data specific by parameter data_name
    filename: string
        Contains the name the file (save name)
    save_text: string
        Specifies if text file should be saved. Options are 'on' and 'off'
        Default: save_text='on'    
    """
    if save_text=='on':   
        if len(data_name)!=len(data):
            print('number of names and number of datapoints not equal')
        else:
            if not os.path.exists(directory):
                os.makedirs(directory)
            file = open(directory+date+filename+'.txt','a')
            file.write('\n')
            file.write(directory+'\n')
            for tel in range(len(data)):
                file.write(data_name[tel]+str(data[tel])+'\n') 
            file.close() 
    return

def image_save(directory,date,image_name,file_format,save_image='on'):
    """image_save(directory,date,image_name,file_format): 
    Description:
    ------------
    This function saves an image.
    
    Parameters:
    -----------
    directory: string
       Contains the path to where the file should be saved. When the directory does not exist the function makes a new directory.
    date: string
        Contains the date of today
    image_name: string
        Contains the name the file (save name)
    file_format: string
        Contains the format of the file: for example .png
    save_image: string
        Specifies if image should be saved. Options are 'on' and 'off'
        Default: save_image='on'
    """
    
    if save_image=='on':
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.tight_layout()
        plt.savefig(directory+date+image_name+file_format)
    return





