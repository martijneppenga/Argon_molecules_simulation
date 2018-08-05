import numpy as np
import matplotlib.pyplot as plt
import datetime
from mpl_toolkits.mplot3d import Axes3D
from constants import directory, NT, save_image, save_thermodynamic_values as save_text
from functions import text_writer, image_save






def start(Ntimesteps, delta_t, N, a, nbins, U_pot, E_kin, x, pressure, k_stable, T_var, cv, T_final, U_final, paircor, realised_temp, energy_per_particle, fitCv, errCvfit, M, T_sim, density):
    
    #for saving files
    today = datetime.date.today()
    date = '/'+str(today)+'_'
    file_format = '.png'
   
    #for plot font
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('font', size=18)
    
    #Energy conservation
    plt.figure(figsize=(6,6))
    plt.plot(np.linspace(0,Ntimesteps-1,Ntimesteps-1)*delta_t,E_kin, label='Kinetic Energy')
    plt.plot(np.linspace(0,Ntimesteps-1,Ntimesteps-1)*delta_t,U_pot, label='Potential Energy')
    plt.plot(np.linspace(0,Ntimesteps-1,Ntimesteps-1)*delta_t,E_kin + U_pot, label='Total Energy')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlabel('Time ($(m\sigma^2/\epsilon)^{1/2}$)')
    plt.ylabel('Energy/$\epsilon$')
    plt.title('Plot of the Energies')
    image_save(directory,date,'Plot_of_the_Energies',file_format,save_image)
    plt.show()

    #start positions of atoms
    plt.figure(2)
    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection='3d')

    ax = fig.add_subplot(111, projection='3d')
    for i in range(N):
       ax.scatter(x[i,0,0],x[i,1,0],x[i,2,0],'.')
    plt.title('Start positions Argon Atoms')
    image_save(directory,date,'Start_positions_Argon_Atoms',file_format,save_image)
    plt.show()

    #plot of the temperature vs timesteps
    plt.figure(3)
    plt.plot(T_var, label='Simulated temperature')
    plt.plot(T_sim*np.ones(Ntimesteps,),label='set Temperature')
    plt.title('Temperature vs time')
    plt.legend(loc=4)
    plt.xlabel('Timestep')
    plt.ylabel('Scaled Temperature ($\epsilon/k_b$)')
    plt.ylim([0,T_sim*1.5])
    image_save(directory,date,'Temperture_of_system',file_format,save_image)
    plt.show()

    
    #print some quantities
    print("Measured pressure and its uncertainty :", pressure)
    print("kstable :", k_stable)
    print("Energy per particle and its uncertainty", energy_per_particle)
    print("Temperature when stabilised and its uncertainty", realised_temp)
    print("Calculated density:", N/a**3)
    print("Calculated specific heat and its uncertainty: ", cv)





    plt.figure(4)
    #Plot the pair-correleation function
    paircorx = np.linspace(0,a,nbins-1)
    plt.plot(paircorx,paircor)
    plt.xlim([0,a/2])
    plt.title('Pair-correleation')
    image_save(directory,date,'Pair-correleation',file_format,save_image)
    plt.show()


    if NT>2:
        #plot pressure vs temperature
        plt.figure(5)
        plt.plot(T_final,pressure[:,0])
        plt.title('Pressure vs temperature')
        plt.xlabel('Temperature ($\epsilon/k_b$)')
        plt.ylabel('$\beta P/n$')
        image_save(directory,date,'Pressure_vs_temperature',file_format,save_image)
        plt.show()

        #plot Cv vs temperature
        plt.figure(6)
        plt.plot(T_final,cv[:,0])
        plt.title('Cv vs temperature')
        plt.xlabel('Temperature ($\epsilon/k_b$)')
        plt.ylabel('CV ($k_b$)')
        plt.show()

        plt.figure(7)
        #plot energy vs temperature
        plt.plot(T_final,U_final/N,'.', label = 'Simulated data')
        plt.title('Energy vs temperature')
        plt.xlabel('Temperature ($\epsilon/k_b$)')
        plt.ylabel('Energy ($\epsilon$)')
        plt.plot(Tfinal,fitCv[0]*Tfinal + fitCv[1], label='Linear Fit')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        image_save(directory,date,'Energy_vs_temperature',file_format,save_image)
        plt.show()
        print("Looped CV =", fitCv[0], "error: ", errCvfit[0])
    
    #save some data
    data = [pressure,k_stable,energy_per_particle,realised_temp,N/a**3,cv]
    data_name = ["Measured pressure and its uncertainty: ","kstable: ",'Energy per particle and its uncertainty: ',
             'Temperature when stabilised and its uncertainty: ','calculated density: ',
            "Calculated specific heat and its uncertainty: "]
    text_writer(directory,date,data_name,data,'data_file',save_text)

    
    

    