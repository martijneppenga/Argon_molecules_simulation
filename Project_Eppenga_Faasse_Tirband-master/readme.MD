# Simulation of argon molecules 

This first project for the computational physics course comprises the simulation of argon molecules. This simulation uses Verlet's algorithm and periodic boundary conditions to simulate the movement of particles. 
Thermodynamic quantities such as specific heat, pressure, and pair correlation function are obtained from this simulation.

## Getting Started
In order to run the simulation, copy all the .py files in the repository to your working directory. Change the input constants in constants.py, save the file, then start the simulation from the simulation module, and use the results to do the processing, as follows:

```
import simulation
import data_processing
results = simulation.start()
data_processing.start(*results)
```

Or, simply run launcher.py which does all of the above steps.

### Options for constants

The following inputs can be adjusted in constants.py

Constants related for atom properties (leave all to one to use dimensionless units)
* **m** mass of the particles
* **kb** Boltzmann constant
* **epsilon** energy scale for the system
* **sigma** characteristic length scale of the system

Parameters of the system
* **T_sim** the temperature of the system
* **density** the density of the system 
* **M** the number of conventional FCC unit cells in the box
* **Ntimesteps** the number of timesteps for the simulation
* **NT** The number of different temperatures to run the script for, if desired

Save of images and thermodynamic properties 
* **save_image** Switch to save images (on or off)
* **save_thermodynamic_values** Switch to save thermodynamic properties (on or off)
* **directory** Folders where the images and thermodynamic properties are saved


### Obtained results
The output contains, apart from parameters used for checks:
* **x** matrix containing the locations of each particle for each timestep
* **E_kin** the total kinetic energy of the system for each timestep
* **pressure** the pressure of the system
* **k_stable** the timestep at which the system became stable
* **T_var** the temperature for each timestep
* **cv** the specific heat of the system
* **Paircor** the pair correlation function
* **realised_temp** the realised temperature of the system


## Authors
* Martijn Eppenga
* Richard Faasse
* Cyrus Tirband 

## Report
The report for this project can be found in the repository as well.
