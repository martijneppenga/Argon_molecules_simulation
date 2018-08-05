import simulation
import data_processing
results = simulation.start()
data_processing.start(*results)
