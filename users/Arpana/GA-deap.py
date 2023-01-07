# To use Genetic Algorithm to optmize the model parameters, first we need to define the search spaces for parameters and the numbers of generations.
# In this example progaram, we have used DEAP libary of GA.
import random
import numpy as np
from deap import base
from deap import creator
from deap import tools

# Define the search space for the parameters
param_bounds = [(10, 1000), (0.01, 1.0), (1, 10)]

# Create a DEAP toolbox with the fitness function and the desired parameters
toolbox = base.Toolbox()
toolbox.register("evaluate", fitness_function)
toolbox.register("attr_int", random.randint, 10, 1000)
toolbox.register("attr_float", random.uniform, 0.01, 1.0)
toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.attr_int, toolbox.attr_float, toolbox.attr_int), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Define the genetic operators
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

# Define the number of generations and the population size
ngen, npop = 100, 100

# Run the optimization
pop = toolbox.population(n=npop)
hof = tools.HallOfFame(1)
stats
