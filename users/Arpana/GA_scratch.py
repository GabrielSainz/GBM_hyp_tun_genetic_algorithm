import numpy as np

# Define the fitness function
def fitness_function(params):
    n_estimators, learning_rate, max_depth = params
    model = GradientBoostingModel(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return -mse  # We need to return a negative value because the GA maximizes the fitness function

# Define the search space for the parameters
param_bounds = [(10, 1000), (0.01, 1.0), (1, 10)]

# Define the number of generations and the population size
ngen, npop = 100, 100

# Initialize the population with random individuals
pop = []
for i in range(npop):
    individual = []
    for low, high in param_bounds:
        individual.append(random.uniform(low, high))
    pop.append(individual)

# Run the optimization loop
for i in range(ngen):
    # Evaluate the fitness of each individual
    fitnesses = [fitness_function(individual) for individual in pop]
    
    # Select the fittest individuals for reproduction
    fittest_indexes = np.argsort(fitnesses)[::-1][:npop//2]
    fittest = [pop[i] for i in fittest_indexes]
    
    # Reproduce the fittest individuals to generate the next generation
    next_gen = []
    for i in range(npop//2):
        parent1 = random.choice(fittest)
        parent2 = random.choice(fittest)
        child1, child2 = crossover(parent1, parent2)
        child1 = mutate(child1)
        child2 = mutate(child2)
        next_gen.append(child1)
        next_gen.append(child2)
    
    # Replace the current population with the next generation
    pop = next_gen

# Get the best individual
best_index = np.argmax(fitnesses)
best_individual = pop[best_index]
