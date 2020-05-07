#________________Necessary libraries________________

from mpi4py import MPI
import random
import matplotlib.pyplot as plt
import math
import pickle
import os
import numpy as np
import time


#________________Meta-parameters________________


#likelihood is a metric for how much a child can be similar to one of the paranets
#(0 - similar to parent a, 1 - similar to parent b. deviation is maximum 0.5 - from the middle)
max_likelihood_dev = 0.4

#mutation is a metric for randomness added to the parents' crossover
max_mutation = 100

individuals_in_generation = 40
generations = 50

#Number of parameters / dimension of search
parameters = 4
boundaries = ((-100,100),(-100,100),(-100,100),(-100,100))

#each island will have different evolutionar behaviours
likelihood_dev = random.uniform(0,max_likelihood_dev)
mutation = random.uniform(0,max_mutation)

#Island migration is the metric for individuals moving from one island habbitat to the other
migration = 30 #(percentage of individuals)

#migration chance 
migration_chance = 50 #(percentage)


#________________Function definitions________________


def generate_population(size, boundaries):
    population = []
    for i in range(size):
        individual = {}
        for p in range(parameters):
            low_bound, high_bound = boundaries[p]
            individual[p] = random.uniform(low_bound,high_bound)
        population.append(individual)
    return population


def apply_function(x):
	# Modify function to be optimised

    return x[0] ** 2 + x[1] ** 2 + x[1] * x[2] + x[3] ** 3 - math.sin(x[3] * (x[1]) + x[1] + x[3]) ** 2


def choice_by_roulette(population):
    offset = 0
    fitness_sum = sum(apply_function(individual) for individual in population)
    normalized_fitness_sum = fitness_sum

    lowest_fitness = apply_function(min(population, key = apply_function))

    if lowest_fitness < 0:
        offset = -lowest_fitness
        normalized_fitness_sum += offset * len(population)

    draw = random.uniform(0, 1)

    accumulated = 0
    for individual in population:
        fitness = apply_function(individual) + offset
        probability = fitness / normalized_fitness_sum
        accumulated += probability
        if draw < accumulated:
            # print("Selected: ",individual," score: ",apply_function(individual))
            return individual


def sort_population_by_fitness(population):
	#Function will sort population with regards to the fitness function
    return sorted(population, key=apply_function)


def crossover(individual_a, individual_b):

    child = {}
   
    for p in range(parameters):
    	param_a = individual_a[p]
    	param_b = individual_b[p]
    	likelihood = random.uniform(0.5 - likelihood_dev, 0.5 + likelihood_dev)
    	child[p] = ( likelihood * param_a + (1 - likelihood) * param_b )

    return child


def mutate(individual):

    mutant = {}
    for p in range(parameters):
        mutant[p] = individual[p] + random.uniform(-mutation, mutation)
        low_bound, high_bound = boundaries[p]
        mutant[p] = min(max(mutant[p],low_bound), high_bound)

    return mutant


def make_next_generation(previous_population):
   
    next_generation = []
    population_size = len(previous_population)
    fitness_sum = sum(apply_function(individual) for individual in previous_population)

    for i in range(population_size):
        first_choice = choice_by_roulette(previous_population)
        second_choice = choice_by_roulette(previous_population)

        individual = crossover(first_choice, second_choice)
        individual = mutate(individual)

        next_generation.append(individual)
       
    return next_generation


    population = generate_population(size=individuals_in_generation, boundaries=boundaries)

    i = 1

    fitness_evolution = []
    while True:
        avg_fit = 0
        for individual in population:
            avg_fit += apply_function(individual)
        avg_fit = avg_fit / len(population)
        fitness_evolution.append(avg_fit)

        sorted_population = sort_population_by_fitness(population)
              
        if i == generations:
            break

        i += 1

        population = make_next_generation(population)

    #from the final population, 10 of the best will be selected (population is ordered in increasing order)
    population = sort_population_by_fitness(population)[slice(-10,None)]

    #Data will be saved in a .txt file
    
    comm.send((population,fitness_evolution),dest = 0)
    

def migration_out(population):

	sorted_population = sort_population_by_fitness(population)

	no_migrated_individuals = int(migration * individuals_in_generation / 100)
	i = 0

	possible_dest = [processor for processor in range(no_processors) if processor != my_id]

	while no_migrated_individuals:
		i += 1

		if migration_chance > 100 * random.random():
			destination = random.choice(possible_dest)
			comm.send(sorted_population[-i], dest = destination)
			no_migrated_individuals -= 1

		if i == no_processors:
			break

	for processor in possible_dest:
		if processor != my_id:
			comm.send("Done", dest = processor)


def migration_in(population,source_island):
	
	while 1:
		message = comm.recv(source = source_island)

		if message != None:
			if message == "Done":
				break
			else:
				immigrant = message
				#print(f"Newcommer on island {my_id} from island {island} with fitness {apply_function(immigrant)}")
				population.append(immigrant)

	return population


def summarize(population,fitness_evolution):

	if my_id == 0:

		print("\nEvolution Summary:\n")

		final_population = []
		all_evolutions = []


	for island in range(no_processors):
		if my_id == 0 and island != 0:

			data = None

			while data == None:
				data = comm.recv(source = island)
				if data != None:

					(received_population, received_evolution,skip) = data
					final_population += received_population
					all_evolutions.append(received_evolution)
			
		else:
			
			individuals_to_send = int(individuals_in_generation / no_processors)

			#Truncating data - too large evolution history will raise errors in communication. 
			skip = int(len(fitness_evolution) / 50) + 1 
			fitness_evolution = fitness_evolution[::skip]

			selected_population = sort_population_by_fitness(population)[slice(-5,None)]
			
			comm.send((selected_population,fitness_evolution,skip), dest = 0)
				
	if my_id == 0:

		#Concatenating data from island 0
		final_population += sort_population_by_fitness(population)[slice(-5,None)]
		all_evolutions.append(fitness_evolution)

		#Printing results
		avg_fit = 0
		for individual in final_population:
			avg_fit += apply_function(individual)
		avg_fit = avg_fit / len(final_population)

		print(f"🔬 FINAL RESULT: {avg_fit}")
		
		print(f"Best {no_processors} individuals:")
		final_population = sort_population_by_fitness(final_population)
		best_individuals = final_population[slice(-no_processors,None)]
		for individual in best_individuals:
			print(individual, apply_function(individual))

		#Plotting all evolutions
		for i in range(no_processors):
			plt.plot(all_evolutions[i])
		plt.title(str(generations) + " generations with " + str(individuals_in_generation) + " individuals, migration = " + str(migration_chance * migration / 100) + "%, mutation = " + str(max_mutation)) 
		plt.xlabel("Generations (x" + str(skip) + ")")
		plt.ylabel("Value of Fitness Function")
		plt.savefig("Evo_popsize" + str(individuals_in_generation) + '_migchance' + str(migration_chance * migration / 100) + "_maxmutation" + str(max_mutation) + ".png",dpi = 500)
		plt.show()
		
	

#________________Run________________   


#Initialize population
population = generate_population(size=individuals_in_generation, boundaries=boundaries)

#Initialise MPI communication
comm = MPI.COMM_WORLD
my_id = comm.Get_rank()
no_processors = comm.Get_size()
	
fitness_evolution = []

i = 1
while True:

	if i == generations:
		break

	if i % 5 == 0 and my_id == 0:
		print(f"Generation {i}")
	i += 1

	#Simulating migartion - each island at a time sends some individuals and the rest listen for newcommers
	for island in range(no_processors):
		if my_id == island:
			#print(f"Migration from island{my_id}")
			migration_out(population)
		else:
			#print(f"Island {my_id} waiting visitors")
			migration_in(population,island)

	avg_fit = 0
	for individual in population:
		avg_fit += apply_function(individual)
	avg_fit = avg_fit / len(population)
	fitness_evolution.append(avg_fit)

	population = make_next_generation(population)

#Getting the best individuals from all the islands (on arbitrarly island 0)
summarize(population,fitness_evolution)



