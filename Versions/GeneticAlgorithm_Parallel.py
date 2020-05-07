from mpi4py import MPI
import random
import matplotlib.pyplot as plt
import math
import pickle
import os
import numpy as np
import time


#Initialise communication between processors
comm = MPI.COMM_WORLD
my_id = comm.Get_rank()
processors = comm.Get_size()
"""Processor 0 will act as a main, concatinating the results of the genethic algorithms computed
by the rest of the processors."""

#likelihood is a metric for how much a child can be similar to one of the paranets
#(0 - similar to parent a, 1- similar to parent b)
likelihood_dev = 0.1
max_likelihood_dev = 0.5

#mutation is a metric for randomness added to the parents' crossover
mutation = 0.1
max_mutation = 0.5

individuals_in_generation = 100
generations = 50

#Number of parameters / dimension of search
parameters = 2
boundaries = ((-100,100),(-100,100))

#each island will have different evolutionar behaviours
if my_id != 0 :
    likelihood_dev = random.uniform(0,max_likelihood_dev)
    mutation = random.uniform(0,max_mutation)

    file_name = "Island_" + str(my_id) +".txt"
    if os.path.exists(file_name):
        os.remove(file_name)

def generate_population(size, boundaries):
    population = []
    for i in range(size):
        individual = {}
        for p in range(parameters):
            low_bound, high_bound = boundaries[p]
            individual[p] = random.uniform(low_bound,high_bound)
        population.append(individual)
    return population

def apply_function(individual):
    return math.sin(math.sqrt(individual[0] ** 2 + individual[1] ** 2))

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

if my_id != 0:

    population = generate_population(size=individuals_in_generation, boundaries=boundaries)

    i = 1

    fitness_evolution = []
    while True:
        avg_fit = 0
        for individual in population:
            avg_fit += apply_function(individual)
        avg_fit = avg_fit / len(population)
        fitness_evolution.append(avg_fit)

        #print(f"🧬 GENERATION {i}, average fitness {avg_fit}")

        sorted_population = sort_population_by_fitness(population)
        #print("Best fit individual: ")
        #print(sorted_population[-1], "(",apply_function(sorted_population[-1]),")")
        
        if i == generations:
            break

        i += 1

        population = make_next_generation(population)

    best_individual = sort_population_by_fitness(population)[-1]
    #print(f"ISLAND {my_id} - 🔬 FINAL RESULT")
    #print(best_individual, apply_function(best_individual))
    population = sort_population_by_fitness(population)[slice(-10,None)]
    file_name = "Island_" + str(my_id) +".txt"
    with open(file_name,"wb") as file:
    	pickle.dump((population,fitness_evolution),file)

else:
    hash_received = np.zeros(processors)
    fitness_evolutions = []
    population = []

    while sum(hash_received) < processors - 1 :

        for i in range(1,processors):
            file_name = file_name = "Island_" + str(i) +".txt"
            
            if os.path.exists(file_name):
                
                if os.path.getsize(file_name) > 0 and hash_received[i] == 0:
                    
                    with open(file_name,"rb") as file:

	                    hash_received[i] = 1

	                    (individuals,evolution) = pickle.load(file)
	                    population += individuals
	                    fitness_evolutions.append(evolution)

	                    avg_fit = 0
	                    for individual in individuals:
	                        avg_fit += apply_function(individual)
	                    avg_fit = avg_fit / len(individuals)
	                    print(f"Island {i} produced average score {avg_fit} ({len(individuals)} indivuduals)")

    for i in range(1,processors):
    	file_name = file_name = "Island_" + str(i) +".txt"
    	if os.path.exists(file_name):
    		os.remove(file_name)
    
    print("Evolving selected individuals")      
    

    second_phase_fitness_evolution = []

    i = 1
    while True:
       
        avg_fit = 0
        for individual in population:
            avg_fit += apply_function(individual)
        avg_fit = avg_fit / len(population)
        second_phase_fitness_evolution.append(avg_fit)

        #print(f"🧬 GENERATION {i}, average fitness {avg_fit}")

        #sorted_population = sort_population_by_fitness(population)
        #print("Best fit individual: ")
        #print(sorted_population[-1], "(",apply_function(sorted_population[-1]),")")
        
        if i == generations:
            break

        i += 1

        population = make_next_generation(population)

    for i in range(0,processors - 1):
    	fitness_evolutions[i] += second_phase_fitness_evolution
    	plt.plot(fitness_evolutions[i])
    plt.show()

    best_individual = sort_population_by_fitness(population)[-1]
    print("\n🔬 FINAL RESULT")
    print(best_individual, apply_function(best_individual))


