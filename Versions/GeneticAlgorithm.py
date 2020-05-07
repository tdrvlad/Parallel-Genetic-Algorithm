import random
import matplotlib.pyplot as plt

#Parameters Likelihood deviation from 0.5: 1 - child identical to parent a, 0 - child identical to parent b
likelihood_dev = 0.3
#Mutation range 
mutation = 0.1
generations = 100
individuals_in_generation = 100
#Number of parameters / dimension of search
parameters = 2
boundaries = ((-100,100),(-100,100))


def generate_population(size, boundaries):
    population = []
    for i in range(size):
        individual = {}
        for p in range(parameters):
            low_bound, high_bound = boundaries[p]
            individual[p] = random.uniform(low_bound,high_bound)
        population.append(individual)
    return population

import math

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

population = generate_population(size=individuals_in_generation, boundaries=boundaries)

i = 1

fitness_evolution = []
while True:
    print(" ")
    avg_fit = 0
    for individual in population:
        avg_fit += apply_function(individual)
    avg_fit = avg_fit / len(population)
    fitness_evolution.append(avg_fit)

    print(f"🧬 GENERATION {i}, average fitness {avg_fit}")

    sorted_population = sort_population_by_fitness(population)
    print("Best fit individual: ")
    print(sorted_population[-1], "(",apply_function(sorted_population[-1]),")")
    

    if i == generations:
        break

    i += 1

    population = make_next_generation(population)

plt.plot(fitness_evolution)
plt.show()
best_individual = sort_population_by_fitness(population)[-1]
print("\n🔬 FINAL RESULT")
print(best_individual, apply_function(best_individual))
