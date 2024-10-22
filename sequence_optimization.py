import numpy as np
import deap
import itertools

import random
from deap import base, creator, tools, algorithms
from tensorflow.keras.models import load_model #type:ignore


def random_nucleotide():
    i = np.random.choice(4)
    nucleotides = np.array(list(set(itertools.permutations([1, 0, 0, 0]))))
    return nucleotides[i]


# Define the fitness function
def evaluate_test(individual):
    sum = 0
    weights = np.array([1, 0., 0., 0.])
    for nucleotide in individual:
        res = weights * nucleotide
        sum += np.sum(res)
    return (sum,)


def evaluate_model(individual, model):
    print(individual.shape)
    expanded_individual = np.expand_dims(individual, axis=0)
    print(expanded_individual.shape)
    return model.predict(expanded_individual), 


def mutate_one_hot_genes(individual, mutation_rate: float):
    nucleotides = np.array(list(set(itertools.permutations([1, 0, 0, 0]))))
    # need to adjust since there is a chance that a nucleotide chosen for mutation will randomly be replaced by the same nucleotide
    actually_tested_rate = mutation_rate / .75
    for i in range(len(individual)):
        #ignore "N". should be part of the central padding of the extracted sequences
        if np.equal(individual[i], np.zeros(4)).all():
            continue
        if np.random.rand() < actually_tested_rate:
            individual[i] = nucleotides[np.random.choice(4)]

    return individual,
        


def setup_population(toolbox):
    # Define the problem as a maximization problem
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

    # Define the individual and population
    toolbox.register("individual", tools.initRepeat, creator.Individual, random_nucleotide, n=3020)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    return toolbox



def main():
    toolbox = base.Toolbox()
    setup_population(toolbox=toolbox)
    model = load_model("saved_models/arabidopsis_1_SSR_train_ssr_models_240816_183905.h5")
    toolbox.register("evaluate", evaluate_model, model=model)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", mutate_one_hot_genes, mutation_rate=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Run the genetic algorithm
    population = toolbox.population(n=5)
    for gen in range(10):
        print(population)
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population))

    # Get the best individual
    fits = [ind.fitness.values[0] for ind in population]
    best_idx = fits.index(max(fits))
    best_ind = population[best_idx]
    print("Best individual:", best_ind, "Fitness:", best_ind.fitness.values)


if __name__ == "__main__":
    main()
    # print(evaluate(np.array([[1, 0, 0, 0], [0, 0, 0, 1]])))