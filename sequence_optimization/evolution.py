import math
from typing import List, Tuple
import numpy as np
import deap
import itertools

import random
from deap import base, creator, tools, algorithms
from tensorflow.keras.models import load_model #type:ignore


class Sequence(np.ndarray):
    original_sequence: np.ndarray

    def __init__(self, original_sequence: np.ndarray):
        super(Sequence, self).__init__()
        self.original_sequence = original_sequence


def compare_sequences(seq1, seq2) -> List[int]:
    """counts differences in two genetic sequnces

    Args:
        seq1 (_type_): first sequence (should be list or numpy array of numpy arrays)
        seq2 (_type_): second sequence (should be list or numpy array of numpy arrays)

    Raises:
        ValueError: raised if input sequences dont have the same length

    Returns:
        Tuple[int, float]: returns count of differences as well as fraction of differences
    """
    sequence_length = len(seq1)
    if sequence_length != len(seq2):
        raise ValueError(f"compared sequences dont have matching lengths!")
    differences = []
    for i, (curr1, curr2) in enumerate(zip(seq1, seq2)):
        if not np.isclose(curr1, curr2).all():
            differences.append(i)
    return differences

def random_nucleotide() -> np.ndarray:
    """function to return a random nucleotide in the one hot encoded format

    Returns:
        np.ndarry: ont hot encoded nucleotide as numpy array
    """
    i = np.random.choice(4)
    nucleotides = np.array(list(set(itertools.permutations([1, 0, 0, 0]))))
    return nucleotides[i]


# Define the fitness function
def evaluate_test(individual) -> Tuple[float]:
    """just a test function to evaluate sequences by assigning a value to each nucleotide and summing them up over the length of the sequence.

    Args:
        individual (_type_): individual of the evolutionary deap toolset. needs to be a list containing numpy array of legth 4.

    Returns:
        Tuple[float]: result of the evaluation
    """
    sum = 0
    weights = np.array([1, 0., 0., 0.])
    for nucleotide in individual:
        res = weights * nucleotide
        sum += np.sum(res)
    return (sum,)


def evaluate_model(individual, models: List):
    """evaluating the fitness of a genetic sequence using a machine learning model

    Args:
        individual (_type_): individual of the evolutionary deap toolset. needs to be a list containing numpy array of legth 4. length of the sequence must meet the training length of the model, so usually 3020 bp. 
        model (_type_): machine learning model to be used for the evaluation.

    Returns:
        Tuple[float]: the result of the evaluation
    """
    expanded_individual = np.expand_dims(individual, axis=0)
    predictions = [model.predict(expanded_individual) for model in models]
    return tuple(predictions)


def mutate_one_hot_genes(individual, mutation_rate: float) -> Tuple:
    """mutates one hot encoded genetic sequences

    Args:
        individual (_type_): individual of the evolutionary deap toolset. needs to be a list containing numpy array of legth 4.
        mutation_rate (float): rate of mutations to be introduced to the sequence

    Returns:
        Tuple: mutated individuaresulting sequences should still resemble inputl
    """
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


def limit_mutations(individual, reference_sequence: np.ndarray, rel_max_difference: float):
    """Makes sure an input individual doesnt divert from a reference sequence by more than a limit. If it does, randomly picked changes will be reverted.

    Args:
        individual (_type_): individual containing genetic sequence. Original individual will be mutated.
        reference_sequence (np.ndarray): reference sequence to compare the individual against
        rel_max_difference (float): Ratio of changes that is allowed between the individual and the reference sequence.

    Returns:
        _type_: returns the individual after possibly adjusting its sequence.
    """
    differences = compare_sequences(individual, reference_sequence)
    max_changes_allowed = math.floor(len(individual) * rel_max_difference)
    number_to_revert = len(differences) - max_changes_allowed
    if number_to_revert > 0:
        indeces_to_revert = random.sample(differences, number_to_revert)
        for index in indeces_to_revert:
            individual[index] = reference_sequence[index]
    return individual,
    

def genetic_algorithm(number_of_nucleotides: int, population_size: int, number_of_generations: int,
                      tournment_size: int, mutation_rate: float, mutation_probability: float,
                      crossover_probability: float, model_paths: List[str], reference_sequence: np.ndarray,
                      rel_max_difference: float):
    # Run the genetic algorithm
    toolbox = base.Toolbox()
    # Define the problem as a maximization problem
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

    # Define the individual and population
    toolbox.register("individual", tools.initRepeat, creator.Individual, random_nucleotide, n=number_of_nucleotides)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    models = [load_model(model_path) for model_path in model_paths]
    toolbox.register("evaluate", evaluate_model, models=models)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", mutate_one_hot_genes, mutation_rate=mutation_rate)
    toolbox.register("select", tools.selTournament, tournsize=tournment_size)
    toolbox.register("limit_mutations", limit_mutations, reference_sequence=reference_sequence, rel_max_difference=rel_max_difference)

    population = toolbox.population(n=population_size)
    for gen in range(number_of_generations):
        # print(population)
        print(gen)
        offspring = algorithms.varAnd(population, toolbox, cxpb=crossover_probability, mutpb=mutation_probability)
        offspring = [element[0] for element in toolbox.map(toolbox.limit_mutations, offspring)]
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population))

    # Get the best individual
    fits = [ind.fitness.values[0] for ind in population]
    best_idx = fits.index(max(fits))
    best_ind = population[best_idx]
    print("Best individual:", best_ind, "Fitness:", best_ind.fitness.values)
        

def main():
    reference_sequence_inds = np.random.choice(np.arange(4), 3020)
    nucleotides = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])
    reference_sequence = np.array([nucleotides[i] for i in reference_sequence_inds])
    number_of_nucleotides = 3020
    population_size = 5
    number_of_generations = 10
    tournment_size = 3
    mutation_rate = 0.1
    mutation_probability = 0.5
    crossover_probability = 0.5
    model_paths = ["saved_models/arabidopsis_1_SSR_train_ssr_models_240816_183905.h5"]
    genetic_algorithm(number_of_nucleotides=number_of_nucleotides, number_of_generations=number_of_generations, population_size=population_size,
                      tournment_size=tournment_size, mutation_rate=mutation_rate, mutation_probability=mutation_probability,
                      crossover_probability=crossover_probability, model_paths=model_paths, rel_max_difference=0.1,
                      reference_sequence=reference_sequence)


if __name__ == "__main__":
    main()
    # print(evaluate(np.array([[1, 0, 0, 0], [0, 0, 0, 1]])))