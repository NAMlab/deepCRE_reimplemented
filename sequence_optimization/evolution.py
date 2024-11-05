import math
import pickle
import pstats
from typing import Callable, List, Tuple, Type
import numpy as np
import deap
import itertools
import cProfile

import random
from deap import base, creator, tools, algorithms
from tensorflow.keras.models import load_model #type:ignore


class FakeModel():
    def predict(self, individual):
        return evaluate_test(individual)[0]


def visually_compare_sequences(seq1, seq2):
    for (a, b) in zip(seq1, seq2):
        print(f"{a}  |  {b}", end="")
        if np.equal(a, b).all():
            print("")
        else:
            print("  <---- Error here!")


def cxOnePointNumPy(ind1, ind2):
    """Executes a one point crossover on the input :term:`sequence` individuals.
    The two individuals are modified in place. The resulting individuals will
    respectively have the length of the other.

    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :returns: A tuple of two individuals.

    This function uses the :func:`~random.randint` function from the
    python base :mod:`random` module.
    """
    size = min(len(ind1), len(ind2))
    cxpoint = random.randint(1, size - 1)
    ind1[cxpoint:], ind2[cxpoint:] = ind2[cxpoint:].copy(), ind1[cxpoint:].copy()

    return ind1, ind2


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
    weights = np.array([1, 0.4, 0.2, 0.])
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


def init_mutated_sequence(np_individual_class: Type, base_sequence: np.ndarray, initial_mutation_rate: float):
    individual = mutate_one_hot_genes(base_sequence.copy(), mutation_rate=initial_mutation_rate)
    individual = np.squeeze(individual, 0)
    individual = limit_mutations(individual=individual, reference_sequence=base_sequence, rel_max_difference=initial_mutation_rate)[0]
    individual = np_individual_class(individual)
    return individual


def init_population(population_class: Type, init_individual: Callable, initial_mutation_rate: float, population_size: int):
    if population_size < 1:
        raise ValueError(f"Population supposed to be initialized with {population_size} < 1. Populations with less than 1 individual make no sense.")
    individuals = [init_individual(initial_mutation_rate=0)]
    for _ in range(population_size - 1):
        individuals.append(init_individual(initial_mutation_rate=initial_mutation_rate))
    return population_class(individual for individual in individuals)


def foo(input: List):
    return len(input)


def bar(input: List):
    return input[0][0]


def setup_stats_objects() -> tools.MultiStatistics:
    stats_fitness = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats_fitness.register("avg", np.mean, axis=0)
    stats_fitness.register("std", np.std, axis=0)
    stats_fitness.register("min", np.min, axis=0)
    stats_fitness.register("max", np.max, axis=0)
    stats_population = tools.Statistics(key=lambda ind: ind)
    stats_population.register("foo", foo)
    stats_population.register("bar", bar)
    multi_stats = tools.MultiStatistics(fitness=stats_fitness, population=stats_population)
    return multi_stats


def genetic_algorithm(number_of_nucleotides: int, population_size: int, number_of_generations: int,
                      tournment_size: int, mutation_rate: float, mutation_probability: float,
                      crossover_probability: float, models: List, reference_sequence: np.ndarray,
                      rel_max_difference: float, optimize: bool = True):
    # Run the genetic algorithm
    toolbox = base.Toolbox()
    # Define the problem as a maximization problem
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

    # Define the individual and population
    if optimize:
        toolbox.register("individual", init_mutated_sequence, creator.Individual, reference_sequence)
    else:
        toolbox.register("individual", tools.initRepeat, creator.Individual, random_nucleotide, n=number_of_nucleotides)
    toolbox.register("population", init_population, list, toolbox.individual, initial_mutation_rate=rel_max_difference)
    toolbox.register("evaluate", evaluate_model, models=models)
    toolbox.register("mate", cxOnePointNumPy)
    toolbox.register("mutate", mutate_one_hot_genes, mutation_rate=mutation_rate)
    toolbox.register("select", tools.selTournament, tournsize=tournment_size)
    toolbox.register("limit_mutations", limit_mutations, reference_sequence=reference_sequence, rel_max_difference=rel_max_difference)
    

    population = toolbox.population(population_size=population_size)
    fits = toolbox.map(toolbox.evaluate, population)
    for fit, ind in zip(fits, population):
        ind.fitness.values = fit

    hall_of_fame = tools.HallOfFame(5, similar=np.allclose)
    hall_of_fame.update(population)
    stats = setup_stats_objects()
    logbook = tools.Logbook()
    for gen in range(number_of_generations):
        # print(population)
        # print(gen)
        offspring = toolbox.select(population, k=len(population))
        offspring = algorithms.varAnd(offspring, toolbox, cxpb=crossover_probability, mutpb=mutation_probability)
        if optimize:
            offspring = [element[0] for element in toolbox.map(toolbox.limit_mutations, offspring)]
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population[:] = offspring
        hall_of_fame.update(population)
        logbook.record(gen=gen, **stats.compile(population))
    
    print_summary(population=population, toolbox=toolbox, reference_sequence=reference_sequence, hall_of_fame=hall_of_fame, logbook=logbook)


def print_summary(population: list, toolbox: base.Toolbox, reference_sequence: np.ndarray, hall_of_fame: tools.HallOfFame, logbook: tools.Logbook):
    # Get the best individual
    fits = [ind.fitness.values[0] for ind in population]
    best_idx = fits.index(max(fits))
    best_ind = population[best_idx]
    initial_fitness = toolbox.evaluate(reference_sequence)
    print(f"initial fitness: {initial_fitness[0]}")
    print(f"best fitness current population: {best_ind.fitness.values[0].item()}")
    # visually_compare_sequences(reference_sequence, best_ind)
    print(f"best fitness total: {hall_of_fame.items[0].fitness.values[0].item()}")
    # visually_compare_sequences(reference_sequence, hall_of_fame.items[0])
    logbook.header = "gen", "fitness", "population"
    logbook.chapters["fitness"].header = "avg", "std", "min", "max"
    logbook.chapters["population"].header = "foo", "bar"
    print(logbook)
    with open("test_folder/logbooks/logbook.pkl", "wb") as f:
        pickle.dump(logbook, f)


def main():
    number_of_nucleotides = 200
    reference_sequence_inds = np.random.choice(np.arange(4), number_of_nucleotides)
    nucleotides = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])
    reference_sequence = np.array([nucleotides[i] for i in reference_sequence_inds])
    population_size = 20
    number_of_generations = 10
    tournment_size = 10
    mutation_rate = 0.1
    mutation_probability = 0.5
    crossover_probability = 0.5
    # models = [load_model("saved_models/arabidopsis_1_SSR_train_ssr_models_240816_183905.h5")]
    models = [FakeModel()]
    rel_max_difference = 0.2
    genetic_algorithm(number_of_nucleotides=number_of_nucleotides, number_of_generations=number_of_generations, population_size=population_size,
                      tournment_size=tournment_size, mutation_rate=mutation_rate, mutation_probability=mutation_probability,
                      crossover_probability=crossover_probability, models=models, rel_max_difference=rel_max_difference,
                      reference_sequence=reference_sequence)


if __name__ == "__main__":
    # with cProfile.Profile() as profile:
    #     main()
    
    # results = pstats.Stats(profile)
    # results.sort_stats(pstats.SortKey.TIME)
    # results.print_stats()

    # print(evaluate(np.array([[1, 0, 0, 0], [0, 0, 0, 1]])))
    main()