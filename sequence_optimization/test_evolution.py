import pickle
from typing import List
from tensorflow.keras.models import load_model #type:ignore
import unittest
import numpy as np
import matplotlib.pyplot as plt

import evolution as evo


def contains_nucleotide(nucleotide: np.ndarray, nucleotide_list: List[np.ndarray]) -> bool:
    for curr_nucleotide in nucleotide_list:
        if np.equal(curr_nucleotide, nucleotide).all():
            return True
    return False

class TestEvolution(unittest.TestCase):
    def test_random_nucleotide(self):
        correct_nucleotides = [
            np.array([1, 0, 0, 0]),
            np.array([0, 1, 0, 0]),
            np.array([0, 0, 1, 0]),
            np.array([0, 0, 0, 1]),
        ]
        nucleotides = [evo.random_nucleotide() for _ in range(1000)]
        for nucleotide in nucleotides:
            self.assertTrue(contains_nucleotide(nucleotide=nucleotide, nucleotide_list=correct_nucleotides))
        
        for corr_nucleotide in correct_nucleotides:
            self.assertTrue(contains_nucleotide(nucleotide=corr_nucleotide, nucleotide_list=nucleotides))

    def test_evaluate_test(self):
        test_individuals = [
            [],
            [[0, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 1, 1, 1], [0, 1, 1, 1]],
            [[1, 0, 0, 0], [1, 0, 0, 0]],
            [[1, 1, 1, 1], [1, 1, 1, 1]],
        ]
        expected_results = [
            0,
            0,
            0,
            0,
            2,
            2,
        ]
        calculated_results = [evo.evaluate_test(individual)[0] for individual in test_individuals]
        self.assertTrue(expected_results == calculated_results)

    def test_evaluate_model(self):
        models = [load_model("saved_models/arabidopsis_1_SSR_train_ssr_models_240816_183905.h5")]
        test_individuals = [
            [evo.random_nucleotide() for _ in range(3020)],
            [evo.random_nucleotide() for _ in range(3020)]
        ]
        test_individuals = np.array(test_individuals)
        results = [evo.evaluate_model(individual, models=models)[0] for individual in test_individuals]
        for result in results:
            self.assertLess(result, 1)
            self.assertGreater(result, 0)
    
    def test_compare_sequences(self):
        ref_seq = [
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
        ]
        empty_seq = []
        tiny_diff = [
            [1.000000000001, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
        ]
        diff_1 = [
            [2, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
        ]
        diff_2 = [
            [1, 0, 1, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 2, 0],
        ]

        results = [bool(evo.compare_sequences(empty_seq, empty_seq))]
        results.append(bool(evo.compare_sequences(ref_seq, ref_seq)))
        results.append(bool(evo.compare_sequences(ref_seq, tiny_diff)))
        results.append(bool(evo.compare_sequences(ref_seq, diff_1)))
        expected_results = [False, False, False, True]
        self.assertEqual(expected_results, results)
        self.assertRaises(ValueError, evo.compare_sequences, ref_seq, empty_seq)
        self.assertRaises(ValueError, evo.compare_sequences, ref_seq, diff_2)

    def test_mutate_one_hot_genes(self):
        length = 1020
        test_individuals = [[evo.random_nucleotide() for _ in range(length)] for _  in range(5)]
        test_individuals = np.array(test_individuals)
        test_individuals[:, 500:520] = np.array([0, 0, 0, 0])
        comparison = np.copy(test_individuals)
        mutated_individuals = [evo.mutate_one_hot_genes(individual, mutation_rate=0.1)[0] for individual in test_individuals]
        mutated_individuals = np.array(mutated_individuals)
        lower_threshold = 0.04
        upper_threshold = 0.14
        for (original, mutated) in zip(comparison, mutated_individuals):
            self.assertEqual(len(mutated), length)
            difference = evo.compare_sequences(original, mutated)
            relative_difference = len(difference) / length
            self.assertLessEqual(relative_difference, upper_threshold)
            self.assertGreaterEqual(relative_difference, lower_threshold)

    def test_limit_mutations(self):
        reference_sequence = np.array([
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
        ])
        seq_1 = np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
        ])
        seq_2 = np.array([
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
        ])
        expected_results = [reference_sequence.copy(), reference_sequence.copy(), seq_1.copy(), seq_2.copy()]
        results = [
            evo.limit_mutations(reference_sequence.copy(), reference_sequence, 0.5),
            evo.limit_mutations(seq_1.copy(), reference_sequence, 0.0),
            evo.limit_mutations(seq_1.copy(), reference_sequence, 0.5),
            evo.limit_mutations(seq_2.copy(), reference_sequence, 1),
        ]
        for i, (expected, result) in enumerate(zip(expected_results, results)):
            if not np.equal(expected, result).all():
                print(i)
                print(expected)
                print(result)
                self.assertTrue(False)
            
def visualize_results():
    with open("test_folder/logs/250109_112525/logbook", "rb") as f:
        logbook = pickle.load(f)
    gen = logbook.select("gen")
    fit_min = logbook.chapters["fitness"].select("min")
    # remove unnecessary dimensions of length 1 from nested list by converting to numpy
    fit_min = np.array(fit_min).flatten()
    fit_avg = logbook.chapters["fitness"].select("avg")
    fit_avg = np.array(fit_avg).flatten()
    fit_max = logbook.chapters["fitness"].select("max")
    fit_max = np.array(fit_max).flatten()



    fig, ax1 = plt.subplots()
    line1 = ax1.plot(gen, fit_min, "b-", label="Minimum Fitness")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness")
    # plt.yticks = [0, 0.2, 0.4, 0.6, 0.8, 1]

    line2 = ax1.plot(gen, fit_avg, "r-", label="Average Fitness")
    line3 = ax1.plot(gen, fit_max, "g-", label="Maximum Fitness")

    lns = line3 + line2 + line1
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="lower right")
    ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.savefig("test_folder/logs/250109_112525/plot.png", bbox_inches="tight")


def is_pareto(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        is_efficient[i] = np.all(np.any(costs[:i]>c, axis=1)) and np.all(np.any(costs[i+1:]>c, axis=1))
    return is_efficient



def vis_pareto():
    # create random normal distributed data
    point_list = []
    for i in range(100):
        point_list.append([np.random.normal(0,1), np.random.normal(0,1)])
    max_x = max(point_list, key=lambda x: x[0])[0]
    max_y = max(point_list, key=lambda x: x[1])[1]
    min_x = min(point_list, key=lambda x: x[0])[0]
    min_y = min(point_list, key=lambda x: x[1])[1]
    # normalize data
    for i in range(len(point_list)):
        point_list[i][0] = round((point_list[i][0] - min_x) / (max_x - min_x) * 50) 
        point_list[i][1] = - (point_list[i][1] - min_y) / (max_y - min_y)
    # mark pareto efficient points
    pareto = np.array(point_list)
    pareto = pareto[is_pareto(pareto)]
    # create plot
    x = [point[0] for point in point_list]
    y = [point[1] for point in point_list]
    x_pareto = [point[0] for point in pareto]
    y_pareto = [point[1] for point in pareto]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.scatter(x, y)
    ax.scatter(x_pareto, y_pareto, color="red")
    # add axis labels
    ax.set_xlabel("Number of Mutations")
    ax.set_ylabel("Negative Fitness")

    plt.savefig("test_folder/logs/250109_112525/pareto.png", bbox_inches="tight")



if __name__ == "__main__":
    # unittest.main()
    # visualize_results()
    vis_pareto()