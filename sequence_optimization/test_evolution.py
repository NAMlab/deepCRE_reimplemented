from typing import List
from tensorflow.keras.models import load_model #type:ignore
import unittest
import numpy as np

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


if __name__ == "__main__":
    unittest.main()