import os
import tempfile
import unittest

import numpy as np
from pyfaidx import Fasta, FetchError
from unittest.mock import patch

from deepCRE.train_models import extract_gene
from deepCRE.utils import one_hot_encode


class TestExtractGene(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.fasta_path = os.path.join(cls.temp_dir.name, "test_genome.fa")
        with open(cls.fasta_path, "w", encoding="utf-8") as fasta_file:
            fasta_file.write(
                ">chr1\n"
                "AACCGGTTCCAAGGTT\n"
            )
        cls.genome = Fasta(cls.fasta_path, as_raw=True)

    @classmethod
    def tearDownClass(cls):
        cls.genome.close()
        cls.temp_dir.cleanup()

    def test_extract_gene_plus_strand_adds_central_padding(self):
        result = extract_gene(
            genome=self.genome,
            extragenic=2,
            intragenic=1,
            ignore_small_genes=False,
            expected_final_size=10,
            chrom="chr1",
            start=4,
            end=8,
            strand="+",
        )

        promoter = one_hot_encode(self.genome["chr1"][2:5])
        terminator = one_hot_encode(self.genome["chr1"][7:10])
        expected = np.concatenate(
            [
                promoter,
                np.zeros((4, 4)),
                terminator,
            ]
        )
        self.assertIsNotNone(result)
        if result is not None:
            np.testing.assert_array_equal(result, expected)
        else:
            self.fail("extract_gene returned None, expected a valid array.")
        result = extract_gene(
            genome=self.genome,
            extragenic=2,
            intragenic=1,
            ignore_small_genes=False,
            expected_final_size=26,
            chrom="chr1",
            start=4,
            end=8,
            strand="+",
        )

        promoter = one_hot_encode(self.genome["chr1"][2:5])
        terminator = one_hot_encode(self.genome["chr1"][7:10])
        expected = np.concatenate(
            [
                promoter,
                np.zeros((20, 4)),
                terminator,
            ]
        )
        self.assertIsNotNone(result)
        if result is not None:
            np.testing.assert_array_equal(result, expected)
        else:
            self.fail("extract_gene returned None, expected a valid array.")

    def test_extract_gene_minus_strand_returns_reverse_complement_orientation(self):
        result = extract_gene(
            genome=self.genome,
            extragenic=2,
            intragenic=1,
            ignore_small_genes=False,
            expected_final_size=10,
            chrom="chr1",
            start=4,
            end=8,
            strand="-",
        )

        promoter = one_hot_encode(self.genome["chr1"][2:5])
        terminator = one_hot_encode(self.genome["chr1"][7:10])
        expected = np.concatenate(
            [
                terminator[::-1, ::-1],
                np.zeros((4, 4)),
                promoter[::-1, ::-1],
            ]
        )

        np.testing.assert_array_equal(result, expected)

    def test_extract_gene_uses_fixed_padding_for_small_genes_when_ignored(self):
        result = extract_gene(
            genome=self.genome,
            extragenic=2,
            intragenic=5,
            ignore_small_genes=True,
            expected_final_size=30,
            chrom="chr1",
            start=4,
            end=7,
            strand="+",
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (26, 4))

    def test_extract_gene_returns_none_when_sequence_fetch_fails(self):
        result = extract_gene(
            genome=self.genome,
            extragenic=1000,
            intragenic=1,
            ignore_small_genes=False,
            expected_final_size=10,
            chrom="chr1",
            start=10,
            end=15,
            strand="+",
        )

        self.assertIsNone(result)
    

if __name__ == "__main__":
    unittest.main()