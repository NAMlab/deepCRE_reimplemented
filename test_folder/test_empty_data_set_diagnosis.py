"""Tests for the diagnosis attached to an empty training or validation data set."""
import unittest

import pandas as pd

from deepCRE.parsing import ModelCase
from deepCRE.train_models import diagnose_empty_data_set


def make_annotation(gene_ids, chromosomes):
    """Builds a minimal annotation frame with the columns the diagnosis reads.

    Args:
        gene_ids: gene ids to put into the "gene_id" column
        chromosomes: chromosome names to put into the "Chromosome" column

    Returns:
        pd.DataFrame: annotation with a "gene_id" and a "Chromosome" column
    """
    return pd.DataFrame({"gene_id": gene_ids, "Chromosome": chromosomes})


def make_tpms(gene_ids):
    """Builds a minimal targets frame indexed by gene id.

    Args:
        gene_ids: gene ids to use as the index

    Returns:
        pd.DataFrame: targets indexed by gene id
    """
    return pd.DataFrame({"target": [0] * len(gene_ids)}, index=pd.Index(gene_ids, name="gene_id"))


class TestDiagnoseEmptyDataSet(unittest.TestCase):
    def test_disjoint_gene_ids_are_reported_as_a_naming_mismatch(self):
        # Arrange: annotation and targets describe the same genome but name genes differently
        annotation = make_annotation(["gene-LOC1", "gene-LOC2"], ["chr1", "chr1"])
        tpms = make_tpms(["Species_chr1_000001", "Species_chr1_000002"])

        # Act
        diagnosis = diagnose_empty_data_set(annotation=annotation, tpms=tpms, validation_genes=[],
                                            pickled_key="key", val_chromosome="chr1",
                                            model_case=ModelCase.SSR, empty_validation_set=True)

        # Assert: the mismatch is named, and the fragmentation explanation is not offered
        self.assertIn("None of the 2 genes in the annotation appear in the targets file", diagnosis)
        self.assertIn("gene-LOC1", diagnosis)
        self.assertIn("Species_chr1_000001", diagnosis)
        self.assertNotIn("fragmented", diagnosis)

    def test_matching_gene_ids_fall_back_to_the_validation_gene_explanation(self):
        # Arrange: gene ids match, so only the validation gene list can explain the empty set
        annotation = make_annotation(["g1", "g2"], ["chr1", "chr2"])
        tpms = make_tpms(["g1", "g2"])

        # Act
        diagnosis = diagnose_empty_data_set(annotation=annotation, tpms=tpms, validation_genes=["g2"],
                                            pickled_key="key", val_chromosome="chr1",
                                            model_case=ModelCase.SSR, empty_validation_set=True)

        # Assert
        self.assertIn("The validation set is empty", diagnosis)
        self.assertIn("chromosome 'chr1' holds 1 annotated genes", diagnosis)
        self.assertNotIn("None of the", diagnosis)

    def test_partial_overlap_is_not_reported_as_a_naming_mismatch(self):
        # Arrange: a single shared gene id is enough to rule the namespaces compatible
        annotation = make_annotation(["g1", "gene-LOC2"], ["chr1", "chr1"])
        tpms = make_tpms(["g1", "Species_chr1_000002"])

        # Act
        diagnosis = diagnose_empty_data_set(annotation=annotation, tpms=tpms, validation_genes=[],
                                            pickled_key="key", val_chromosome="chr1",
                                            model_case=ModelCase.SSR, empty_validation_set=True)

        # Assert
        self.assertNotIn("None of the", diagnosis)
        self.assertIn("The validation set is empty", diagnosis)

    def test_msr_with_matching_ids_and_full_validation_set_yields_no_diagnosis(self):
        # Arrange: nothing the diagnosis knows about can explain this run
        annotation = make_annotation(["g1"], ["chr1"])
        tpms = make_tpms(["g1"])

        # Act
        diagnosis = diagnose_empty_data_set(annotation=annotation, tpms=tpms, validation_genes=["g1"],
                                            pickled_key="key", val_chromosome="chr1",
                                            model_case=ModelCase.MSR, empty_validation_set=False)

        # Assert
        self.assertEqual(diagnosis, "")


if __name__ == "__main__":
    unittest.main()
