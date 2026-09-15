"""Unit tests for :func:`deepCRE.utils.load_annotation`.

The regression guarded here is that every annotation format must yield a
``gene_id`` column. GFF3 files commonly name genes with an ``ID`` attribute
instead, and the rest of the pipeline indexes the annotation by ``gene_id``
(``train_models.extract_genes_training``), so an un-renamed ``ID`` column made
training fail immediately with ``"['gene_id'] not in index"``.
"""

import os
import unittest

from deepCRE.utils import load_annotation

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "test_outputs", "load_annotation"
)


def write_lines(path: str, lines) -> str:
    """Write lines to a file and return its path.

    Args:
        path: Destination path.
        lines: Iterable of lines, without trailing newlines.

    Returns:
        The destination path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    return path


class TestLoadAnnotation(unittest.TestCase):
    """Tests for annotation loading across formats."""

    def test_gff3_with_id_attribute_yields_gene_id(self) -> None:
        """A GFF3 naming genes by ``ID`` is returned with a ``gene_id`` column."""
        path = write_lines(
            os.path.join(OUTPUT_DIR, "id_only.gff3"),
            [
                "##gff-version 3",
                "s1\ttest\tgene\t2001\t5000\t.\t+\t.\tID=gene_one",
                "s1\ttest\tmRNA\t2001\t5000\t.\t+\t.\tID=gene_one.1;Parent=gene_one",
                "s1\ttest\tgene\t9001\t12000\t.\t-\t.\tID=gene_two",
            ],
        )

        annotation = load_annotation(path)

        self.assertIn("gene_id", annotation.columns)
        self.assertEqual(list(annotation["gene_id"]), ["gene_one", "gene_two"])

    def test_only_gene_features_are_kept(self) -> None:
        """Transcript and exon rows are discarded."""
        path = write_lines(
            os.path.join(OUTPUT_DIR, "mixed.gff3"),
            [
                "##gff-version 3",
                "s1\ttest\tgene\t2001\t5000\t.\t+\t.\tID=gene_one",
                "s1\ttest\texon\t2001\t2500\t.\t+\t.\tID=gene_one.exon1;Parent=gene_one",
            ],
        )

        annotation = load_annotation(path)

        self.assertEqual(len(annotation), 1)

    def test_gtf_gene_id_is_preserved(self) -> None:
        """A GTF already naming genes by ``gene_id`` is unaffected."""
        path = write_lines(
            os.path.join(OUTPUT_DIR, "genes.gtf"),
            [
                "##gtf-version 2.2",
                's1\ttest\tgene\t2001\t5000\t.\t+\t.\tgene_id "gene_one";',
                's1\ttest\tgene\t9001\t12000\t.\t-\t.\tgene_id "gene_two";',
            ],
        )

        annotation = load_annotation(path)

        self.assertEqual(list(annotation["gene_id"]), ["gene_one", "gene_two"])

    def test_expected_columns_are_returned(self) -> None:
        """The column set the pipeline relies on is returned, in order."""
        path = write_lines(
            os.path.join(OUTPUT_DIR, "columns.gff3"),
            [
                "##gff-version 3",
                "s1\ttest\tgene\t2001\t5000\t.\t+\t.\tID=gene_one",
            ],
        )

        annotation = load_annotation(path)

        self.assertEqual(
            list(annotation.columns),
            ["Chromosome", "Start", "End", "Strand", "gene_id"],
        )

    def test_annotation_without_any_gene_name_raises(self) -> None:
        """An annotation with no usable identifier is reported clearly."""
        path = write_lines(
            os.path.join(OUTPUT_DIR, "nameless.gff3"),
            [
                "##gff-version 3",
                "s1\ttest\tgene\t2001\t5000\t.\t+\t.\tNote=nothing",
            ],
        )

        with self.assertRaises(ValueError):
            load_annotation(path)


if __name__ == "__main__":
    unittest.main()
