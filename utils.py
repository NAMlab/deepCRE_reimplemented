import datetime
from typing import Any, Dict, List, Tuple
import numpy as np
import os
from pyfaidx import Fasta
import pyranges as pr
import pandas as pd


def one_hot_encode(sequence: str,
                   alphabet: str = 'ACGT',
                   neutral_alphabet: str = 'N',
                   neutral_value: Any = 0,
                   dtype=np.float32) -> np.ndarray:
    """
    One-hot encode sequence. This function expects a nucleic acid sequences with 4 bases: ACGT.
    It also assumes that unknown nucleotides within the sequence are N's.
    :param sequence: nucleotide sequence

    :return: 4 x L one-hot encoded matrix
    """
    def to_uint8(string):
        return np.frombuffer(string.encode('ascii'), dtype=np.uint8)
    hash_table = np.zeros((np.iinfo(np.uint8).max, len(alphabet)), dtype=dtype)
    hash_table[to_uint8(alphabet)] = np.eye(len(alphabet), dtype=dtype)
    hash_table[to_uint8(neutral_alphabet)] = neutral_value
    hash_table = hash_table.astype(dtype)
    return hash_table[to_uint8(sequence)]


def get_time_stamp() -> str:
    """creates a time stamp for the current time

    Returns:
        str: string in the format date_time
    """
    return datetime.datetime.now().strftime("%y%m%d_%H%M%S")


def get_filename_from_path(path: str) -> str:
    """takes a path and returns the name of the file it leads to

    Args:
        path (str): path to a file

    Returns:
        str: name of the file
    """
    file_name = os.path.splitext(os.path.basename(path))[0]
    return file_name


def load_annotation(annotation_path):
    if annotation_path.endswith('.gtf'):
        gene_model = pr.read_gtf(f=annotation_path, as_df=True)
        gene_model = gene_model[gene_model['gene_biotype'] == 'protein_coding']
        gene_model = gene_model[gene_model['Feature'] == 'gene']
        gene_model = gene_model[['Chromosome', 'Start', 'End', 'Strand', 'gene_id']]
    else:
        gene_model = pr.read_gff3(annotation_path, as_df=True)
        gene_model = gene_model[gene_model['Feature'] == 'gene']
        gene_model = gene_model[['Chromosome', 'Start', 'End', 'Strand', 'ID']]
    return gene_model



def make_absolute_path(*steps_on_path, start_file: str = "") -> str:
    """creates an absoulte path from a starting location

    Args:
        start_file (str, optional): file from which the path starts. Defaults to "".
        steps_on_path (str, optional): arbitrary number of folders with an optionally file at the end which will be appended to the start path.

    Returns:
        str: absolute version of the path
    """
    if start_file == "":
        start_file = __file__
    start_folder = os.path.dirname(os.path.abspath(start_file))
    result_path = os.path.join(start_folder, *steps_on_path)
    return result_path


def load_input_files(genome_file_name: str = "", annotation_file_name: str = "", tpm_counts_file_name: str = "") -> Dict[str, pd.DataFrame]:
    """loads input files and returns them in a Dict

    Args:
        genome_file_name (str, optional): file name of the genome, saved in the subfolder \"genome\". Defaults to "".
        annotation_file_name (str, optional): file name of the annotation, saved in the subfolder \"gene_models\". Defaults to "".
        tpm_counts_file_name (str, optional):  file name of the tpm counts, saved in the subfolder \"tpm_counts\". Defaults to "".

    Raises:
        ValueError: _description_

    Returns:
        Dict[str, pd.DataFrame]: _description_
    """
    if genome_file_name == "" and annotation_file_name == "" and tpm_counts_file_name == "":
        raise ValueError("at least one of the file names must be given!")
    results = {}

    if genome_file_name != "":
        #see if given name is full path to file
        if os.path.isfile(genome_file_name):
            genome = Fasta(filename=genome_file_name, as_raw=True, read_ahead=10000, sequence_always_upper=True)
        else:
            genome_path = make_absolute_path("genome", genome_file_name, start_file=__file__)
            genome = Fasta(filename=genome_path, as_raw=True, read_ahead=10000, sequence_always_upper=True)
        results["genome"] = genome

    if annotation_file_name != "":
        #see if given name is full path to file
        if os.path.isfile(annotation_file_name):
            annotation = load_annotation(annotation_path=annotation_file_name)
        else:
            annotation_path = make_absolute_path("gene_models", annotation_file_name, start_file=__file__)
            annotation = load_annotation(annotation_path=annotation_path)
        # annot = annot[annot['Chromosome'] == val_chromosome]
        results["annotation"] = annotation

    if tpm_counts_file_name != "":
        #see if given name is full path to file
        if os.path.isfile(tpm_counts_file_name):
            tpms = pd.read_csv(filepath_or_buffer=tpm_counts_file_name, sep=',')

        else:
            tpm_path = make_absolute_path("tpm_counts", tpm_counts_file_name, start_file=__file__)
            tpms = pd.read_csv(filepath_or_buffer=tpm_path, sep=',')
        tpms.set_index('gene_id', inplace=True)
        results["tpms"] = tpms

    return results


def result_summary(failed_trainings: List[Tuple[str, int, Exception]], passed_trainings: List[Tuple[str, int]], input_length: int, script: str) -> None:
    if failed_trainings:
        print("_______________________________________________________________")
        print(f"During your run of the script {script} the following errors occurred:")
        for name, line, err in failed_trainings:
            print(f"\"{name}\" (line {line + 1} in the input file) failed with error message:\n{err}")
            print("_______________________________________________________________")
        print(f"{len(passed_trainings)} / {input_length} passed.")
        print("_______________________________________________________________")
        print(f"names of the failed runs:", end=" ")
        for name, line, _ in failed_trainings:
            print(f"{name} (line {line + 1})", sep=", ")

