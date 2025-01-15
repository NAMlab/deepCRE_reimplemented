import datetime
from typing import Any, Dict, List, Tuple
import numpy as np
import os
from pyfaidx import Fasta
import pyranges as pr
import pandas as pd


def read_feature_from_input_dict(input_dict: Dict[str, str], key: str):
    val = input_dict[key]
    if isinstance(val, str):
        return val.strip()
    return val


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


def load_annotation(annotation_path) -> pd.DataFrame:
    gene_model = pr.read_gtf(f=annotation_path, as_df=True) if annotation_path.endswith(".gtf") else pr.read_gff3(f=annotation_path, as_df=True)
    gene_model = gene_model[gene_model['Feature'] == 'gene']
    columns = ['Chromosome', 'Start', 'End', 'Strand']
    if 'gene_id' in gene_model.columns:
        columns.append('gene_id')
    else:
        columns.append('ID')
    if 'gene_biotype' in gene_model.columns:
        gene_model = gene_model[gene_model['gene_biotype'] == 'protein_coding']
    gene_model = gene_model[columns]
    return gene_model           #type:ignore


# Function for reading and adapting the concatenated gtf file for MSR training
def load_annotation_msr(annotation_path) -> pd.DataFrame:
    gene_model = pd.read_csv(annotation_path, header=None, sep="\s+")           #type:ignore

    expected_columns = ['species', 'Chromosome', 'Start', 'End', 'Strand', 'gene_id']
    
    if gene_model.shape[1] != len(expected_columns):
        raise ValueError(f"CSV file must contain exactly {len(expected_columns)} columns.")
    gene_model.columns = expected_columns
    #gene_model['gene_id'] = gene_model['gene_id'].str.split(':').str[1]

    # Check if Chromosome is a single number and modify accordingly, same naming change in fasta
    for index, row in gene_model.iterrows():
        if row['Chromosome'].isdigit():  # Check if Chromosome is a number
            species_abbr = row['species'][:3]  # Get the first 3 letters of the species
            gene_model.at[index, 'Chromosome'] = f"{row['Chromosome']}{species_abbr}"  # Concatenate

    # Remove rows where 'Chromosome' contains 'Mt' or 'Pt'
    gene_model = gene_model[~gene_model['Chromosome'].str.contains('Mt|Pt|^scaffold', na=False)]

    
    return gene_model


def get_input_file_path(file_name: str, output_dir: str, species_name: str) -> str:
    combined_path = make_absolute_path(output_dir, file_name, start_file=__file__)
    absolute_path = file_name
    if os.path.exists(combined_path):
        file_path = combined_path
    elif os.path.exists(absolute_path):
        file_path = absolute_path
    else:
        raise Exception(f"Warning: annotation {file_name} could not be found. Neither {combined_path} nor {absolute_path} do exist for {species_name}.")
    return file_path


def get_fasta_data(fasta_path: str, species_abbr: str) -> List[str]:
    fasta_data = []
    include_sequence = False
    with open(fasta_path, 'r') as f:
        for line in f:
            if line.startswith('>'):  # Header line in FASTA
                parts = line.strip().split()
                chrom_name = parts[0][1:]  # Remove '>' from chromosome name

                # Skip headers containing 'Mt', 'Pt', or 'scaffold'
                if any(term in chrom_name for term in ['Mt', 'Pt', 'scaffold']):
                    include_sequence = False
                    continue  # Skip this header and go to the next line
                
                # Otherwise, mark the sequence for inclusion
                include_sequence = True  

                # Modify the header
                line = f">{chrom_name}_{species_abbr}\n"
                
                # Append the modified header to `fasta_data`
                fasta_data.append(line.strip())

            elif include_sequence:
                # Append sequence lines if the header was marked for inclusion
                fasta_data.append(line.strip())
    return fasta_data


def combine_annotations(species_data: List[Dict[str, Any]], naming: str) -> str:
    output_dir_name = 'gene_models'
    output_dir_path = make_absolute_path(output_dir_name, start_file=__file__)
    os.makedirs(output_dir_path, exist_ok=True)
    save_path = make_absolute_path(output_dir_name, f"gtf_{naming}.csv", start_file=__file__)
    
    # return if file already exists
    if os.path.exists(save_path):
        print(f"combined file for annotations \"{save_path}\" already exists.")
        return os.path.abspath(save_path)

    print(f'Now generating combined annotation file:')
    combined_data = []
    for specie_data in species_data:
        species_name = specie_data['species_name']
        input_file_path = get_input_file_path(specie_data["annotation"], output_dir=output_dir_name, species_name=species_name)
        file_data = load_annotation(input_file_path)
        # Add species name as a new column
        file_data.insert(0, 'species', species_name)
        #append the species name to each entry of the chromosome column and gene_id column
        chrom_col = file_data.columns[1]
        file_data[chrom_col] = file_data[chrom_col].apply(lambda x: f"{x}_{species_name}")
        gene_id_col = file_data.columns[-1]
        file_data[gene_id_col] = file_data[gene_id_col].apply(lambda x: f"{x}_{species_name}")
        combined_data.append(file_data)

    combined_data_df = pd.concat(combined_data, ignore_index=True)
    combined_data_df.to_csv(save_path, sep=' ', index=False, header=False)  # Save as GTF
    print(f"Combined annotation file saved as {save_path}")
    return save_path


def combine_annotations_old(data: pd.DataFrame) -> str:
    file_type = "gtf"
    file_key = "_".join([specie[:3] for specie in data['specie'].unique()])
    output_dir = 'gene_models'
    save_path = os.path.abspath(f"{output_dir}/{file_type}_{file_key}.csv")
    os.makedirs(output_dir, exist_ok=True)
    
    # return if file already exists
    if os.path.exists(save_path):
        print(f"combined file for annotations \"{save_path}\" already exists.")
        return os.path.abspath(save_path)

    print(f'Now generating combined {file_type} file:')
    combined_data = []
    for _, row in data.iterrows():
        input_file_path = get_input_file_path(row[file_type], output_dir=output_dir, species_name=row["species"])
        species_name = row['specie']
        file_data = load_annotation(input_file_path)
        # Add species name as a new column
        file_data.insert(0, 'species', species_name)
        combined_data.append(file_data)

    combined_data_df = pd.concat(combined_data, ignore_index=True)
    combined_data_df.to_csv(save_path, sep=' ', index=False, header=False)  # Save as GTF
    print(f"Combined {file_type} file saved as {save_path}")
    return save_path


def combine_tpms(species_data: List[Dict[str, Any]], naming: str) -> str:
    output_dir_name = "tpm_counts"
    output_dir_path = make_absolute_path("tpm_counts", start_file=__file__)
    os.makedirs(output_dir_path, exist_ok=True)
    save_path = os.path.join(output_dir_path, f"tpm_{naming}.csv")
    if os.path.exists(save_path):
        print(f"Combined file {save_path} already exists.")
        return save_path

    print(f'Now generating combined tpms file:')
    combined_data = []

    for specie_data in species_data:
        input_file_path = get_input_file_path(file_name=specie_data["targets"], output_dir=output_dir_name, species_name=specie_data["species_name"])
        file_data = pd.read_csv(input_file_path)
        # Select only the "gene_id" and "target" columns
        file_data = file_data.loc[:, ["gene_id", "target"]]
        # append the species name to each entry of the gene_id column
        file_data['gene_id'] = file_data['gene_id'].apply(lambda x: f"{x}_{specie_data['species_name']}")
        combined_data.append(file_data)
    
    combined_data_df = pd.concat(combined_data, ignore_index=True)
    combined_data_df.to_csv(save_path, index=False)
    print(f"Combined tpm file saved as {save_path}")
    return save_path


def combine_tpms_old(data: pd.DataFrame, naming: str) -> str:
    file_type = "tpm"
    output_dir = "tpm_counts"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.abspath(f"{output_dir}/{file_type}_{naming}.csv")
    if os.path.exists(save_path):
        print(f"Combined file {save_path} already exists.")
        return save_path

    print(f'Now generating combined {file_type} file:')
    combined_data = []

    for _, row in data.iterrows():
        input_file_path = get_input_file_path(file_name=row[file_type], output_dir=output_dir, species_name=row["species"])
        file_data = pd.read_csv(input_file_path)
        # Select only the "gene_id" and "target" columns
        file_data = file_data.loc[:, ["gene_id", "target"]]
        combined_data.append(file_data)
    
    combined_data_df = pd.concat(combined_data, ignore_index=True)
    combined_data_df.to_csv(save_path, index=False)
    print(f"Combined {file_type} file saved as {save_path}")
    return save_path


def combine_fasta(species_data: List[Dict[str, Any]], naming: str) -> str:
    file_type = 'genome'
    output_dir_name = "genome"
    output_dir_path = make_absolute_path(output_dir_name, start_file=__file__)
    os.makedirs(output_dir_path, exist_ok=True)
    save_path = os.path.join(output_dir_path, f"{file_type}_{naming}.fa")
    if os.path.exists(save_path):
        print(f"Combined file {save_path} already exists.")
        return save_path

    print(f'Now generating combined {file_type} file:')
    combined_data = []
    for specie_data in species_data:
        species_name = specie_data['species_name']
        input_file_path = get_input_file_path(file_name=specie_data[file_type], output_dir=output_dir_name, species_name=specie_data["species_name"])
        fasta_data = get_fasta_data(fasta_path=input_file_path, species_abbr=species_name)
        combined_data.append("\n".join(fasta_data))

    with open(save_path, 'w') as f_out:
        f_out.write("\n".join(combined_data))
    print(f"Combined {file_type} file saved as {save_path}")
    return save_path


def combine_fasta_old(data: pd.DataFrame) -> str:
    file_type = 'genome'
    output_dir = 'genome'
    file_key = "_".join([specie[:3] for specie in data['specie'].unique()])
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.abspath(f"{output_dir}/{file_type}_{file_key}.fa")
    if os.path.exists(save_path):
        print(f"Combined file {save_path} already exists.")
        return save_path

    print(f'Now generating combined {file_type} file:')
    combined_data = []
    for _, row in data.iterrows():
        species_name = row['specie']
        species_abbr = species_name[:3]
        input_file_path = get_input_file_path(file_name=row[file_type], output_dir=output_dir, species_name=row["species"])
        fasta_data = get_fasta_data(fasta_path=input_file_path, species_abbr=species_abbr)
        combined_data.append("\n".join(fasta_data))

    with open(save_path, 'w') as f_out:
        f_out.write("\n".join(combined_data))
    print(f"Combined {file_type} file saved as {save_path}")
    return save_path


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


def load_input_files(genome_file_name: str = "", annotation_file_name: str = "", tpm_counts_file_name: str = "", model_case: str = "ssr") -> Dict:
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
        try:    
            #see if given name is full path to file
            if os.path.isfile(genome_file_name):
                genome = Fasta(filename=genome_file_name, as_raw=True, read_ahead=10000, sequence_always_upper=True)
            else:
                genome_path = make_absolute_path("genome", genome_file_name, start_file=__file__)
                genome = Fasta(filename=genome_path, as_raw=True, read_ahead=10000, sequence_always_upper=True)
            results["genome"] = genome

        except ValueError as e:
            # Check if the error is due to duplicate keys
            if 'Duplicate key' in str(e):
                raise ValueError("Error: Chromosome names in the FASTA file cannot be the same across species. "
                                 "Please ensure unique chromosome identifiers for each species. ") from e
            

    if annotation_file_name != "":
        #see if given name is full path to file
        annotation_file_name = annotation_file_name if os.path.isfile(annotation_file_name) else make_absolute_path("gene_models", annotation_file_name, start_file=__file__)
        if model_case.lower() == "msr":
            annotation = load_annotation_msr(annotation_path=annotation_file_name)
        else:
            annotation = load_annotation(annotation_path=annotation_file_name)
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


def result_summary(failed_trainings: List[Tuple[str, int, Exception]], input_length: int, script: str) -> None:
    if failed_trainings:
        print("_______________________________________________________________")
        print(f"During your run of the script \"{script}\" the following errors occurred:")
        for name, line, err in failed_trainings:
            print(f"\"{name}\" (line {line + 1} in the input file) failed with error message:\n{err}")
            print("_______________________________________________________________")
        print(f"{input_length - len(failed_trainings)} / {input_length} passed.")
        print("_______________________________________________________________")
        print(f"names of the failed runs:", end=" ")
        for name, line, _ in failed_trainings:
            print(f"{name} (line {line + 1})", sep=", ")
    else:
        print("_______________________________________________________________")
        print(f"Your run of the script \"{script}\" finished with no errors!")
        print("_______________________________________________________________")


