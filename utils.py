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
        #gene_model = gene_model[gene_model['gene_biotype'] == 'protein_coding']
        if 'gene_biotype' in gene_model.columns:
            gene_model = gene_model[gene_model['gene_biotype'] == 'protein_coding']
        gene_model = gene_model[gene_model['Feature'] == 'gene']
        gene_model = gene_model[['Chromosome', 'Start', 'End', 'Strand', 'gene_id']]
    else:
        gene_model = pr.read_gff3(annotation_path, as_df=True)
        gene_model = gene_model[gene_model['Feature'] == 'gene']
        gene_model = gene_model[['Chromosome', 'Start', 'End', 'Strand', 'ID']]

    return gene_model


def load_annotation_msr(annotation_path):
    gene_model = pd.read_csv(annotation_path, header=None, sep=" ")

    expected_columns = ['species', 'Chromosome', 'Start', 'End', 'Strand', 'gene_id']
    dtypes = {
        'species': 'category',
        'Chromosome': 'str', 
        'Start': 'int32',
        'End': 'int32',
        'Strand': 'category',
        'gene_id': 'str'
        }
    
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


# for MSR training
def combine_files(data, file_type, file_extension, output_dir, file_key, input_filename=None, load_func=None): 

    if file_type in ['gtf', 'gff', 'gff3']:
        file_extension = 'csv'

    combined_file = f"{output_dir}/{file_type}_{file_key}.{file_extension}"
    combined_file_tpm = f"{output_dir}/{file_type}_{file_key}_{input_filename}.{file_extension}"

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(combined_file_tpm):
        print(f"Combined file {combined_file_tpm} already exists.")
        return
    
    if os.path.exists(combined_file):
        print(f"Combined file {combined_file} already exists.")
        return
    
    
    if not os.path.exists(combined_file) or not os.path.exists(combined_file_tpm):
        print(f'Now generating combined {file_type} file:')
        combined_data = []

    
        for index, row in data.iterrows():
            # in case the data lays somehwere else
            default_path = os.path.join(output_dir, row[file_type])
            fallback_path = os.path.join(row[file_type])
            
            if os.path.exists(default_path):
                file_path = default_path
            elif os.path.exists(fallback_path):
                file_path = fallback_path
 
               
            species_name = row['specie']
            species_abbr = species_name[:3]

            if os.path.exists(file_path):
                #print(f"Found file: {file_path}") 
                if file_type == 'tpm':
                    file_data = pd.read_csv(file_path)
                    # Select only the "gene_id" and "target" columns
                    file_data = file_data.loc[:, ["gene_id", "target"]]
                    combined_data.append(file_data)
                elif file_type in ['gtf', 'gff','gff3']:  # Handle both GTF and GFF
                        if load_func:  
                            file_data = load_func(file_path)
                            # Add species name as a new column
                            file_data.insert(0, 'species', species_name)
                            combined_data.append(file_data)
                        else:
                            print(f"Warning: No loading function provided for {file_type}.")
                            continue  # Skip if no loading function is available
                else:  # For fasta
                    with open(file_path, 'r') as f:
                            fasta_data = []
                            include_sequence = False

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

                                    # Modify the header if the chromosome name is numeric
                                    if chrom_name.isdigit():
                                        line = f">{chrom_name}{species_abbr}\n"
                                    
                                    # Append the modified header to `fasta_data`
                                    fasta_data.append(line.strip())

                                elif include_sequence:
                                    # Append sequence lines if the header was marked for inclusion
                                    fasta_data.append(line.strip())
                    
                    # Add the modified FASTA data as a single string block
                    combined_data.append("\n".join(fasta_data))
                       
            else:
                raise Exception(f"Warning: {file_path} does not exist.")


        # Concatenate and save combined data if files were processed
        if combined_data:
            if file_type == 'tpm':
                combined_data_df = pd.concat(combined_data, ignore_index=True)
                combined_data_df.to_csv(combined_file_tpm, index=False)
                print(f"Combined {file_type} file saved as {combined_file_tpm}")
            else:  # For fasta and GTF/GFF
                if file_type in ['gtf', 'gff','gff3']:
                    combined_data_df = pd.concat(combined_data, ignore_index=True)
                    combined_data_df.to_csv(combined_file, sep=' ', index=False, header=False)  # Save as GTF
                else:  # For fata
                    with open(combined_file, 'w') as f_out:
                        f_out.write("\n".join(combined_data))
                print(f"Combined {file_type} file saved as {combined_file}")
        else:
            print(f"No output generated.")
    
    

            
   





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

