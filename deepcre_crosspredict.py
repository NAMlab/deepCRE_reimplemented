import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import load_model #type:ignore
from typing import List, Tuple, Dict, Any, Union
import argparse

from train_ssr_models import extract_genes
from utils import make_absolute_path, load_input_files, get_filename_from_path, get_time_stamp, result_summary


def predict_other(extragenic: int, intragenic: int, val_chromosome: str, model_names: List[str], extracted_genes: Dict[str, Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]]) -> Tuple[pd.DataFrame, Dict[str, Any]]:

    x, y, gene_ids = extracted_genes[val_chromosome]

    # Masking
    x[:, extragenic:extragenic + 3, :] = 0                                                                                                  #type:ignore
    x[:, extragenic + (intragenic * 2) + 17:extragenic + (intragenic * 2) + 20, :] = 0                                                      #type:ignore

    path_to_models = make_absolute_path("saved_models", start_file=__file__)
    model_names_dict = {}
    for model_name in model_names:
        # if model name is a full path to a model: only use basename (filename + file extension) as identifier. 
        if os.path.isfile(model_name):
            identifier = os.path.basename(model_name)
            full_model_path = model_name
        else:
            identifier = model_name
            full_model_path = os.path.join(path_to_models, model_name)
        if identifier in model_names_dict.keys():
            raise ValueError(f"two models have the file name \"{identifier}\". Please make sure that the file names for all used models are unique.")
        else:
            model_names_dict[identifier] = full_model_path

    models = {model_name: load_model(model_path) for model_name, model_path in model_names_dict.items()}

    df_dict = {model_name: model.predict(x).ravel() for model_name, model in models.items()}
    result_df = pd.DataFrame(df_dict)
    result_df["pred_probs"] = result_df.mean(axis=1)
    result_df['true_targets'] = y
    result_df['genes'] = gene_ids
    return result_df, models


def parse_args():
    parser = argparse.ArgumentParser(
                        prog='deepCRE',
                        description="This script performs the deepCRE prediction. We assume you have the following three" + 
                        "directories:tmp_counts (contains your counts files), genome (contains the genome fasta files), gene_models (contains the gtf files)")

    parser.add_argument('--input', "-i", 
                        help="path to a .csv file with column names. Must contain \"genome\", \"gene_model\", \"model_names\", \"subject_species_name\".",
                        required=True)

    args = parser.parse_args()
    return args


def check_input(data: pd.DataFrame) -> bool:
    """validates whether the essential columns are present in the input data frame

    Args:
        data (pd.DataFrame): the input data frame

    Raises:
        ValueError: raised if essential columns are missing
    """
    present_cols = data.columns
    necessary_columns = ["genome", "gene_model", "model_names", "subject_species_name"]
    missing_columns = []
    for necessary_column in necessary_columns:
        if necessary_column not in present_cols:
            missing_columns.append(necessary_column)
    
    if missing_columns:
        raise ValueError(f"column/s {missing_columns} not found. The following columns are absolutely necessary: {necessary_columns}")
    return True

    
def get_optional_values(row: pd.Series) -> Tuple[str, str, str, str, str]:
    """gets the values for all optional columns out of a row.

    Args:
        row (pd.Series): current row of the input data frame

    Returns:
        Tuple[str, str, bool]: Tuple with path to file with target_classes, path to file with selected chromosomes, flag whether small genes should be ignored, sequence length of the extragenic part to be extracted,
            sequence length of the intragenic part to be extracted
    """
    results = {}
    for column in ["target_classes", "chromosome_selection", "ignore_small_genes", "intragenic_extraction_length", "extragenic_extraction_length"]:
        if column in row.index:
            results[column] = row[column]
        else:
            results[column] = ""
    return results["target_classes"], results["chromosome_selection"], results["ignore_small_genes"], results["intragenic_extraction_length"], results["extragenic_extraction_length"]


def set_defaults(ignore_small_genes: str, intragenic: str, extragenic: str) -> Tuple[bool, int, int]:
    """parses strings from optional inputs and sets defaults where applicable

    Args:
        ignore_small_genes (str): input for ignore_small_genes
        intragenic (str): input for length of sequence to extract inside of the gene
        extragenic (str): input for length of sequence to extract outside of the gene

    Raises:
        ValueError: raises error if the inputs can not be parsed correctly

    Returns:
        Tuple[bool, int, int]: returns the values for ignore_small_genes, intragenic and extragenic.
    """
    if ignore_small_genes == "yes":
        ignore_small_genes_res = True
    elif ignore_small_genes.lower() in ["no", ""]:
        ignore_small_genes_res = False
    else:
        raise ValueError(f"Value \"{ignore_small_genes}\" for ignore small genes could not be parsed. Allowed values are \"yes\" or \"no\". Leaving the column empty or omitting the column entirely will default to using all genes, so ignore_small_genes = False")
    
    extragenic_res = 1000 if extragenic == "" else int(extragenic)
    intragenic_res = 500 if intragenic == "" else int(intragenic)

    assert extragenic_res >= 0
    assert intragenic_res >= 0
    
    return ignore_small_genes_res, intragenic_res, extragenic_res


def parse_model_names(model_names: str) -> List[str]:
    """splits model names from the input file

    Args:
        model_names (str): model names separated by \";\"

    Returns:
        List[str]: List of model names, stripped of white spaces
    """
    results = [model.strip() for model in model_names.split(";")]
    if results:
        return results
    else:
        raise ValueError(f"no model names were found in column columns \"model_names\". At least one model in necessary to run predictions. Additional models can be ginve in the same columns, separated with semicolons \";\".")

def get_chromosomes(chromosomes_file: str, annotation: pd.DataFrame) -> Tuple[List, Tuple]:
    """loads the chromosomes to be used for predictions.

    Depending on the inputs, the chromosomes will be extracted from the file provided in the chomosome_selction column. If no file is provided there, all chromosomes from the annotation file will be used.

    Args:
        chromosomes_file (str): path to the file containing the chromosomes to be used. If this is the empty string, all chromosomes from annotation file will be used instead.
        annotation (pd.DataFrame): annotation for the genome to do the predictions on.

    Returns:
        Tuple[List, Tuple]: List and tuple containing the names of the chromosomes to be used. If the chromosomes file is empty, the tuple will be empty.
    """
    if chromosomes_file:
        chromosomes = pd.read_csv(chromosomes_file, header=None, dtype={0: str})
        chromosomes_tuple = tuple(chromosomes[0].values)
        chromosomes = list(chromosomes[0].values)
    else:
        chromosomes_tuple = ()
        chromosomes = list(annotation["Chromosome"].unique())
    return chromosomes,chromosomes_tuple


def get_required_values(row: pd.Series, failed_trainings: List, i):
    """reads the values off the required columns

    Args:
        row (pd.Series): current row of the input file
        failed_trainings (List): List of Tuples containing information on failed rows of the input file.
        i: line of the input data frame.

    Returns:
        Tuple[str, str, str, List[str], bool]: the extracted values as well as a flag comunicating whether extracted values are empty.
    """
    genome_file_name, annotation_file_name, subject_species_name  = row["genome"], row["gene_model"], row["subject_species_name"]
    error = False
    for col in [genome_file_name, annotation_file_name, subject_species_name]:
        if col == "":
            failed_trainings.append((f"", i, ValueError(f"input line is missing an entry in column \"{col}\"")))
            error = True
    models = parse_model_names(row["model_names"])
    return genome_file_name,annotation_file_name,subject_species_name,models, error
        

def run_cross_predictions(data: Union[pd.DataFrame, None] = None):
    """runs cross predictions as specified by the input file.

    Args:
        data (Union[pd.DataFrame, None]): input data frame. Usually will be loaded from input file. Can be provided directly for testing purposes.
    """
    if data is None:
        args = parse_args()
        data = pd.read_csv(args.input, sep=',')
    check_input(data)
    print(data.head())
    folder_name = make_absolute_path('results', 'predictions', start_file=__file__)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    file_name = get_filename_from_path(__file__)

    failed_trainings, passed_trainings = [],[]
    for i, row in data.iterrows():
        try:
            values = get_required_values(row=row, failed_trainings=failed_trainings, i=i)
            genome_file_name, annotation_file_name, subject_species_name, models, failed_extraction = values
            if failed_extraction:
                continue
        
            target_class_file, chromosomes_file, ignore_small_genes, intragenic, extragenic = get_optional_values(row)

            model_file_name = get_filename_from_path(models[0])
            ignore_small_genes, intragenic, extragenic = set_defaults(ignore_small_genes=ignore_small_genes, intragenic=intragenic, extragenic=extragenic)
            loaded_input_files = load_input_files(genome_file_name=genome_file_name, annotation_file_name=annotation_file_name, tpm_counts_file_name=target_class_file)
            chromosomes, chromosomes_tuple = get_chromosomes(chromosomes_file, loaded_input_files["annotation"])
            extracted_genes = extract_genes(genome=loaded_input_files["genome"], annotation=loaded_input_files["annotation"], extragenic=extragenic, intragenic=intragenic,
                                            ignore_small_genes=ignore_small_genes, tpms=loaded_input_files.get("tpms", None), target_chromosomes=chromosomes_tuple, for_prediction=True)
            results_dfs = []
            for chrom in chromosomes:
                results, _ = predict_other(extragenic=extragenic, intragenic=intragenic, val_chromosome=str(chrom),
                                           model_names=models, extracted_genes=extracted_genes)
                results_dfs.append(results)
            result = pd.concat(results_dfs)
            print(result.head())
            output_location = os.path.join(folder_name, f'{model_file_name}_{file_name}_{subject_species_name}_{get_time_stamp()}.csv')
            result.to_csv(output_location, index=False)
            passed_trainings.append((f"{model_file_name} -> {subject_species_name}", i))
        except Exception as e:
            print(e)
            failed_trainings.append((f"{model_file_name} -> {subject_species_name}", i, e))

    result_summary(failed_trainings=failed_trainings, passed_trainings=passed_trainings, input_length=len(data), script=get_filename_from_path(__file__))


if __name__ == "__main__":
    run_cross_predictions()