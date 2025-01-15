import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import load_model #type:ignore
from typing import List, Tuple, Dict, Any, Union
import argparse

from deepCRE.train_models import extract_genes_prediction
from deepCRE.utils import make_absolute_path, load_input_files, get_filename_from_path, get_time_stamp, result_summary
from deepCRE.parsing import ModelCase, ParsedInputs, RunInfo


def predict_other(extragenic: int, intragenic: int, curr_chromosome: str, model_names: List[str], extracted_genes: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    no_genes = False
    try:
        x, y, gene_ids = extracted_genes[curr_chromosome]
        # Masking
        x[:, extragenic:extragenic + 3, :] = 0                                                                                                  #type:ignore
        x[:, extragenic + (intragenic * 2) + 17:extragenic + (intragenic * 2) + 20, :] = 0                                                      #type:ignore
    except KeyError:
        no_genes = True
        print(f"no genes found for Chromosome \"{curr_chromosome}\"")


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

    if no_genes:
        df_dict = {model_name: np.zeros((0)) for model_name, model in models.items()}
    else:
        df_dict = {model_name: model.predict(x).ravel() for model_name, model in models.items()}
    result_df = pd.DataFrame(df_dict)
    result_df["pred_probs"] = result_df.mean(axis=1)
    if no_genes:
        result_df['true_targets'] = result_df["pred_probs"]
        result_df['genes'] = result_df["pred_probs"]
    else:
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


def get_chromosomes(chromosomes: List[str], annotation: pd.DataFrame) -> Tuple[List, Tuple]:
    """loads the chromosomes to be used for predictions.

    Depending on the inputs, the chromosomes will be extracted from the file provided in the chomosome_selction column. If no file is provided there, all chromosomes from the annotation file will be used.

    Args:
        chromosomes_file (str): path to the file containing the chromosomes to be used. If this is the empty string, all chromosomes from annotation file will be used instead.
        annotation (pd.DataFrame): annotation for the genome to do the predictions on.

    Returns:
        Tuple[List, Tuple]: List and tuple containing the names of the chromosomes to be used. If the chromosomes file is empty, the tuple will be empty.
    """
    if chromosomes:
        chromosomes_tuple = tuple(chromosomes)
        chromosomes_list = chromosomes
    else:
        chromosomes_tuple = ()
        chromosomes_list = list(annotation["Chromosome"].unique())
    return chromosomes_list,chromosomes_tuple


def read_df(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=',', na_values={"target_classes": [], "chromosome_selection": [], "ignore_small_genes": [], "intragenic_extraction_length": [], "extragenic_extraction_length":[]}, keep_default_na=False)


def get_output_location(run_info: RunInfo, folder_name: str, model_file_name: str, file_name: str, time_stamp: str) -> str:
    if not run_info.general_info["output_path"] and not run_info.general_info["output_base"]:
        raise ValueError("Neither output_path nor output_base are given! Giving a value for output_base will auto generate a save location within the results/predictions folder." +
                         "output_path should be the path to the desired save location and will override the auto generated file location.")
    output_location = run_info.general_info["output_path"] if run_info.general_info["output_path"] else os.path.join(folder_name, f'{model_file_name}_{file_name}_{run_info.general_info["output_base"]}_{time_stamp}.csv')
    while os.path.exists(output_location):
        print(f"Warning: output path {output_location} already exists!")
        base, ext = os.path.splitext(output_location)
        output_location = base + "_1" + ext
    return output_location
        

def run_cross_predictions(run_infos: ParsedInputs, failed_trainings: List[Tuple], input_length: int, test: bool = False) -> List[Tuple]:
    """runs cross predictions as specified by the input file.

    Args:
        data (Union[pd.DataFrame, None]): input data frame. Usually will be loaded from input file. Can be provided directly for testing purposes.
    """
    time_stamp = get_time_stamp()
    folder_name = make_absolute_path('results', 'predictions', start_file=__file__)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    file_name = get_filename_from_path(__file__)
    run_info: RunInfo
    for i, run_info in enumerate(run_infos):           #type:ignore
        try:
            models = run_info.general_info["prediction_models"]
            model_file_name = get_filename_from_path(models[0])
            output_location = get_output_location(run_info=run_info, folder_name=folder_name, model_file_name=model_file_name, file_name=file_name, time_stamp=time_stamp)
            model_case = "msr" if run_info.general_info["annotation"].endswith(".csv") else "ssr"
            loaded_input_files = load_input_files(genome_file_name=run_info.general_info["genome"], annotation_file_name=run_info.general_info["annotation"], tpm_counts_file_name=run_info.general_info["targets"], model_case=model_case)
            chromosomes, chromosomes_tuple = get_chromosomes(run_info.species_info[0]["chromosomes"], loaded_input_files["annotation"])
            extracted_genes = extract_genes_prediction(genome=loaded_input_files["genome"], annotation=loaded_input_files["annotation"], extragenic=run_info.general_info["extragenic"],
                                                       intragenic=run_info.general_info["intragenic"], ignore_small_genes=run_info.general_info["ignore_small_genes"],
                                                       tpms=loaded_input_files.get("tpms", None), target_chromosomes=chromosomes_tuple)
            results_dfs = []
            for chrom in chromosomes:
                results, _ = predict_other(extragenic=run_info.general_info["extragenic"], intragenic=run_info.general_info["intragenic"], curr_chromosome=chrom,
                                           model_names=models, extracted_genes=extracted_genes)
                results_dfs.append(results)
            result = pd.concat(results_dfs)
            print(result.head())
            result.to_csv(output_location, index=False)
        except Exception as e:
            print(e)
            failed_trainings.append((f"{model_file_name} -> {run_info.general_info['output_base']}", i, e))

    result_summary(failed_trainings=failed_trainings, input_length=input_length, script=get_filename_from_path(__file__))
    return failed_trainings


def parse_input_file(file: str):
    possible_general_parameters = {
        "genome": None,
        "annotation": None,
        "targets": "",
        "output_base": "",
        "output_path": "",
        "chromosomes": "",
        "prediction_models": None,
        "ignore_small_genes": True,
        "extragenic": 1000,
        "intragenic": 500
    }

    possible_species_parameters = {
        "chromosomes": "",
    }
    inputs, failed_trainings, input_length = ParsedInputs.parse(file, possible_general_parameters=possible_general_parameters, possible_species_parameters=possible_species_parameters, allow_multiple_species=False)
    inputs = inputs.replace_both()
    print(inputs)
    return inputs, failed_trainings, input_length



def main():
    args = parse_args()
    inputs, failed_trainings, input_length = parse_input_file(args.input)
    run_cross_predictions(inputs, failed_trainings=failed_trainings, input_length=input_length)


if __name__ == "__main__":
    main()
