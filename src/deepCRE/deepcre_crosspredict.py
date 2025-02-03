import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import load_model #type:ignore
from typing import List, Tuple, Dict, Any, Union
import argparse

from deepCRE.train_models import extract_genes_prediction
from deepCRE.utils import make_absolute_path, load_input_files, get_filename_from_path, get_time_stamp, result_summary
from deepCRE.parsing import ParsedInputs, RunInfo


def load_models(model_names: List[str]) -> Dict[str, Any]:
    """loads the models to be used for predictions.

    Args:
        model_names (List[str]): List of model names to be used for predictions. Can be the file name of models in
            the saved_models folder or the full path to the model.

    Raises:
        ValueError: Is raised if two models have the same file names.

    Returns:
        Dict[str, Any]: Dictionary containing the loaded models. Key is the model name, value is the loaded model.
    """
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
    return models


def predict_other(extragenic: int, intragenic: int, curr_chromosome: str, model_names: List[str],
                  extracted_genes: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """creates predictions for a given chromosome using the provided models.

    Args:
        extragenic (int): number of base pairs to be used for extragenic extraction.
        intragenic (int): number of base pairs to be used for intragenic extraction.
        curr_chromosome (str): name of the chromosome to be used for predictions.
        model_names (List[str]): list of model names to be used for predictions. Can be the file name of models in
            the saved_models folder or the full path to the model.
        extracted_genes (Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]): dictionary containing the extracted genes for each chromosome.
            Key are the chromosome names, values are tuples containing the extracted genes one hot encoded, the true targets and the gene ids
            all as numpy arrays.

    Raises:
        ValueError: Is raised if two models have the same file names.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: Dataframe containing the predictions and the models used for the predictions.
    """
    no_genes = False
    try:
        x, y, gene_ids = extracted_genes[curr_chromosome]
        # Masking
        x[:, extragenic:extragenic + 3, :] = 0                                                                                                  #type:ignore
        x[:, extragenic + (intragenic * 2) + 17:extragenic + (intragenic * 2) + 20, :] = 0                                                      #type:ignore
    except KeyError:
        no_genes = True
        print(f"no genes found for Chromosome \"{curr_chromosome}\"")

    models = load_models(model_names)

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


def parse_args() -> argparse.Namespace:
    """Parses the command line arguments.

    Returns:
        argparse.Namespace: parsed arguments
    """
    parser = argparse.ArgumentParser(
                        prog='deepCRE',
                        description="This script can be used to run predictions on models trained by the deepCRE framework." +
                            "Models dont have to be trained on the species that the predictions are run on. The predictions will be saved in the results/predictions folder.")
    parser.add_argument('--input', "-i", 
                        help="Path to the input file. Details on the input file can be found in the README.md file.",
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
    """reads a csv file into a pandas dataframe.

    Args:
        path (str): path to the csv file to be read.

    Returns:
        pd.DataFrame: dataframe containing the data from the csv file.
    """
    return pd.read_csv(path, sep=',', na_values={"target_classes": [], "chromosome_selection": [], "ignore_small_genes": [], "intragenic_extraction_length": [], "extragenic_extraction_length":[]}, keep_default_na=False)


def get_output_location(run_info: RunInfo, folder_name: str, model_file_name: str, file_name: str, time_stamp: str) -> str:
    """gets the output location for the predictions.

    Args:
        run_info (RunInfo): RunInfo object containing the information for the current run.
        folder_name (str): path to the folder where the predictions will be saved if no output path is given in the input file.
        model_file_name (str): name of the model file used for the predictions.
        file_name (str): name of the file containing the script.
        time_stamp (str): time stamp for the current run.

    Raises:
        ValueError: Is raised if neither output_path nor output_base are given.

    Returns:
        str: path to the output file for the predictions.
    """
    if not run_info.general_info["output_path"] and not run_info.general_info["output_base"]:
        raise ValueError("Neither output_path nor output_base are given! Giving a value for output_base will auto generate a save location within the results/predictions folder." +
                         "output_path should be the path to the desired save location and will override the auto generated file location.")
    output_location = run_info.general_info["output_path"] if run_info.general_info["output_path"] else os.path.join(folder_name, f'{model_file_name}_{file_name}_{run_info.general_info["output_base"]}_{time_stamp}.csv')
    while os.path.exists(output_location):
        print(f"Warning: output path {output_location} already exists!")
        base, ext = os.path.splitext(output_location)
        output_location = base + "_1" + ext
    return output_location
        

def run_cross_predictions(run_infos: ParsedInputs, failed_runs: List[Tuple], input_length: int, test: bool = False) -> List[Tuple[str, int, Exception]]:
    """runs predictions on the genomes with the related models provided in the run_infos.

    Args:
        run_infos (ParsedInputs): ParsedInputs object containing the information for the runs to be executed.
        failed_runs (List[Tuple]): list containing the failed runs.
        input_length (int): number of runs in the input file.
        test (bool, optional): supposed to indicate test runs. currently unused but important for running tests. Defaults to False.

    Returns:
        List[Tuple]: returns the updated list of failed runs.
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
            if model_case == "msr" and chromosomes_tuple == ():
                chromosomes = extracted_genes.keys()
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
            print(run_info)
            failed_runs.append((f"{model_file_name} -> {run_info.general_info['output_base']}", i, e))

    result_summary(failed_runs=failed_runs, input_length=input_length, script=get_filename_from_path(__file__))
    return failed_runs


def parse_input_file(file: str) -> Tuple[ParsedInputs, List[Tuple[str, int, Exception]], int]:
    """parses the input file and returns the parsed inputs.

    Args:
        file (str): path to the input file.

    Returns:
        Tuple[ParsedInputs, List[Tuple[str, int, Exception]], int]: returns the parsed inputs, the runs that were not able to be parsed and the number of runs in the input file.
    """
    possible_general_parameters = {
        "genome": None,
        "annotation": None,
        "targets": "",
        "prediction_models": None,
        "output_base": "",
        "output_path": "",
        "chromosomes": "",
        "ignore_small_genes": True,
        "extragenic": 1000,
        "intragenic": 500
    }

    possible_species_parameters = {
        "chromosomes": "",
    }
    inputs, failed_runs, input_length = ParsedInputs.parse(file, possible_general_parameters=possible_general_parameters, possible_species_parameters=possible_species_parameters, allow_multiple_species=False)
    inputs = inputs.replace_both()
    return inputs, failed_runs, input_length



def main() -> None:
    """Main function for running cross predictions.
    """
    args = parse_args()
    inputs, failed_runs, input_length = parse_input_file(args.input)
    run_cross_predictions(inputs, failed_runs=failed_runs, input_length=input_length)


if __name__ == "__main__":
    main()
