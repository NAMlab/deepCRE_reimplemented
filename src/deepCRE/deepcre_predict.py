import argparse
import os
from typing import Any, Dict, List, Optional, Tuple
from tensorflow.keras.models import load_model #type:ignore
import pandas as pd
import numpy as np
from pyfaidx import Fasta

from deepCRE.parsing import ModelCase, ParsedInputs, RunInfo
from deepCRE.utils import get_filename_from_path, get_time_stamp, load_input_files, make_absolute_path, result_summary
from deepCRE.train_models import extract_genes_prediction, find_newest_model_path


def predict_self(extragenic: int, intragenic: int, val_entity: str, output_name: str, model_case: ModelCase,
                 extracted_genes: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]])\
                    -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any]:
    """Makes a prediction on the given val_entity using the fitting trained model.

    Args:
        extragenic (int): Number of bases to be extracted from the extragenic region
        intragenic (int): Number of bases to be extracted from the intragenic region
        val_entity (str): The validation entity (chromosome or species) to be predicted
        output_name (str): Prefix for the output file
        model_case (ModelCase): The model case to be used
        extracted_genes (Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]): Extracted genes for the given species
            or pool of species

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any]: Returns the input data, true targets, predicted probabilities,
            gene ids and the model used for prediction
    """

    x, y, gene_ids = extracted_genes[str(val_entity)]
    newest_model_paths = find_newest_model_path(output_name=output_name, val_chromosome=val_entity, model_case=model_case)
    model = load_model(newest_model_paths[val_entity])

    # Masking
    x[:, extragenic:extragenic + 3, :] = 0                                                                                                  #type:ignore
    x[:, extragenic + (intragenic * 2) + 17:extragenic + (intragenic * 2) + 20, :] = 0                                                      #type:ignore

    
    pred_probs = model.predict(x).ravel()
    return x, y, pred_probs, gene_ids, model


def run_ssr(folder_name: str, file_name: str, general_info: Dict, specie_info: Dict, genome: Fasta, annotation: pd.DataFrame,
            tpms: Optional[pd.DataFrame], extragenic: int, intragenic: int, output_name: str, time_stamp: str) -> None:
    """ Does a prediction run for a single species for ssr models.

    Args:
        folder_name (str): path to the output folder
        file_name (str): name of the current script file (used for output file naming)
        general_info (Dict): general info part of the run info for th current run
        specie_info (Dict): species info part of the run info for the current run
        genome (Fasta): Fasta object containing the genome
        annotation (pd.DataFrame): DataFrame containing information on the genes in the genome
        tpms (Optional[pd.DataFrame]): DataFrame containing the target values for the genes
        extragenic (int): bases to be extracted from the extragenic region
        intragenic (int): bases to be extracted from the intragenic region
        output_name (str): prefix for the output file
        time_stamp (str): time stamp for the output file
    """
    true_targets, preds, genes = [], [], []
    extracted_genes = extract_genes_prediction(genome=genome, annotation=annotation, extragenic=extragenic, intragenic=intragenic,
                                                           ignore_small_genes=general_info["ignore_small_genes"], tpms=tpms, target_chromosomes=())

    for chrom in specie_info["chromosomes"]:
        _, y, pred_probs, gene_ids, _ = predict_self(extragenic=extragenic, intragenic=intragenic, val_entity=str(chrom), output_name=output_name,
                                                            model_case=general_info["model_case"], extracted_genes=extracted_genes)
        true_targets.extend(y)
        preds.extend(pred_probs)
        genes.extend(gene_ids)

    result = pd.DataFrame({'true_targets': true_targets, 'pred_probs': preds, 'genes': genes})
    print(result.head())
    output_location = os.path.join(folder_name, f'{output_name}_SSR_{file_name}_{time_stamp}.csv')
    result.to_csv(output_location, index=False)


def run_msr(folder_name: str, file_name: str, general_info: Dict, extragenic: int, intragenic: int, species_name: str, time_stamp: str, extracted_genes, output_name: str) -> None:
    """ Does a prediction run for a single species for msr models.

    Args:
        folder_name (str): path to the output folder
        file_name (str): name of the current script file (used for output file naming)
        general_info (Dict): general info part of the run info for th current run
        extragenic (int): bases to be extracted from the extragenic region
        intragenic (int): bases to be extracted from the intragenic region
        species_name (str): name of the species to be predicted
        time_stamp (str): time stamp for the output file
        extracted_genes (_type_): extracted genes from the concatenated genome for the species
        output_name (str): prefix for the output file
    """
    print(f"Predicting for: {species_name}")
    _, true_targets, preds, genes, _ = predict_self(extragenic=extragenic, intragenic=intragenic, val_entity=species_name, output_name=output_name,
                                                            model_case=general_info["model_case"], extracted_genes=extracted_genes)
    result = pd.DataFrame({'true_targets': true_targets, 'pred_probs': preds, 'genes': genes})
    print(result.head())
    output_location = os.path.join(folder_name, f'{output_name}_{species_name}_MSR_{file_name}_{time_stamp}.csv')
    result.to_csv(output_location, index=False)


def check_inputs(run_info: RunInfo) -> None:
    """Checks if the input parameters are valid for the current run.

    Args:
        run_info (RunInfo): RunInfo object containing the information on the current run

    Raises:
        ValueError: raises a ValueError if species information is missing for MSR runs or if chromosome information is missing for SSR runs
    """
    gen_info = run_info.general_info
    spec_info = run_info.species_info
    if run_info.is_msr():
        for specie_data in spec_info:
            if specie_data["chromosomes"] != []:
                print(f"WARNING: chromosome information for MSR runs is not used!")
            if specie_data["species_name"] == "":
                raise ValueError(f"name of species needs to be provided!")
    else:
        for specie_data in spec_info:
            if specie_data["chromosomes"] == []:
                raise ValueError(f"chromosome information needs to be provided for SSR runs!")
        if gen_info["training_output_name"] == "":
            raise ValueError(f"Output name needs to be provided for SSR / SSC runs!")


def predict(inputs: ParsedInputs, failed_runs: List[Tuple], input_length: int, test: bool = False) -> List[Tuple[str, int, Exception]]:
    """Runs the predictions for the given input parameters.

    Args:
        inputs (ParsedInputs): ParsedInputs object containing the information on the current run
        failed_runs (List[Tuple]): List of runs that could not be parsed properly
        input_length (int): number of runs in the input file
        test (bool, optional): For testing purposes. Currently not used. Defaults to False.

    Returns:
        List[Tuple]: List of runs that could not be executed properly. Contains the output name, index of the run and the exception for each failed run.
    """
    folder_name = make_absolute_path('results', 'predictions', start_file=__file__)
    time_stamp = get_time_stamp()
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    file_name = get_filename_from_path(__file__)
    run_info: RunInfo
    for i, run_info in enumerate(inputs):       #type:ignore
        output_name = ""
        try:
            species_info = run_info.species_info
            general_info = run_info.general_info
            loaded_files = load_input_files(genome_file_name=general_info["genome"], annotation_file_name=general_info["annotation"], tpm_counts_file_name=general_info["targets"], model_case=str(general_info["model_case"]))
            genome = loaded_files["genome"]
            annotation = loaded_files["annotation"]
            tpms = loaded_files["tpms"] if "tpms" in loaded_files.keys() else None
            extragenic = general_info["extragenic"]
            intragenic = general_info["intragenic"]
            output_name = general_info["training_output_name"]

            check_inputs(run_info)
            if run_info.is_msr(): 
                extracted_genes = extract_genes_prediction(genome=genome, annotation=annotation, extragenic=extragenic, intragenic=intragenic,
                                                           ignore_small_genes=general_info["ignore_small_genes"], tpms=tpms, target_chromosomes=())
                #for specie, genome_file_name, annotation_file_name, tpm_counts_file_name, output_name, chromosome_file,_  in data.values:
                for specie_info in species_info:                                                                     # use this 
                    species_name = specie_info["species_name"]
                    run_msr(folder_name=folder_name, file_name=file_name, general_info=general_info, extragenic=extragenic,
                            intragenic=intragenic, species_name=species_name, time_stamp=time_stamp, extracted_genes=extracted_genes,
                            output_name=output_name)

            else:
                run_ssr(folder_name=folder_name, file_name=file_name, general_info=general_info, specie_info=species_info[0],
                        genome=genome, annotation=annotation, tpms=tpms, extragenic=extragenic, intragenic=intragenic,
                        output_name=output_name, time_stamp=time_stamp)
        except Exception as e:
            print(e)
            print(run_info)
            failed_runs.append((output_name, i, e))
    result_summary(failed_runs=failed_runs, input_length=input_length, script=get_filename_from_path(__file__))
    return failed_runs


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog='deepCRE',
        description="This script can be used to run predictions on models trained by the deepCRE framework."
    )

    parser.add_argument(
        '--input', "-i", 
        help="""json file containing the required input parameters. Information on pssible parameters can be found in the readme.md file.""", required=True
    )
    args = parser.parse_args()
    return args


def parse_input_file(file: str) -> Tuple[ParsedInputs, List[Tuple], int]:
    """ parse the input file and return the parsed inputs.

    Args:
        file (str): path to the input file

    Returns:
        Tuple[ParsedInputs, List[Tuple], int]: ParsedInputs object containing the information on the current run, List of runs that could not be parsed properly, number of runs in the input file
    """
    possible_general_parameters = {
        "model_case": None,
        "genome": None,
        "annotation": None,
        # if targets are provided, all genes that are not contained in the targets file are ignored
        "targets": "",
        "training_output_name": None,
        "chromosomes": "",
        "ignore_small_genes": True,
        "extragenic": 1000,
        "intragenic": 500
    }

    possible_species_parameters = {
        "species_name": "",
        "chromosomes": "",
    }
    inputs, failed_trainings, input_length = ParsedInputs.parse(file, possible_general_parameters=possible_general_parameters, possible_species_parameters=possible_species_parameters)
    inputs = inputs.replace_both()
    return inputs, failed_trainings, input_length


def main() -> None:
    """Main function for running predictions.
    """
    args = parse_args()
    inputs, failed_trainings, input_length = parse_input_file(args.input)
    predict(inputs, failed_runs=failed_trainings, input_length=input_length)


if __name__ == "__main__":
    main()
