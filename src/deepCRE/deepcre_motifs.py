import argparse
import numpy as np
from typing import List, Optional, Tuple
import pandas as pd
import tensorflow as tf
import os
import modisco
from importlib import reload
import h5py
from deepCRE.utils import get_filename_from_path, get_time_stamp, make_absolute_path, result_summary
from deepCRE.deepcre_interpret import extract_scores, find_newest_interpretation_results, get_val_obj_names
from deepCRE.parsing import ModelCase, ParsedInputs, RunInfo


def modisco_run(contribution_scores: np.ndarray, hypothetical_scores: np.ndarray, one_hots: np.ndarray, output_name: str):
    """ Runs motif extraction using modisco.

    Args:
        contribution_scores (np.ndarray): Contribution scores calculated using shap
        hypothetical_scores (np.ndarray): Hypothetical contribution scores calculated using shap
        one_hots (np.ndarray): One hot encoded input sequences
        output_name (str): prefix for the output file
    """
    folder_path = make_absolute_path('results', 'modisco', start_file=__file__)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    this_file_name = get_filename_from_path(__file__)
    save_file = os.path.join(folder_path, f"{output_name}_{this_file_name}_{get_time_stamp()}.hdf5")

    print('contributions shape', contribution_scores.shape)
    print('hypothetical contributions shape', hypothetical_scores.shape)
    print('correct predictions shape', one_hots.shape)
    # -----------------------Running modisco----------------------------------------------#

    null_per_pos_scores = modisco.coordproducers.LaplaceNullDist(num_to_samp=5000)
    tfmodisco_results = modisco.tfmodisco_workflow.workflow.TfModiscoWorkflow(
        # Slight modifications from the default settings
        sliding_window_size=15,
        flank_size=5,
        target_seqlet_fdr=0.15,
        seqlets_to_patterns_factory=modisco.tfmodisco_workflow.seqlets_to_patterns.TfModiscoSeqletsToPatternsFactory(
            trim_to_window_size=10,
            initial_flank_to_add=2,
            final_flank_to_add=0,
            final_min_cluster_size=30,
            n_cores=5
        )
    )(
        task_names=['task0'],
        contrib_scores={'task0': contribution_scores},
        hypothetical_contribs={'task0': hypothetical_scores},
        one_hot=one_hots,
        null_per_pos_scores=null_per_pos_scores)

    reload(modisco.util)
    with h5py.File(save_file, "w") as grp:
        tfmodisco_results.save_hdf5(grp)

    print(f"Done with {output_name} Modisco run")


def generate_motifs(genome_file: str, annotation_file: str, tpm_targets_file: str, extragenic: int, intragenic: int, ignore_small_genes: bool,
                    output_name: str, model_case: ModelCase, validation_object_names: List[str], force_interpretation: bool = False) -> None:
    """Generates motifs using modisco.

    Args:
        genome_file (str): name of the genome file saved in the genome folder or the path to the genome file
        annotation_file (str): name of the annotation file saved in the gene_models folder or the path to the annotation file
        tpm_targets_file (str): name of the tpm targets file saved in the tpm_counts folder or the path to the tpm targets file
        extragenic (int): number of bases to extract from the extragenic region
        intragenic (int): number of bases to extract from the intragenic region
        ignore_small_genes (bool): determines if small genes should be ignored
        output_name (str): prefix for the output file
        model_case (ModelCase): type of model used for training
        validation_object_names (List[str]): names of the validation objects
        force_interpretation (bool, optional): If set to false, saved interpretation results will be used if interpretation results have been
            saved before. Will run an interpretation run if set to true o no fitting results can be found. Defaults to False.
    """
    #just load existing scores
    if not force_interpretation:
        try: 
            # print current workind directory
            print(os.getcwd())
            print(os.path.join("results", "shap"))
            saved_interpretation_results_path = find_newest_interpretation_results(output_name=output_name, results_path=os.path.join("results", "shap"))
            with h5py.File(saved_interpretation_results_path, "r") as f:
                actual_scores = f["contrib_scores"][:] #type:ignore
                hypothetical_scores = f["hypothetical_contrib_scores"][:] #type:ignore
                one_hots = f["one_hot_seqs"][:] #type:ignore
        except ValueError as e:
            force_interpretation = True
    # recalculate the scores if the user wants to force the interpretation or if the scores were not found
    if force_interpretation:
        actual_scores, hypothetical_scores, one_hots, _, _ = extract_scores(genome_file_name=genome_file, annotation_file_name=annotation_file,
                                                                            tpm_counts_file_name=tpm_targets_file,
                                                                            extragenic=extragenic, intragenic=intragenic,
                                                                            validation_obj_names=validation_object_names,
                                                                            ignore_small_genes=ignore_small_genes,
                                                                            output_name=output_name,
                                                                            model_case=model_case)
        

    print("Now running MoDisco --------------------------------------------------\n")
    print(f"Species: {output_name} \n")
    modisco_run(contribution_scores=actual_scores, hypothetical_scores=hypothetical_scores,     #type:ignore
                one_hots=one_hots, output_name=output_name)                                     #type:ignore


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments."""
    parser = argparse.ArgumentParser(
                        prog='deepCRE',
                        description="This script can be used to run motif extraction on models trained by the deepCRE framework.")

    parser.add_argument('--input', "-i",
                        help="""json file containing the required input parameters. Possible arguments can be seen in the readme.md file.""", required=True)

    args = parser.parse_args()
    return args


def run_motif_extraction(inputs: ParsedInputs, failed_trainings: List[Tuple], input_length: int, test: bool = False) -> List[Tuple[str, int, Exception]]:
    """Runs motif extraction on the given inputs.

    Args:
        inputs (ParsedInputs): Object containing the parsed inputs from the input file
        failed_trainings (List[Tuple]): List of runs that could not be parsed
        input_length (int): Number of runs in the input file
        test (bool, optional): For testing purposes. Currently not used. Defaults to False.

    Returns:
        List[Tuple]: List of runs that could not be completed. Each tuple contains the output name, index of the run and the error that occured.
    """

    run_info: RunInfo
    for i, run_info in enumerate(inputs):       #type:ignore
        try:
            if run_info.general_info["model_case"] == ModelCase.MSR:
                output_name = run_info.general_info["training_output_name"]

            elif run_info.general_info["model_case"] in [ModelCase.SSR, ModelCase.SSC]:
                output_name = run_info.general_info["training_output_name"]
            validation_obj_names = get_val_obj_names(run_info)
            generate_motifs(genome_file=run_info.general_info["genome"], annotation_file=run_info.general_info["annotation"], tpm_targets_file=run_info.general_info["targets"],
                            extragenic=run_info.general_info["extragenic"], intragenic=run_info.general_info["intragenic"],
                            ignore_small_genes=run_info.general_info["ignore_small_genes"], output_name=output_name,
                            model_case=run_info.general_info["model_case"], validation_object_names=validation_obj_names, force_interpretation=run_info.general_info["force_interpretations"])
        except Exception as e:
            print(e)
            print(run_info)
            failed_trainings.append((output_name, i, e))

    result_summary(failed_runs=failed_trainings, input_length=input_length, script=get_filename_from_path(__file__))
    return failed_trainings


def parse_input_file(file: str) -> Tuple[ParsedInputs, List[Tuple], int]:
    """Parses the input file and returns the parsed inputs.

    Args:
        file (str): path to the input file

    Returns:
        Tuple[ParsedInputs, List[Tuple], int]: Parsed inputs, list of runs that could not be parsed and the number of runs in the input file
    """
    possible_general_parameters = {
        "model_case": None,
        "genome": None,
        "annotation": None,
        "targets": None,
        "training_output_name": None,
        "chromosomes": "",
        "ignore_small_genes": True,
        "extragenic": 1000,
        "intragenic": 500,
        "force_interpretations": False
    }
    possible_species_parameters = {
        "chromosomes": "",
        "species_name": "",
    }
    inputs, failed_runs, input_length = ParsedInputs.parse(file, possible_general_parameters=possible_general_parameters, possible_species_parameters=possible_species_parameters)
    inputs = inputs.replace_both()
    return inputs, failed_runs, input_length


def main() -> None:
    """Main function for running motif extraction.
    """
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.disable_v2_behavior()
    tf.config.set_visible_devices([], 'GPU')
    args = parse_args()
    inputs, failed_runs, input_length = parse_input_file(args.input)
    run_motif_extraction(inputs, failed_runs, input_length)


if __name__ == "__main__":
    main()
