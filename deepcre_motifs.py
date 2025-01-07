import argparse
from typing import List, Optional, Tuple
import pandas as pd
import tensorflow as tf
import os
import modisco
from importlib import reload
import h5py
from utils import get_filename_from_path, get_time_stamp, make_absolute_path, load_annotation_msr, result_summary
from deepcre_interpret import extract_scores, find_newest_interpretation_results
from parsing import ModelCase, ParsedInputs, RunInfo


def modisco_run(contribution_scores, hypothetical_scores, one_hots, output_name):
    folder_path = make_absolute_path('results', 'modisco', start_file=__file__)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    this_file_name = get_filename_from_path(__file__)
    save_file = os.path.join(folder_path, f"{output_name}_{this_file_name}_{get_time_stamp()}.hdf5")

    print('contributions', contribution_scores.shape)
    print('hypothetical contributions', hypothetical_scores.shape)
    print('correct predictions', one_hots.shape)
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
            n_cores=5)
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


def generate_motifs(genome, annot, tpm_targets, upstream, downstream, ignore_small_genes,
                    output_name, model_case, chromosome_list: Optional[List[str]], force_interpretation: bool = False):
    #just load existing scores
    if not force_interpretation:
        try: 
            saved_interpretation_results_path = find_newest_interpretation_results(output_name=output_name, results_path=os.path.join("results", "shap"))
            with h5py.File(saved_interpretation_results_path, "r") as f:
                actual_scores = f["contrib_scores"][:] #type:ignore
                hypothetical_scores = f["hypothetical_contrib_scores"][:] #type:ignore
                one_hots = f["one_hot_seqs"][:] #type:ignore
        except ValueError as e:
            force_interpretation = True
    # recalculate the scores if the user wants to force the interpretation or if the scores were not found
    if force_interpretation:
        actual_scores, hypothetical_scores, one_hots, _, _ = extract_scores(genome_file_name=genome, annotation_file_name=annot,
                                                                            tpm_counts_file_name=tpm_targets,
                                                                            upstream=upstream, downstream=downstream,
                                                                            chromosome_list=chromosome_list,
                                                                            ignore_small_genes=ignore_small_genes,
                                                                            output_name=output_name,
                                                                            model_case=model_case)
        

    print("Now running MoDisco --------------------------------------------------\n")
    print(f"Species: {output_name} \n")
    modisco_run(contribution_scores=actual_scores, hypothetical_scores=hypothetical_scores,
                one_hots=one_hots, output_name=output_name)


def parse_args():
    parser = argparse.ArgumentParser(
                        prog='deepCRE',
                        description="This script performs the deepCRE prediction. We assume you have the following three" + 
                        "directories:tmp_counts (contains your counts files), genome (contains the genome fasta files), gene_models (contains the gtf files)")

    parser.add_argument('--input', "-i",
                        help="""json file containing the required input parameters. Possible arguments can be seen in the file parsing.py in the two global dictionaries.
                        Example file is inputs.json.""", required=True)

    args = parser.parse_args()
    return args


def run_motif_extraction(inputs: ParsedInputs, failed_trainings: List[Tuple], input_length: int):
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.disable_v2_behavior()
    tf.config.set_visible_devices([], 'GPU')

    run_info: RunInfo
    for i, run_info in enumerate(inputs):       #type:ignore
        try:
            if run_info.general_info["model_case"] == ModelCase.MSR:
                # train_specie = data.copy()
                # train_specie = train_specie[train_specie['specie'] != specie]

                # output_name = "_".join([sp[:3].lower() for sp in train_specie['specie'].unique()])
                # chromosomes=""
                output_name = run_info.species_info[0]["subject_species"]

            elif run_info.general_info["model_case"] in [ModelCase.SSR, ModelCase.SSC]:
                output_name = run_info.general_info["training_output_name"]
            generate_motifs(genome=run_info.general_info["genome"], annot=run_info.general_info["annotation"], tpm_targets=run_info.general_info["targets"],
                            upstream=run_info.general_info["extragenic"], downstream=run_info.general_info["intragenic"],
                            ignore_small_genes=run_info.general_info["ignore_small_genes"], output_name=output_name,
                            model_case=run_info.general_info["model_case"], chromosome_list=run_info.general_info["chromosomes"], force_interpretation=run_info.general_info["force_interpretations"])
        except Exception as e:
            print(e)
            failed_trainings.append((output_name, i, e))

    result_summary(failed_trainings=failed_trainings, input_length=input_length, script=get_filename_from_path(__file__))


def main():
    possible_general_parameters = {
        "model_case": None,
        "genome": None,
        "annotation": None,
        "targets": None,
        "training_output_name": "",
        "chromosomes": "",
        "ignore_small_genes": True,
        "subject_species": "",
        "extragenic": 1000,
        "intragenic": 500,
        "force_interpretations": False
    }

    possible_species_parameters = {
        "subject_species": "",
    }
    args = parse_args()
    inputs, failed_trainings, input_length = ParsedInputs.parse(args.input, possible_general_parameters=possible_general_parameters, possible_species_parameters=possible_species_parameters)
    inputs = inputs.replace_both()
    print(inputs)

    run_motif_extraction(inputs, failed_trainings, input_length)


if __name__ == "__main__":
    main()
