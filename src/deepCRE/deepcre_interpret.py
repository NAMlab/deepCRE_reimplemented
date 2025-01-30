import argparse
import os
from typing import List, Dict, Optional, Tuple
import pandas as pd
import tensorflow as tf
import h5py
import numpy as np
from deeplift.dinuc_shuffle import dinuc_shuffle
import shap
from pyfaidx import Fasta 
import re

from deepCRE.utils import get_time_stamp, get_filename_from_path, load_input_files, make_absolute_path, load_annotation_msr, result_summary
from deepCRE.deepcre_predict import predict_self
from deepCRE.train_models import extract_genes_prediction
from deepCRE.parsing import ModelCase, ParsedInputs, RunInfo


def find_newest_interpretation_results(output_name: str, results_path: str = "") -> str:
    """finds path to newest model fitting the given parameters

    Args:
        output_name (str): output name the was used for creating the predictions in the first place
        results_path (str): path to the directory where prediction results are stored.

    Raises:
        ValueError: raises an error if no fitting model is found

    Returns:
        str: Path to the newest prediction results for the given output name.
    """
    if results_path == "":
        path_to_interpretations = make_absolute_path("results", "shap", start_file=__file__)
    else:
        path_to_interpretations = make_absolute_path(results_path, start_file=__file__)
    # ^ and $ mark start and end of a string. \d singnifies any digit. \d+ means a sequence of digits with at least length 1
    regex_string = f"^{output_name}_deepcre_interpret_\d+_\d+\.h5$"                                                        #type:ignore
    regex = re.compile(regex_string)
    candidate_results = [result for result in os.listdir(path_to_interpretations) if regex.match(result)]
    if not candidate_results:
        raise ValueError("no interpretation results fitting the given parameters were found! Consider running the interpretation script (deepcre_interpret.py)")
    # models only differ in the time stamp. So if sorted, the last model will be the most recently trained
    candidate_results.sort()
    full_path = os.path.join(path_to_interpretations, candidate_results[-1])
    return full_path


# 1. Shap
def dinuc_shuffle_several_times(list_containing_input_modes_for_an_example, seed=1234):
    assert len(list_containing_input_modes_for_an_example) == 1
    onehot_seq = list_containing_input_modes_for_an_example[0]
    rng = np.random.RandomState(seed)
    to_return = np.array([dinuc_shuffle(onehot_seq, rng=rng) for i in range(50)])

    return [to_return]


def combine_mult_and_diffref(mult, orig_inp, bg_data):
    to_return = []
    for l in range(len(mult)):
        projected_hypothetical_contribs = np.zeros_like(bg_data[l]).astype("float")
        assert len(orig_inp[l].shape) == 2
        for i in range(orig_inp[l].shape[-1]):
            hypothetical_input = np.zeros_like(orig_inp[l]).astype("float")
            hypothetical_input[:, i] = 1.0
            hypothetical_difference_from_reference = (hypothetical_input[None, :, :] - bg_data[l])
            hypothetical_contribs = hypothetical_difference_from_reference * mult[l]
            projected_hypothetical_contribs[:, :, i] = np.sum(hypothetical_contribs, axis=-1)
        to_return.append(np.mean(projected_hypothetical_contribs, axis=0))
    return to_return


def compute_actual_hypothetical_scores(x, model):
    """
    This function computes the actual hypothetical scores given a model.

    :param x: onehot encodings of correctly predicted sequences
    :param model: loaded keras model used for predictions
    """
    shap.explainers.deep.deep_tf.op_handlers["AddV2"] = shap.explainers.deep.deep_tf.passthrough #type:ignore
    shap.explainers.deep.deep_tf.op_handlers["FusedBatchNormV3"] = shap.explainers.deep.deep_tf.linearity_1d(0)#type:ignore
    dinuc_shuff_explainer = shap.DeepExplainer(
        (model.input, model.layers[-2].output[:, 0]),
        data=dinuc_shuffle_several_times,
        combine_mult_and_diffref=combine_mult_and_diffref)
    hypothetical_scores = dinuc_shuff_explainer.shap_values(x)
    actual_scores = hypothetical_scores * x
    return actual_scores, hypothetical_scores


def single_score_calc(upstream: int, downstream: int, output_name: str, model_case: ModelCase, shap_actual_scores,
                      shap_hypothetical_scores, one_hots_seqs: List[np.ndarray], gene_ids_seqs: List[np.ndarray],
                      preds_seqs: List[np.ndarray], extracted_genes: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
                      val_chrom: str, validation_text: str):
    x, y, preds, gene_ids, model = predict_self(extragenic=upstream, intragenic=downstream, val_chromosome=val_chrom,
                                                output_name=output_name, model_case=model_case, extracted_genes=extracted_genes)
    preds = preds > 0.5
    preds = preds.astype(int)
    correct_x, correct_y, correct_gene_ids = [], [], []
    for idx in range(x.shape[0]): #type:ignore
        if preds[idx] == y[idx]:
            correct_x.append(x[idx])
            correct_y.append(y[idx])
            correct_gene_ids.append(gene_ids[idx])

    correct_x = np.array(correct_x)

    # Compute scores
    print(f"\n-----------------------------------------")
    print(f"running shap with validation on {validation_text}")
    print(f"-----------------------------------------\n")

    actual_scores, hypothetical_scores = compute_actual_hypothetical_scores(x=correct_x, model=model)
    shap_actual_scores.append(actual_scores)
    shap_hypothetical_scores.append(hypothetical_scores)
    one_hots_seqs.append(correct_x)
    gene_ids_seqs.extend(correct_gene_ids)
    preds_seqs.extend(correct_y)


def extract_scores(genome_file_name, annotation_file_name, tpm_counts_file_name, upstream, downstream, validation_obj_names: List[str], ignore_small_genes,
                   output_name, model_case):
    """
    This function performs predictions, extracts correct predictions and performs shap computations. This will be
    done iteratively per chromosome.

    :param genome: genome fasta file
    :param annot: gtf annotation file
    :param tpm_targets: targets file; must have a target column
    :param upstream: 1000
    :param downstream: 500
    :param n_chromosome: total number of chromosomes in the species
    :param ignore_small_genes: whether to ignore small genes
    :param output_name: prefix name used to create output files
    :param model_case: SSR, SSC or MSR
    :return: actual scores, hypothetical scores and one hot encodings of correct predictions across the entire genome
    """
    folder_path = make_absolute_path("results", "shap")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    shap_actual_scores, shap_hypothetical_scores, one_hots_seqs, gene_ids_seqs, preds_seqs = [], [], [], [], []
    
    loaded_input_files = load_input_files(genome_file_name=genome_file_name, annotation_file_name=annotation_file_name,
                                          tpm_counts_file_name=tpm_counts_file_name, model_case=str(model_case))
    genome = loaded_input_files["genome"]
    annotation = loaded_input_files["annotation"]
    tpms = loaded_input_files["tpms"]
    extracted_genes = extract_genes_prediction(genome, annotation, extragenic=upstream, intragenic=downstream,
                                               ignore_small_genes=ignore_small_genes, tpms=tpms, target_chromosomes=())

    for val_obj_name in validation_obj_names:
        validation_text = f"{val_obj_name} from {output_name}"
        single_score_calc(upstream=upstream, downstream=downstream, output_name=output_name, model_case=model_case,
                          shap_actual_scores=shap_actual_scores, shap_hypothetical_scores=shap_hypothetical_scores,
                          one_hots_seqs=one_hots_seqs, gene_ids_seqs=gene_ids_seqs, preds_seqs=preds_seqs,
                          extracted_genes=extracted_genes, val_chrom=val_obj_name, validation_text=validation_text)

    shap_actual_scores = np.concatenate(shap_actual_scores, axis=0)
    shap_hypothetical_scores = np.concatenate(shap_hypothetical_scores, axis=0)
    one_hots_seqs = np.concatenate(one_hots_seqs, axis=0)
    save_results(shap_actual_scores=shap_actual_scores, shap_hypothetical_scores=shap_hypothetical_scores,
                 output_name=output_name, gene_ids_seqs=gene_ids_seqs, preds_seqs=preds_seqs, one_hot_seqs=one_hots_seqs)

    return shap_actual_scores, shap_hypothetical_scores, one_hots_seqs, gene_ids_seqs, preds_seqs


def save_results(output_name: str, shap_actual_scores, shap_hypothetical_scores, gene_ids_seqs: List, preds_seqs: List, one_hot_seqs: np.ndarray):
    folder_name = make_absolute_path("results", "shap")
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    file_name = get_filename_from_path(__file__)
    h5_file_name = os.path.join(folder_name, f'{output_name}_{file_name}_{get_time_stamp()}.h5')
    with h5py.File(name=h5_file_name, mode='w') as h5_file:
        h5_file.create_dataset(name='contrib_scores', data=shap_actual_scores)
        h5_file.create_dataset(name="hypothetical_contrib_scores", data=shap_hypothetical_scores)
        h5_file.create_dataset(name="one_hot_seqs", data=one_hot_seqs)
        save_path = make_absolute_path('results', 'shap', f'{output_name}_{file_name}_{get_time_stamp()}_shap_meta.csv', start_file=__file__)
        pd.DataFrame({'gene_ids': gene_ids_seqs, 'preds': preds_seqs}).to_csv(path_or_buf=save_path, index=False)


def parse_args():
    parser = argparse.ArgumentParser(
                        prog='deepCRE',
                        description="This script performs calculations of contribution scores for models trained wit the deepCRE framework.")

    parser.add_argument('--input', "-i",
                        help="""json file containing the required input parameters. For information on the possible parameters, refer to the readme.md file.""", required=True)

    args = parser.parse_args()
    return args


def check_species_info(run_info: RunInfo):
    if run_info.is_msr():
        for specie_data in run_info.species_info:
            if specie_data["species_name"] == "":
                raise ValueError(f"name of species needs to be provided for MSR runs! Species names cant be empty strings.")
    else:
        if len(run_info.species_info) > 1:
            raise ValueError("only one species can be provided for SSR/SSC model cases!")


def get_val_obj_names(run_info: RunInfo) -> List[str]:
    model_case = run_info.general_info["model_case"]
    if model_case in [ModelCase.SSR, ModelCase.SSC]:
        if len(run_info.species_info) > 1:
            raise ValueError("only one species can be provided for SSR/SSC model cases!")
        val_obj_names = run_info.species_info[0]["chromosomes"]
        if val_obj_names is None or not val_obj_names:
            raise ValueError("chromosome list must be provided for SSR/SSC model cases!")
    elif model_case == ModelCase.MSR:
        val_obj_names = [specie_info["species_name"] for specie_info in run_info.species_info]
        if not val_obj_names:
            raise ValueError("species names must be provided for MSR model case!")
    for val_obj_name in val_obj_names:
        if val_obj_name == "":
            if model_case == ModelCase.MSR:
                raise ValueError("species names must be provided for MSR model case! Species names cant be empty strings.")
            else:
                raise ValueError("chromosome names must be provided for SSR/SSC model cases! Chromosome names cant be empty strings.")
    return val_obj_names


def run_interpretation(inputs: ParsedInputs, failed_trainings: List[Tuple], input_length: int, test: bool = False) -> List[Tuple]:
    for i, run_info in enumerate(inputs):     #type:ignore
        output_name = ""
        try: 
            check_species_info(run_info)
            output_name = run_info.general_info["training_output_name"]
            val_obj_names = get_val_obj_names(run_info)
            extract_scores(genome_file_name=run_info.general_info["genome"], annotation_file_name=run_info.general_info["annotation"],
                            tpm_counts_file_name=run_info.general_info["targets"], upstream=run_info.general_info["extragenic"],
                            downstream=run_info.general_info["intragenic"], validation_obj_names=val_obj_names,
                            ignore_small_genes=run_info.general_info["ignore_small_genes"], output_name=output_name,
                            model_case=run_info.general_info["model_case"])
        except Exception as e:
            print(e)
            print(run_info)
            failed_trainings.append((output_name, i, e))
                
    result_summary(failed_trainings=failed_trainings, input_length=input_length, script=get_filename_from_path(__file__))
    return failed_trainings


def parse_input_file(file: str):
    possible_general_parameters = {
        "model_case": None,
        "genome": None,
        "annotation": None,
        # genes not included in the target file will be ignored
        "targets": None,
        "training_output_name": None,
        "chromosomes": "",
        "ignore_small_genes": True,
        "extragenic": 1000,
        "intragenic": 500,
    }

    possible_species_parameters = {
        "species_name": "",
        "chromosomes": "",
    }
    inputs, failed_trainings, input_length = ParsedInputs.parse(file, possible_general_parameters=possible_general_parameters, possible_species_parameters=possible_species_parameters)
    inputs = inputs.replace_both()
    return inputs, failed_trainings, input_length


def main():
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.disable_v2_behavior()
    tf.config.set_visible_devices([], 'GPU')
    args = parse_args()
    inputs, failed_trainings, input_length = parse_input_file(args.input)
    run_interpretation(inputs=inputs, failed_trainings=failed_trainings, input_length=input_length)


if __name__ == "__main__":
    main()
