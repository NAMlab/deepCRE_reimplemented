import argparse
import os
from typing import List, Dict
import pandas as pd
import tensorflow as tf
import h5py
import numpy as np
from deeplift.dinuc_shuffle import dinuc_shuffle
import shap
from pyfaidx import Fasta 

from utils import get_time_stamp, get_filename_from_path, load_input_files, make_absolute_path, load_annotation_msr
from deepcre_predict import predict_self
from train_models import extract_genes


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
    :return:
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


def extract_scores(genome_file_name, annotation_file_name, tpm_counts_file_name, upstream, downstream, chromosome_list: pd.DataFrame, ignore_small_genes,
                   output_name, model_case, train_val_split):
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
    
    if model_case.lower() in ["ssr", "ssc"]:
        loaded_input_files = load_input_files(genome_file_name=genome_file_name, annotation_file_name=annotation_file_name, tpm_counts_file_name=tpm_counts_file_name)
        genome = loaded_input_files["genome"]
        annotation = loaded_input_files["annotation"]
        tpms = loaded_input_files["tpms"]
    
        extracted_genes = extract_genes(genome, annotation, extragenic=upstream, intragenic=downstream, model_case=model_case,ignore_small_genes=ignore_small_genes, train_val_split=train_val_split, tpms=tpms, target_chromosomes=())
        for val_chrom in chromosome_list:
            x, y, preds, gene_ids, model = predict_self(extragenic=upstream, intragenic=downstream, val_chromosome=val_chrom,
                                                output_name=output_name, model_case=model_case, extracted_genes=extracted_genes, train_val_split=train_val_split)
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
            print(f"Running shap for chromosome -----------------------------------------\n")
            print(f"Chromosome: {val_chrom}: Species: {output_name}\n")
            print(f"Running shap for chromosome -----------------------------------------\n")

            actual_scores, hypothetical_scores = compute_actual_hypothetical_scores(x=correct_x, model=model)
            shap_actual_scores.append(actual_scores)
            shap_hypothetical_scores.append(hypothetical_scores)
            one_hots_seqs.append(correct_x)
            gene_ids_seqs.extend(correct_gene_ids)
            preds_seqs.extend(correct_y)

    if model_case.lower() =="msr":
        genome = Fasta(filename=genome_file_name, as_raw=True, read_ahead=10000, sequence_always_upper=True)
        tpms = pd.read_csv(filepath_or_buffer=tpm_counts_file_name, sep=',')
        tpms.set_index('gene_id', inplace=True)
        annotation = load_annotation_msr(annotation_file_name)
        val_chrom=""

        extracted_genes = extract_genes(genome, annotation, extragenic=upstream, intragenic=downstream, model_case=model_case,ignore_small_genes=ignore_small_genes, train_val_split=train_val_split, tpms=tpms, target_chromosomes=())
        #for val_chrom in chromosome_list:
        x, y, preds, gene_ids, model = predict_self(extragenic=upstream, intragenic=downstream, val_chromosome=val_chrom,
                                                output_name=output_name, model_case=model_case, extracted_genes=extracted_genes, train_val_split=train_val_split)
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
        print(f"Running shap -----------------------------------------\n")
        print(f"Validation species: {output_name}\n")
        print(f"Running shap -----------------------------------------\n")

        actual_scores, hypothetical_scores = compute_actual_hypothetical_scores(x=correct_x, model=model)
        shap_actual_scores.append(actual_scores)
        shap_hypothetical_scores.append(hypothetical_scores)
        one_hots_seqs.append(correct_x)
        gene_ids_seqs.extend(correct_gene_ids)
        preds_seqs.extend(correct_y)

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
                        description="This script performs the deepCRE prediction. We assume you have the following three" + 
                        "directories:tmp_counts (contains your counts files), genome (contains the genome fasta files), gene_models (contains the gtf files)")

    parser.add_argument('--input', "-i", 
                        help="""For model case SSR/SSC: This is a six column csv file with entries: species, genome, gtf, tpm, output name, number of chromosomes and pickle_key. \n 
                        For model case MSR: This is a five column csv file with entries: species, genome, gtf, tpm, output name.""", required=True)
    parser.add_argument('--model_case', help="Can be SSC, SSR or MSR", required=True, choices=["msr", "ssr", "ssc", "both"])
    parser.add_argument('--ignore_small_genes', help="Ignore small genes, can be yes or no", required=False, choices=["yes", "no"], default="yes")
    parser.add_argument('--train_val_split', help="For SSR /SSC training: Creates a training/validation dataset with 80%/20% of genes, can be yes or no", required=False, choices=["yes", "no"], default="no")

    args = parser.parse_args()
    return args


def main():
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.disable_v2_behavior()
    tf.config.set_visible_devices([], 'GPU')

    intragenic = 500
    extragenic = 1000

    args = parse_args()
    model_case = args.model_case 

    dtypes = {0: str, 1: str, 2: str, 3: str, 4: str}
    names = ['specie','genome', 'gtf', 'tpm', 'output'] if model_case.lower() == "msr" else ['genome', 'gtf', 'tpm', 'output', 'chroms']
    data = pd.read_csv(args.input, sep=',', header=None, dtype=dtypes, names = names)
    expected_columns = len(names)

    print(data.head())
    
    ignore_small_genes_flag = args.ignore_small_genes.lower() == "yes"


    if model_case.lower() == "msr":
        naming = "_".join([specie[:3] for specie in data['specie'].unique()])
        input_filename = args.input.split('.')[0] 

        genome_path = make_absolute_path("genome", f"genome_{naming}.fa", start_file=__file__)     
        tpm_path = make_absolute_path("tpm_counts", f"tpm_{naming}_{input_filename}.csv", start_file=__file__)  # tpm_targets = f"tpm_{p_keys}.csv"
        annotation_path = make_absolute_path("gene_models", f"gtf_{naming}.csv", start_file=__file__)  
        #annotation = load_annotation_msr(annotation_path)


        for specie in data['specie'].unique():                                                                     
            test_specie = data.copy()
            test_specie = test_specie[test_specie['specie'] == specie]
            train_specie = data.copy()
            train_specie = train_specie[train_specie['specie'] != specie]

            output_name = test_specie['output'].values[0]
            chromosomes = ""

            results = extract_scores(genome_file_name=genome_path, annotation_file_name=annotation_path, tpm_counts_file_name=tpm_path, upstream=1000, downstream=500,
                        chromosome_list=chromosomes, ignore_small_genes=ignore_small_genes_flag,
                        output_name=output_name, model_case=args.model_case, train_val_split=args.train_val_split)
            shap_actual_scores, shap_hypothetical_scores, one_hots_seqs, gene_ids_seqs, pred_seqs = results
            save_results(shap_actual_scores=shap_actual_scores, shap_hypothetical_scores=shap_hypothetical_scores,
                        output_name=output_name, gene_ids_seqs=gene_ids_seqs, preds_seqs=pred_seqs)


    
    if model_case.lower() in ["ssr", "ssc"]:
    
        for genome, gtf, tpm_counts, output_name, chromosomes_file in data.values:
            chromosomes = pd.read_csv(filepath_or_buffer=f'genome/{chromosomes_file}', header=None).values.ravel().tolist()
            results = extract_scores(genome_file_name=genome, annotation_file_name=gtf, tpm_counts_file_name=tpm_counts, upstream=1000, downstream=500,
                        chromosome_list=chromosomes, ignore_small_genes=ignore_small_genes_flag,
                        output_name=output_name, model_case=args.model_case, train_val_split=args.train_val_split)
            shap_actual_scores, shap_hypothetical_scores, one_hots_seqs, gene_ids_seqs, pred_seqs = results
            save_results(shap_actual_scores=shap_actual_scores, shap_hypothetical_scores=shap_hypothetical_scores,
                        output_name=output_name, gene_ids_seqs=gene_ids_seqs, preds_seqs=pred_seqs)
            
    print("Restults saved in results/shap.")

if __name__ == "__main__":
    main()