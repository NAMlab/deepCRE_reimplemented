import argparse
import pandas as pd
import tensorflow as tf
import os
import modisco
from importlib import reload
import h5py
from utils import get_filename_from_path, get_time_stamp, make_absolute_path
from deepcre_interpret import extract_scores


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
                    output_name, model_case, chromosome_list: pd.DataFrame, train_val_split):

    actual_scores, hypothetical_scores, one_hots, _, _ = extract_scores(genome_file_name=genome, annotation_file_name=annot,
                                                                        tpm_counts_file_name=tpm_targets,
                                                                        upstream=upstream, downstream=downstream,
                                                                        chromosome_list=chromosome_list,
                                                                        ignore_small_genes=ignore_small_genes,
                                                                        output_name=output_name,
                                                                        model_case=model_case, 
                                                                        train_val_split=train_val_split)

    print("Now running MoDisco --------------------------------------------------\n")
    print(f"Species: {output_name} \n")
    modisco_run(contribution_scores=actual_scores, hypothetical_scores=hypothetical_scores,
                one_hots=one_hots, output_name=output_name)


def parse_args():
    parser = argparse.ArgumentParser(
                        prog='deepCRE',
                        description="""
                        This script performs the deepCRE prediction. We assume you have the following three directories:
                        tmp_counts (contains your counts files), genome (contains the genome fasta files),
                        gene_models (contains the gtf files)
                        """)

    parser.add_argument('--input',
                        help="""
                        This is a 5 column csv file with entries: genome, gtf, tpm, output name, number of chromosomes.""",
                        required=True)
    parser.add_argument('--model_case', help="Can be SSC, SSR or MSR", required=True)
    parser.add_argument('--ignore_small_genes', help="Ignore small genes, can be yes or no", required=True)
    parser.add_argument('--train_val_split', help="Creates a training/validation dataset with 80%/20% of genes, can be yes or no", required=True)

    args = parser.parse_args()
    return args


def main():
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.disable_v2_behavior()
    tf.config.set_visible_devices([], 'GPU')
    args = parse_args()
    data = pd.read_csv(args.input, sep=',', header=None,
                    dtype={0: str, 1: str, 2: str, 3: str, 4: str},
                    names=['genome', 'gtf', 'tpm', 'output', 'chroms'])
    ignore_small_genes = args.ignore_small_genes.lower() == "yes"
    print(data.head())
    if data.shape[1] != 5:
        raise Exception("Input file incorrect. Your input file must contain 5 columns and must be .csv")


    for genome, gtf, tpm_counts, output_name, chromosomes_file in data.values:
        chromosomes = pd.read_csv(filepath_or_buffer=f'genome/{chromosomes_file}', header=None).values.ravel().tolist()
        generate_motifs(genome=genome, annot=gtf, tpm_targets=tpm_counts, upstream=1000, downstream=500,
                        ignore_small_genes=ignore_small_genes, output_name=output_name,
                        model_case=args.model_case, chromosome_list=chromosomes, train_val_split=args.train_val_split)


if __name__ == "__main__":
    main()