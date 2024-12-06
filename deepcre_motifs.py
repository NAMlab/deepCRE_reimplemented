import argparse
import pandas as pd
import tensorflow as tf
import os
import modisco
from importlib import reload
import h5py
from utils import get_filename_from_path, get_time_stamp, make_absolute_path, load_annotation_msr, result_summary
from deepcre_interpret import extract_scores, find_newest_interpretation_results


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
                    output_name, model_case, chromosome_list: pd.DataFrame, train_val_split, force_interpretation: bool = False):
    try:
        if force_interpretation:
            raise ValueError()
        saved_interpretation_results_path = find_newest_interpretation_results(output_name=output_name, results_path=os.path.join("results", "shap"))
        with h5py.File(saved_interpretation_results_path, "r") as f:
            actual_scores = f["contrib_scores"][:] #type:ignore
            hypothetical_scores = f["hypothetical_contrib_scores"][:] #type:ignore
            one_hots = f["one_hot_seqs"][:] #type:ignore
    except ValueError:
        actual_scores, hypothetical_scores, one_hots, _, _ = extract_scores(genome_file_name=genome, annotation_file_name=annot,
                                                                            tpm_counts_file_name=tpm_targets,
                                                                            upstream=upstream, downstream=downstream,
                                                                            chromosome_list=chromosome_list,
                                                                            ignore_small_genes=ignore_small_genes,
                                                                            output_name=output_name,
                                                                            model_case=model_case, train_val_split=train_val_split)

    print("Now running MoDisco --------------------------------------------------\n")
    print(f"Species: {output_name} \n")
    modisco_run(contribution_scores=actual_scores, hypothetical_scores=hypothetical_scores,
                one_hots=one_hots, output_name=output_name)


def parse_args():
    parser = argparse.ArgumentParser(
                        prog='deepCRE',
                        description="This script performs the deepCRE prediction. We assume you have the following three" + 
                        "directories:tmp_counts (contains your counts files), genome (contains the genome fasta files), gene_models (contains the gtf files)")

    parser.add_argument('--input', "-i", "-i", 
                        help="""For model case SSR/SSC: This is a six column csv file with entries: species, genome, gtf, tpm, output name, number of chromosomes and pickle_key. \n 
                        For model case MSR: This is a five column csv file with entries: species, genome, gtf, tpm, output name.""", required=True)
    parser.add_argument('--model_case', "-mc", help="Can be SSC, SSR or MSR", required=True, choices=["msr", "ssr", "ssc", "both"])
    parser.add_argument('--ignore_small_genes', "-isg", help="Ignore small genes, can be yes or no", required=False, choices=["yes", "no"], default="yes")
    parser.add_argument('--force_interpretations', "-fi", help="determines whether interpretations are recalculated, even if there are saved interpretations. If false (default), saved values will be used.", required=False, default="false", choices=["true", "false"])
    parser.add_argument('--train_val_split', help="For SSR /SSC training: Creates a training/validation dataset with 80%/20% of genes, can be yes or no", required=False, choices=["yes", "no"], default="no")

    args = parser.parse_args()
    return args


def main():
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.disable_v2_behavior()
    tf.config.set_visible_devices([], 'GPU')
    args = parse_args()
    model_case = args.model_case 


    dtypes = {0: str, 1: str, 2: str, 3: str, 4: str}
    names = ['specie','genome', 'gtf', 'tpm', 'output'] if model_case.lower() == "msr" else ['genome', 'gtf', 'tpm', 'output', 'chroms']
    data = pd.read_csv(args.input, sep=',', header=None, dtype=dtypes, names = names)
    expected_columns = len(names)

    print(data.head())
    ignore_small_genes_flag = args.ignore_small_genes.lower() == "yes"
    
    if model_case.lower() == "msr":
        p_keys = "_".join(data['p_key'].unique())
        input_filename = args.input.split('.')[0] 

        genome_path = make_absolute_path("genome", f"genome_{p_keys}.fa", start_file=__file__)     
        tpm_path = make_absolute_path("tpm_counts", f"tpm_{p_keys}_{input_filename}.csv", start_file=__file__)  # tpm_targets = f"tpm_{p_keys}.csv"
        annotation_path = make_absolute_path("gene_models", f"gtf_{p_keys}.csv", start_file=__file__)  
        annotation = load_annotation_msr(annotation_path)


        for specie in data['specie'].unique():                                                                     
            test_specie = data.copy()
            test_specie = test_specie[test_specie['specie'] == specie]
            train_specie = data.copy()
            train_specie = train_specie[train_specie['specie'] != specie]

            output_name = "_".join([sp[:3].lower() for sp in train_specie['specie'].unique()])
            
            test_specie_name = test_specie['specie'].values[0]
            #chromosomes = annotation[annotation['species'] == test_specie_name]['Chromosome'].unique().tolist()
            #chromosomes = sorted(chromosomes, key=lambda x: int("".join(filter(str.isdigit, x)))) 
            chromosomes=""

            generate_motifs(genome=genome_path, annot=annotation_path, tpm_targets=tpm_path, upstream=1000, downstream=500,
                            ignore_small_genes=ignore_small_genes_flag, output_name=output_name,
                            model_case=args.model_case, chromosome_list=chromosomes, train_val_split=args.train_val_split)

    elif model_case.lower() in ["ssr", "ssc"]:
        failed_trainings = []
        for i, (genome, gtf, tpm_counts, output_name, chromosomes_file) in enumerate(data.values):
            try:
                chromosomes = pd.read_csv(filepath_or_buffer=f'genome/{chromosomes_file}', header=None).values.ravel().tolist()
                generate_motifs(genome=genome, annot=gtf, tpm_targets=tpm_counts, upstream=1000, downstream=500,
                                ignore_small_genes=ignore_small_genes_flag, output_name=output_name,
                                model_case=args.model_case, chromosome_list=chromosomes, train_val_split=args.train_val_split, force_interpretation=force_interpretation)
            except Exception as e:
                print(e)
                failed_trainings.append((output_name, i, e))

        result_summary(failed_trainings=failed_trainings, input_length=len(data), script=get_filename_from_path(__file__))


if __name__ == "__main__":
    main()
