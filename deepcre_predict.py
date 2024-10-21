import argparse
import os
from typing import Any, Dict, List, Tuple
from tensorflow.keras.models import load_model #type:ignore
import pandas as pd

from utils import get_filename_from_path, get_time_stamp, load_input_files, one_hot_encode, make_absolute_path, result_summary
from train_ssr_models import extract_genes, find_newest_model_path


def predict_self(extragenic, intragenic, val_chromosome, output_name, model_case, extracted_genes, train_val_split):

    x, y, gene_ids = extracted_genes[str(val_chromosome)]

    # Masking
    x[:, extragenic:extragenic + 3, :] = 0                                                                                                  #type:ignore
    x[:, extragenic + (intragenic * 2) + 17:extragenic + (intragenic * 2) + 20, :] = 0                                                      #type:ignore

    if train_val_split.lower() == 'yes':
        val_chromosome = "1|2|3" 

    newest_model_paths = find_newest_model_path(output_name=output_name, val_chromosome=val_chromosome, model_case=model_case)
    model = load_model(newest_model_paths[val_chromosome])
    pred_probs = model.predict(x).ravel()
    return x, y, pred_probs, gene_ids, model


def parse_args():
    parser = argparse.ArgumentParser(
                        prog='deepCRE',
                        description="This script performs the deepCRE prediction. We assume you have the following three" + 
                        "directories:tmp_counts (contains your counts files), genome (contains the genome fasta files), gene_models (contains the gtf files)")

    parser.add_argument('--input', "-i", 
                        help="This is a 5 column csv file with entries: genome, gtf, tpm, output name, number of chromosomes.",
                        required=True)
    parser.add_argument('--model_case', help="Can be SSC or SSR", required=True)
    parser.add_argument('--ignore_small_genes', help="Ignore small genes, can be yes or no", required=True)
    parser.add_argument('--train_val_split', help="Creates a training/validation dataset with 80%/20% of genes, can be yes or no", required=True)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    data = pd.read_csv(args.input, sep=',', header=None,
                    dtype={0: str, 1: str, 2: str, 3: str, 4: str},
                    names=['genome', 'gtf', 'tpm', 'output', 'chroms'])
    print(data.head())
    if data.shape[1] != 5:
        raise Exception("Input file incorrect. Your input file must contain 5 columns and must be .csv")

    folder_name = make_absolute_path('results', 'predictions', start_file=__file__)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    file_name = get_filename_from_path(__file__)

    failed_trainings = []
    for i, (genome_file_name, annotation_file_name, tpm_counts_file_name, output_name, chromosome_file) in enumerate(data.values):
        try:
            true_targets, preds, genes = [], [], []
            loaded_input_files = load_input_files(genome_file_name=genome_file_name, annotation_file_name=annotation_file_name, tpm_counts_file_name=tpm_counts_file_name)
            genome = loaded_input_files["genome"]
            annotation = loaded_input_files["annotation"]
            tpms = loaded_input_files["tpms"]
            extragenic = 1000
            intragenic = 500
            ignore_small_genes = args.ignore_small_genes.lower() == "yes"
            train_val_split=args.train_val_split
            chromosomes = pd.read_csv(filepath_or_buffer=f'genome/{chromosome_file}', header=None).values.ravel().tolist()
            extracted_genes = extract_genes(genome=genome, annotation=annotation, extragenic=extragenic, intragenic=intragenic, ignore_small_genes=ignore_small_genes, tpms=tpms, target_chromosomes=(), train_val_split=train_val_split)
            for chrom in chromosomes:
                _, y, pred_probs, gene_ids, _ = predict_self(extragenic=extragenic, intragenic=intragenic, val_chromosome=str(chrom), output_name=output_name,
                                                        model_case=args.model_case, extracted_genes=extracted_genes)
                true_targets.extend(y)
                preds.extend(pred_probs)
                genes.extend(gene_ids)

            result = pd.DataFrame({'true_targets': true_targets, 'pred_probs': preds, 'genes': genes})
            print(result.head())
            output_location = os.path.join(folder_name, f'{output_name}_{file_name}_{get_time_stamp()}.csv')
            result.to_csv(output_location, index=False)
        except Exception as e:
            print(e)
            failed_trainings.append((output_name, i, e))

    result_summary(failed_trainings=failed_trainings, input_length=len(data), script=get_filename_from_path(__file__))


if __name__ == "__main__":
    main()