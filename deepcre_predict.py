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
    #print(f"model path: {newest_model_paths}")
    model = load_model(newest_model_paths[val_chromosome])
    pred_probs = model.predict(x).ravel()
    return x, y, pred_probs, gene_ids, model


def parse_args():
    parser = argparse.ArgumentParser(
                        prog='deepCRE',
                        description="This script performs the deepCRE prediction. We assume you have the following three" + 
                        "directories:tmp_counts (contains your counts files), genome (contains the genome fasta files), gene_models (contains the gtf files)")

    parser.add_argument('--input', "-i", 
                        help="This is a 7 column csv file with entries: specie, genome, gtf, tpm, output name, number of chromosomes and pickle_key.",
                        required=True)
    parser.add_argument('--model_case', help="Can be SSC, SSR or MSR", required=True)
    parser.add_argument('--ignore_small_genes', help="Ignore small genes, can be yes or no", required=True)
    parser.add_argument('--train_val_split', help="Creates a training/validation dataset with 80%/20% of genes, can be yes or no", required=True)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    model_case = args.model_case 

    data = pd.read_csv(args.input, sep=',', header=None,
                    dtype={0: str, 1: str, 2: str, 3: str, 4: str},
                    names=["specie",'genome', 'gtf', 'tpm', 'output', 'chroms', "p_key"])
    print(data.head())
    if data.shape[1] != 7:
        raise Exception("Input file incorrect. Your input file must contain 7 columns and must be .csv")

    folder_name = make_absolute_path('results', 'predictions', start_file=__file__)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    file_name = get_filename_from_path(__file__)

       
    
    if model_case.lower() == "msr":
        p_keys = "_".join(data['p_key'].unique())
        input_filename = args.input.split('.')[0] 

        genome_path = make_absolute_path("genome", f"genome_{p_keys}.fa", start_file=__file__)     
        tpm_path = make_absolute_path("tpm_counts", f"tpm_{p_keys}_{input_filename}.csv", start_file=__file__)  # tpm_targets = f"tpm_{p_keys}.csv"
        annotation_path = make_absolute_path("gene_models", f"gtf_{p_keys}.csv", start_file=__file__)  

        genome = Fasta(filename=genome_path, as_raw=True, read_ahead=10000, sequence_always_upper=True)
        tpms = pd.read_csv(filepath_or_buffer=tpm_path, sep=',')
        tpms.set_index('gene_id', inplace=True)
        annotation = load_annotation_msr(annotation_path)
        extragenic = 1000
        intragenic = 500
        ignore_small_genes = args.ignore_small_genes.lower() == "yes"
        train_val_split=args.train_val_split

        #for specie, genome_file_name, annotation_file_name, tpm_counts_file_name, output_name, chromosome_file,_  in data.values:
        for specie in data['specie'].unique():                                                                     # use this 
            test_specie = data.copy()
            test_specie = test_specie[test_specie['specie'] == specie]
            train_specie = data.copy()
            train_specie = train_specie[train_specie['specie'] != specie]

            output_name = "_".join([sp[:3].lower() for sp in train_specie['specie'].unique()])

            true_targets, preds, genes = [], [], []
            
            test_specie_name = test_specie['specie'].values[0]
            chromosomes = annotation[annotation['species'] == test_specie_name]['Chromosome'].unique().tolist()
            chromosomes = sorted(chromosomes, key=lambda x: int("".join(filter(str.isdigit, x))))

            extracted_genes = extract_genes(genome=genome, annotation=annotation, extragenic=extragenic, intragenic=intragenic, model_case=args.model_case,ignore_small_genes=ignore_small_genes, train_val_split=train_val_split, tpms=tpms, target_chromosomes=())

            for chrom in chromosomes:
                _, y, pred_probs, gene_ids, _ = predict_self(extragenic=extragenic, intragenic=intragenic, val_chromosome=str(chrom), output_name=output_name,
                                                        model_case=args.model_case, extracted_genes=extracted_genes, train_val_split=train_val_split)
                true_targets.extend(y)
                preds.extend(pred_probs)
                genes.extend(gene_ids)

            result = pd.DataFrame({'true_targets': true_targets, 'pred_probs': preds, 'genes': genes})
            print(result.head())
            output_location = os.path.join(folder_name, f'{output_name}_MSR_{file_name}_{get_time_stamp()}.csv')
            result.to_csv(output_location, index=False)

    
    if model_case.lower() in ["ssr", "ssc"]:
    # og ssr case 
        for _,genome_file_name, annotation_file_name, tpm_counts_file_name, output_name, chromosome_file in data.values:
            true_targets, preds, genes = [], [], []
            loaded_input_files = load_input_files(genome_file_name=genome_file_name, annotation_file_name=annotation_file_name, tpm_counts_file_name=tpm_counts_file_name)
            genome = loaded_input_files["genome"]
            annotation = load_annotation_msr["annotation"]
            tpms = loaded_input_files["tpms"]
            extragenic = 1000
            intragenic = 500
            ignore_small_genes = args.ignore_small_genes.lower() == "yes"
            train_val_split=args.train_val_split
            chromosomes = pd.read_csv(filepath_or_buffer=f'genome/{chromosome_file}', header=None).values.ravel().tolist()
            
            extracted_genes = extract_genes(genome=genome, annotation=annotation, extragenic=extragenic, intragenic=intragenic, ignore_small_genes=ignore_small_genes, train_val_split=train_val_split, tpms=tpms, target_chromosomes=())

            for chrom in chromosomes:
                _, y, pred_probs, gene_ids, _ = predict_self(extragenic=extragenic, intragenic=intragenic, val_chromosome=str(chrom), output_name=output_name,
                                                        model_case=args.model_case, extracted_genes=extracted_genes, train_val_split=train_val_split)
                true_targets.extend(y)
                preds.extend(pred_probs)
                genes.extend(gene_ids)

            result = pd.DataFrame({'true_targets': true_targets, 'pred_probs': preds, 'genes': genes})
            print(result.head())
            output_location = os.path.join(folder_name, f'{output_name}_{file_name}_{get_time_stamp()}.csv')
            result.to_csv(output_location, index=False)


if __name__ == "__main__":
    main()