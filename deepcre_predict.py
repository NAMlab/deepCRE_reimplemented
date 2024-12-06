import argparse
import os
from typing import Any, Dict, List, Tuple
from tensorflow.keras.models import load_model #type:ignore
import pandas as pd
from utils import get_filename_from_path, get_time_stamp, load_annotation_msr, load_input_files, one_hot_encode, make_absolute_path, result_summary
from train_ssr_models import extract_genes, find_newest_model_path
import numpy as np
from pyfaidx import Fasta


def predict_self(extragenic, intragenic, val_chromosome, output_name, model_case, extracted_genes):

    if model_case.lower() == "msr":
        # Combine data from all chromosomes
        x,y,gene_ids = [], [],[]
        for chrom, tuple_ in extracted_genes.items():
            if tuple_:  
                x_chrom, y_chrom, gene_ids_chrom = tuple_
                x.extend(x_chrom)  
                y.extend(y_chrom)
                gene_ids.extend(gene_ids_chrom)
        # Convert lists to arrays
        x = np.array(x)
        y = np.array(y)
        gene_ids = np.array(gene_ids)

        newest_model_paths = find_newest_model_path(output_name=output_name, model_case=model_case)
        model = load_model(newest_model_paths["model"])
        #print(f"Trying to load model from: {newest_model_paths}")

    else:
        # Handle specific chromosome
        x, y, gene_ids = extracted_genes[str(val_chromosome)]
        newest_model_paths = find_newest_model_path(output_name=output_name, val_chromosome=val_chromosome, model_case=model_case)
        model = load_model(newest_model_paths[val_chromosome])

    # Masking
    x[:, extragenic:extragenic + 3, :] = 0                                                                                                  #type:ignore
    x[:, extragenic + (intragenic * 2) + 17:extragenic + (intragenic * 2) + 20, :] = 0                                                      #type:ignore

    
    pred_probs = model.predict(x).ravel()
    return x, y, pred_probs, gene_ids, model


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
    args = parse_args()
    model_case = args.model_case 

    dtypes = {0: str, 1: str, 2: str, 3: str, 4: str, 5: str, 6: str} if model_case.lower() == "msr" else {0: str, 1: str, 2: str, 3: str, 4: str, 5: str}
    names = ['specie','genome', 'gtf', 'tpm', 'output', "chroms", "p_keys"] if model_case.lower() == "msr" else ['genome', 'gtf', 'tpm', 'output', 'chroms']
    data = pd.read_csv(args.input, sep=',', header=None, dtype=dtypes, names = names)
    expected_columns = len(names)

    print(data.head())
    if data.shape[1] != expected_columns:
        raise Exception("Input file incorrect. Your input file must contain 7 columns and must be .csv")
    

    folder_name = make_absolute_path('results', 'predictions', start_file=__file__)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    file_name = get_filename_from_path(__file__)

       
    
    if model_case.lower() == "msr":
        naming = "_".join([specie[:3] for specie in data['specie'].unique()])
        input_filename = args.input.split('.')[0] 

        genome_path = make_absolute_path("genome", f"genome_{naming}.fa", start_file=__file__)     
        tpm_path = make_absolute_path("tpm_counts", f"tpm_{naming}_{input_filename}.csv", start_file=__file__) 
        annotation_path = make_absolute_path("gene_models", f"gtf_{naming}.csv", start_file=__file__)  
        
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

            output_name = test_specie['output'].values[0]
            chrom = ""

            true_targets, preds, genes = [], [], []
            
            extracted_genes = extract_genes(genome=genome, annotation=annotation, extragenic=extragenic, intragenic=intragenic, model_case=args.model_case,ignore_small_genes=ignore_small_genes, train_val_split=train_val_split, tpms=tpms, target_chromosomes=())

            # one predcition per model 
            print(f"Predicting for: {output_name}")
            _, y, pred_probs, gene_ids, _ = predict_self(extragenic=extragenic, intragenic=intragenic, val_chromosome=str(chrom), output_name=output_name,
                                                    model_case=args.model_case, extracted_genes=extracted_genes)
            true_targets.extend(y)
            preds.extend(pred_probs)
            genes.extend(gene_ids)

            result = pd.DataFrame({'true_targets': true_targets, 'pred_probs': preds, 'genes': genes})
            print(result.head())
            output_location = os.path.join(folder_name, f'{output_name}_MSR_{file_name}_{get_time_stamp()}.csv')
            result.to_csv(output_location, index=False)

    elif model_case.lower() in ["ssr", "ssc"]:
    # og ssr case 
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
                
                extracted_genes = extract_genes(genome=genome, annotation=annotation, extragenic=extragenic, intragenic=intragenic, ignore_small_genes=ignore_small_genes, train_val_split=train_val_split, tpms=tpms, target_chromosomes=(), model_case=args.model_case.lower())

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
