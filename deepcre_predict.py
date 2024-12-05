import argparse
import os
from typing import Any, Dict, List, Tuple
import numpy as np
from tensorflow.keras.models import load_model #type:ignore
import pandas as pd
import re
from pyfaidx import Fasta 

from utils import get_filename_from_path, get_time_stamp, load_input_files, one_hot_encode, make_absolute_path, load_annotation_msr
from train_models import extract_genes


def find_newest_model_path(output_name: str, model_case: str, val_chromosome: str = "", model_path: str = "") -> Dict[str, str]:
    """finds path to newest model fitting the given parameters

    Args:
        output_name (str): output name the was used for model training
        val_chromosome (str): validation chromosome of the model. If it is not given, all models regardless of the val_chromosome will be returned
        model_case (str): SSR, SSC or MSR for the model to be loaded
        model_path (str): path to the directory where models are stored. used for testing, probably not really stable

    Raises:
        ValueError: raises an error if no fitting model is found

    Returns:
        List[str]: List of path to the newest model fitting the given parameters for a single chromosome, or all fitting models if chromosome is ommitted.
    """
    if model_path == "":
        path_to_models = make_absolute_path("saved_models", start_file=__file__)
    else:
        path_to_models = make_absolute_path(model_path, start_file=__file__)
    # ^ and $ mark start and end of a string. \d singnifies any digit. \d+ means a sequence of digits with at least length 1
    # more detailed explanation at https://regex101.com/, put in "^ara_(\d+)_ssr_\d+_\d+\.h5$"
    
    if val_chromosome == "":
        regex_string = f"^{output_name}_(.+)_{model_case}_train_ssr_models_\d+_\d+\.h5$"                                                                    #type:ignore
    else:
        regex_string = f"^{output_name}_{val_chromosome}_{model_case}_train_ssr_models_\d+_\d+\.h5$"                                                        #type:ignore
        
    if model_case.lower() == "msr": 
        regex_string = f"^{output_name}_model_{model_case}_train_ssr_models_\d+_\d+\.h5$"     # specific for now
      

    regex = re.compile(regex_string)
    #print(regex)
    candidate_models = [model for model in os.listdir(path_to_models)]
    fitting_models = {}
    for candidate in candidate_models:
        match = regex.match(candidate)
        if match:
            print(f"Match found: {candidate}")

        if match and model_case.lower() in ["ssr", "ssc"]:
            # group 1 is the "(.+)" part of the regex, so the name of the validation chromosome for the model
            chromosome = val_chromosome if val_chromosome else match.group(1)
            if chromosome in fitting_models:
                fitting_models[chromosome].append(candidate)
            else:
                fitting_models[chromosome] = [candidate]

            if not fitting_models:
                raise ValueError(f"no trained models fitting the given parameters (output_name: '{output_name}', val_chromosome: '{val_chromosome}', model_case: '{model_case}') were found! Consider training models first (train_models.py)")
            for chromosome, models in fitting_models.items():
                # models per chromosome only differ in the time stamp. So if sorted, the last model will be the most recently trained
                models.sort()
                fitting_models[chromosome] = os.path.join(path_to_models, models[-1])

        if match and model_case.lower() == "msr":
            #chromosome = "all"
            #fitting_models["all"] = os.path.join(path_to_models, candidate)
            return os.path.join(path_to_models, candidate)

    return fitting_models


def predict_self(extragenic, intragenic, val_chromosome, output_name, model_case, extracted_genes, train_val_split):

    if train_val_split.lower() == 'yes':
        val_chromosome = "1|2|3" 
        
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

        newest_model_paths = find_newest_model_path(output_name=output_name, val_chromosome=None, model_case=model_case)
        model = load_model(newest_model_paths)
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
    names = ['specie','genome', 'gtf', 'tpm', 'output'] if model_case.lower() == "msr" else ['genome', 'gtf', 'tpm', 'output', 'chroms']
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
        for genome_file_name, annotation_file_name, tpm_counts_file_name, output_name, chromosome_file in data.values:
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
            
            extracted_genes = extract_genes(genome=genome, annotation=annotation, extragenic=extragenic, intragenic=intragenic, ignore_small_genes=ignore_small_genes, train_val_split=train_val_split, tpms=tpms, target_chromosomes=tuple(chromosomes), model_case=args.model_case.lower())

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