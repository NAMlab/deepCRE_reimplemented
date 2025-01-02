import argparse
import os
from typing import Any, Dict, List, Optional, Tuple
from tensorflow.keras.models import load_model #type:ignore
import pandas as pd
from utils import get_filename_from_path, get_time_stamp, load_annotation_msr, load_input_files, one_hot_encode, make_absolute_path, result_summary
from train_models import extract_genes_prediction, find_newest_model_path
import numpy as np
from pyfaidx import Fasta
from parsing import ModelCase, ParsedTrainingInputs, RunInfo


def predict_self(extragenic, intragenic, val_chromosome, output_name, model_case: ModelCase, extracted_genes):

    if model_case == ModelCase.MSR:
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


def run_ssr(folder_name: str, file_name: str, general_info: Dict, specie_info: Dict, genome: Fasta, annotation: pd.DataFrame, tpms: Optional[pd.DataFrame], extragenic: int, intragenic: int, output_name: str):
    true_targets, preds, genes = [], [], []
    extracted_genes = extract_genes_prediction(genome=genome, annotation=annotation, extragenic=extragenic, intragenic=intragenic,
                                                           ignore_small_genes=general_info["ignore_small_genes"], tpms=tpms, target_chromosomes=(), model_case=general_info["model_case"])

    for chrom in specie_info["chromosomes"]:
        _, y, pred_probs, gene_ids, _ = predict_self(extragenic=extragenic, intragenic=intragenic, val_chromosome=str(chrom), output_name=output_name,
                                                            model_case=general_info["model_case"], extracted_genes=extracted_genes)
        true_targets.extend(y)
        preds.extend(pred_probs)
        genes.extend(gene_ids)

    result = pd.DataFrame({'true_targets': true_targets, 'pred_probs': preds, 'genes': genes})
    print(result.head())
    output_location = os.path.join(folder_name, f'{output_name}_{file_name}_{get_time_stamp()}.csv')
    result.to_csv(output_location, index=False)


def run_msr(folder_name: str, file_name: str, general_info: Dict, genome: Fasta, annotation: pd.DataFrame, tpms: Optional[pd.DataFrame], extragenic: int, intragenic: int, species_name: str):
    extracted_genes = extract_genes_prediction(genome=genome, annotation=annotation, extragenic=extragenic, intragenic=intragenic,
                                                            model_case=general_info["model_case"],ignore_small_genes=general_info["ignore_small_genes"], tpms=tpms, target_chromosomes=())
                    # one predcition per model
    print(f"Predicting for: {species_name}")
    _, true_targets, preds, genes, _ = predict_self(extragenic=extragenic, intragenic=intragenic, val_chromosome="", output_name=species_name,
                                                            model_case=general_info["model_case"], extracted_genes=extracted_genes)
    result = pd.DataFrame({'true_targets': true_targets, 'pred_probs': preds, 'genes': genes})
    print(result.head())
    output_location = os.path.join(folder_name, f'{species_name}_MSR_{file_name}_{get_time_stamp()}.csv')
    result.to_csv(output_location, index=False)


def check_inputs(run_info: RunInfo):
    gen_info = run_info.general_info
    spec_info = run_info.species_info
    if run_info.is_msr():
        for specie_data in spec_info:
            if specie_data["chromosomes"] != "":
                print(f"WARNING: chromosome information for MSR runs is not used!")
            if specie_data["species_name"] == "":
                raise ValueError(f"name of species needs to be provided!")
    else:
        for specie_data in spec_info:
            if specie_data["chromosomes"] == "":
                raise ValueError(f"chromosome information needs to be provided for SSR runs!")
        if gen_info["output_name"] == "":
            raise ValueError(f"Output name needs to be provided for SSR / SSC runs!")


def predict(inputs: ParsedTrainingInputs, failed_trainings: List[Tuple], input_length: int):
    folder_name = make_absolute_path('results', 'predictions', start_file=__file__)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    file_name = get_filename_from_path(__file__)
    run_info: RunInfo
    for i, run_info in enumerate(inputs):       #type:ignore
        species_info = run_info.species_info
        general_info = run_info.general_info
        loaded_files = load_input_files(genome_file_name=general_info["genome"], annotation_file_name=general_info["annotation"], tpm_counts_file_name=general_info["targets"], model_case=str(general_info["model_case"]))
        genome = loaded_files["genome"]
        annotation = loaded_files["annotation"]
        tpms = loaded_files["tpms"] if "tpms" in loaded_files.keys() else None
        extragenic = general_info["extragenic"]
        intragenic = general_info["intragenic"]

        try:
            check_inputs(run_info)
            if run_info.is_msr(): 
                #for specie, genome_file_name, annotation_file_name, tpm_counts_file_name, output_name, chromosome_file,_  in data.values:
                for specie_info in species_info:                                                                     # use this 
                    output_name = specie_info["species_name"]
                    run_msr(folder_name=folder_name, file_name=file_name, general_info=general_info, genome=genome,
                            annotation=annotation, tpms=tpms, extragenic=extragenic, intragenic=intragenic, species_name=output_name)

            else:
                output_name = general_info["output_name"]
                run_ssr(folder_name=folder_name, file_name=file_name, general_info=general_info, specie_info=specie_info,
                        genome=genome, annotation=annotation, tpms=tpms, extragenic=extragenic, intragenic=intragenic,
                        output_name=output_name)
        except Exception as e:
            print(e)
            failed_trainings.append((output_name, i, e))
    result_summary(failed_trainings=failed_trainings, input_length=input_length, script=get_filename_from_path(__file__))


def parse_args():
    parser = argparse.ArgumentParser(
        prog='deepCRE',
        description="This script performs the deepCRE prediction. We assume you have the following three" + 
        "directories:tmp_counts (contains your counts files), genome (contains the genome fasta files), gene_models (contains the gtf files)"
    )

    parser.add_argument(
        '--input', "-i", 
        help="""json file containing the required input parameters. Possible arguments can be seen in the file parsing.py in the two global dictionaries.
        Example file is inputs.json.""", required=True
    )
    args = parser.parse_args()
    return args


def main():
    possible_general_parameters = {
        "genome": None,
        "annotation": None,
        "targets": "",
        "output_name": "",
        "chromosomes": "",
        "model_case": None,
        "ignore_small_genes": True,
        "extragenic": 1000,
        "intragenic": 500
    }

    possible_species_parameters = {
        "chromosomes": "",
        "species_name": ""
    }
    args = parse_args()
    inputs, failed_trainings, input_length = ParsedTrainingInputs.parse(args.input, possible_general_parameters=possible_general_parameters, possible_species_parameters=possible_species_parameters)
    inputs = inputs.replace_both()
    print(inputs)
    predict(inputs, failed_trainings=failed_trainings, input_length=input_length)


if __name__ == "__main__":
    main()
