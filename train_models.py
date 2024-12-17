from __future__ import annotations
import argparse
import json
from enum import Enum, auto
import os
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from utils import get_filename_from_path, get_time_stamp, one_hot_encode, make_absolute_path, load_annotation, load_annotation_msr, result_summary, combine_annotations, combine_fasta, combine_tpms, load_input_files, read_feature_from_input_dict
from tensorflow.keras.layers import Dropout, Dense, Input, Conv1D, Activation, MaxPool1D, Flatten               #type:ignore
from tensorflow.keras.optimizers import Adam                                                                    #type:ignore
from tensorflow.keras import Model                                                                              #type:ignore
from tensorflow.keras.models import load_model                                                                  #type:ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau                        #type:ignore
from tensorflow.keras.metrics import AUC                                                                        #type:ignore
import pickle
import numpy as np
from pyfaidx import Fasta
import pyranges as pr
from sklearn.utils import shuffle
import re
import sys

possible_general_parameters = {
    "genome": None,
    "annotation": None,
    "targets": None,
    "output_name": None,
    "chromosomes": None,
    "pickle_key": None,
    "pickle_file": "validation_genes.pickle",
    "model_case": None,
    "ignore_small_genes": True,
    "train_val_split": False,
}

possible_species_parameters = {
    "genome": None,
    "annotation": None,
    "targets": None,
    "chromosomes": None,
    "pickle_key": None,
    "pickle_file": "validation_genes.pickle",
    "species_name": None,
}


class RunInfo:
    general_info: Dict[str, Any]
    species_info: List[Dict[str, Any]]

    def set_up_defaults(self):
        #get general defaults
        defaults = possible_species_parameters.copy()
        #get values from general columns that can be used for species infos
        applicable_general_inputs = list(set(possible_general_parameters.keys()).intersection(set(possible_species_parameters.keys())))
        applicable_general_inputs = {key: self.general_info[key] for key in applicable_general_inputs if key in self.general_info.keys()}
        #overwrite the defaults with info from general_info, where applicable
        defaults.update(applicable_general_inputs)
        applicable_general_inputs = defaults
        #remove key/value pairs that dont have a meaningful default or a value from the general_info, so they dont interfere later
        to_delete = [key for key, value in applicable_general_inputs.items() if value is None]
        for key in to_delete:
            del applicable_general_inputs[key]
        return applicable_general_inputs

    @staticmethod
    def check_general_keys(general_dict: Dict[str, Any], missing_output_name_ok: bool = False):
        necessary_parameters = list(set(possible_general_parameters).difference(set(possible_species_parameters)))
        missing = [parameter for parameter in necessary_parameters if parameter not in general_dict.keys()]
        if missing_output_name_ok and missing == ["output_name"]:
            return
        if missing:
            raise ValueError(f"Error parsing general info dict! Input parameters {missing} missing!")

    @staticmethod
    def check_species_keys(specie_dict: Dict[str, Any], missing_species_name_ok: bool = False):
        missing = [parameter for parameter in possible_species_parameters.keys() if parameter not in specie_dict.keys()]
        if missing_species_name_ok and missing == ["species_name"]:
            return
        if missing:
            raise ValueError(f"Error parsing specie dict! Input parameters {missing} missing!")

    def parse_species(self, run_dict: Dict[str, Any]):
        #list of parameters in both general and species parameters
        applicable_general_inputs = self.set_up_defaults()
        species_info = []
        for species_dict in run_dict.get("species_data", []):
            curr_specie_dict = {key: read_feature_from_input_dict(species_dict, key) for key in possible_species_parameters.keys() if key in species_dict.keys()}
            #get defaults, then overwrite them with the data that was read in for all keys where data was read in
            curr_general_info = applicable_general_inputs.copy()
            curr_general_info.update(curr_specie_dict)
            #make sure all necessary parameters are filled
            RunInfo.check_species_keys(curr_general_info)
            species_info.append(curr_general_info)
        if not species_info:
            curr_general_info = applicable_general_inputs.copy()
            RunInfo.check_species_keys(curr_general_info, missing_species_name_ok=True)
            species_info.append(curr_general_info)
        self.species_info = species_info

    def parse_general_inputs(self, run_dict: Dict[str, str]):
        #load defaults
        defaults = possible_general_parameters.copy()
        read_data = {key: read_feature_from_input_dict(run_dict, key) for key in possible_general_parameters.keys() if key in run_dict.keys()}
        #overwrite defaults with read data
        defaults.update(read_data)
        general_info = defaults
        #remove empty defaults
        to_delete = [key for key, value in general_info.items() if value is None]
        for key in to_delete:
            del general_info[key]
        general_info["model_case"] = ModelCase.parse(general_info["model_case"])
        #make check
        missing_output_name_ok = general_info["model_case"] == ModelCase.MSR
        RunInfo.check_general_keys(general_dict=general_info, missing_output_name_ok=missing_output_name_ok)
        self.general_info = general_info

    @staticmethod
    def parse(run_dict: Dict[str, Any]):
        run_info_object = RunInfo()
        run_info_object.parse_general_inputs(run_dict=run_dict)
        # load general info first, and use it as defaults for specific species
        run_info_object.parse_species(run_dict)
        if run_info_object.general_info["model_case"] == ModelCase.MSR and len(run_info_object.species_info) < 2:
            raise ValueError(f"Need at least 2 species for MSR training! Only found {len(run_info_object.species_info)}.")
        for species_key in possible_species_parameters.keys():
            if species_key in run_info_object.general_info.keys():
                del run_info_object.general_info[species_key]
        return run_info_object
    
    def __str__(self) -> str:
        self.general_info["model_case"] = str(self.general_info["model_case"])
        specie_info_json = json.dumps(self.species_info, indent=2)
        gen_info_json = json.dumps(self.general_info, indent=2)
        self.general_info["model_case"] = ModelCase.parse(self.general_info["model_case"])
        result = "RunInfo(\n  General Info{"
        for gen_info in gen_info_json.split("\n"):
            if gen_info in ["{", "}"]:
                continue
            result += "\n  " + gen_info
        result += "\n  },\n  Species info["
        for species_info in specie_info_json.split("\n"):
            if species_info in ["[", "]"]:
                continue
            result += "\n  " + species_info
        result += "\n  ]\n)"
        return result


class ParsedTrainingInputs:
    run_infos: List[RunInfo]

    def __init__(self):
        self.run_infos = []

    @staticmethod
    def parse(json_file_name: str) -> ParsedTrainingInputs:
        json_file_name = json_file_name if os.path.isfile(json_file_name) else make_absolute_path(json_file_name, __file__)
        parsed_object = ParsedTrainingInputs()
        with open(json_file_name, "r") as f:
            input_list = json.load(f)
        for i, run_dict in enumerate(input_list):
            try:
                curr_run_info = RunInfo.parse(run_dict)
                parsed_object.run_infos.append(curr_run_info)
            except Exception as e:
                print(f"error reading input run number {i}.")
                print(f"error message is: \"{e}\"")
                print(f"the dictionary that was loaded for the run is the following:")
                print(f"{json.dumps(run_dict, indent=2)}")
        return parsed_object
    
    def __str__(self) -> str:
        result = "ParsedTrainingInputs["
        for info in self.run_infos:
            for info_line in str(info).split("\n"):
                result += "\n  " + info_line
        result += "\n]"
        return result
            


class ModelCase(Enum):
    MSR = auto()
    SSR = auto()
    SSC = auto()
    BOTH = auto()

    @staticmethod
    def parse(input_string: str) -> ModelCase:
        if input_string.lower() == "msr":
            return ModelCase.MSR
        elif input_string.lower() == "ssr":
            return ModelCase.SSR
        elif input_string.lower() == "ssc":
            return ModelCase.SSC
        elif input_string.lower() == "both":
            return ModelCase.BOTH
        else:
            raise ValueError(f"model case \"{input_string}\" not recognized!")
    
    def __str__(self) -> str:
        if self == ModelCase.MSR:
            return "msr"
        elif self == ModelCase.SSR:
            return "ssr"
        elif self == ModelCase.SSC:
            return "ssc"
        elif self == ModelCase.BOTH:
            return "both"
        else:
            raise NotImplementedError("string method was not implemented for this Variant yet!")


class TerminationError(Exception):
    pass


def find_newest_model_path(output_name: str, model_case: str, val_chromosome: str = "", model_path: str = "") -> Dict[str, str]:
    """finds path to newest model fitting the given parameters

    Args:
        output_name (str): output name the was used for model training
        val_chromosome (str): validation chromosome of the model. If it is not given, all models regardless of the val_chromosome will be returned
        model_case (str): SSR or SSC for the model to be loaded
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
            # group 1 is the "(.+)" part of the regex, so the name of the validation chromosome for the model
            if model_case.lower() == "msr":
                chromosome = "model"
            elif val_chromosome:
                chromosome = val_chromosome
            else:
                chromosome = match.group(1)
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


    return fitting_models


def extract_gene(genome: Fasta, extragenic: int, intragenic: int, ignore_small_genes: bool, expected_final_size: int,
                 chrom: str, start: int, end: int, strand: str, ssc_training: bool = False, val_chromosome: Optional[str] = None) -> np.ndarray:
    """extracts the gene flanking region for a single gene and converts it to a numpy encoded one-hot encoding

    Args:
        genome (Fasta): Fasta representation of the genome to extract the gene flanking region from
        extragenic (int): length of the sequence to be extracted before the start and after the end of the gene
        intragenic (int): length of the sequence to be extracted after the start and before the end of the gene
        ignore_small_genes (bool): determines how to deal with genes that are smaller than 2x intragenic. If True,
            these genes will be extracted with the wrong length. If False, central padding will be extended.
        expected_final_size (int): the length the extracted sequence should have at the end
        chrom (str): chrom on which the gene lies
        start (int): start index of the gene
        end (int): end index of the gene
        strand (str): determines whether the gene is on + or - strand

    Returns:
        np.ndarray: One hot encoded gene flanking region as numpy array
    """
    gene_size = end - start
    extractable_intragenic = intragenic if gene_size // 2 > intragenic else gene_size // 2
    prom_start, prom_end = start - extragenic, start + extractable_intragenic
    term_start, term_end = end - extractable_intragenic, end + extragenic

    promoter = one_hot_encode(genome[chrom][prom_start:prom_end])
    terminator = one_hot_encode(genome[chrom][term_start:term_end])
    extracted_size = promoter.shape[0] + terminator.shape[0]
    central_pad_size = expected_final_size - extracted_size

    if ssc_training and chrom != val_chromosome:
            np.random.shuffle(promoter)
            np.random.shuffle(terminator)

    # this means that even with ignore small genes == true, small genes will be extracted.
    # They just dont have the expected size and have to be filtered out later
    pad_size = 20 if ignore_small_genes else central_pad_size

    if strand == '+':
        seq = np.concatenate([
            promoter,
            np.zeros(shape=(pad_size, 4)),
            terminator
        ])
    else:
        seq = np.concatenate([
            terminator[::-1, ::-1],
            np.zeros(shape=(pad_size, 4)),
            promoter[::-1, ::-1]
        ])
        
    return seq


def append_sequence_prediction(tpms: pd.DataFrame, extracted_seqs: Dict[str, Tuple[List[np.ndarray], List[int], List[str]]], expected_final_size: int, chrom: str, gene_id: str, sequence_to_append: np.ndarray) -> None:
    if sequence_to_append.shape[0] == expected_final_size:
        extracted_tuple = extracted_seqs.get(chrom, ())
        if extracted_tuple == ():
            x, y, gene_ids = [], [], []
        else:
            x = extracted_tuple[0]                      #type:ignore
            y = extracted_tuple[1]                      #type:ignore
            gene_ids = extracted_tuple[2]               #type:ignore
        x.append(sequence_to_append)
            # tpms check for for_prediction happened earlier
        if tpms is None:
            y.append("NA")
        else:
            y.append(tpms.loc[gene_id, 'target'])       #type:ignore
        gene_ids.append(gene_id)
        extracted_seqs[chrom] = (x, y, gene_ids)


def extract_genes_prediction(genome: Fasta, annotation: pd.DataFrame, extragenic: int, intragenic: int, ignore_small_genes: bool, tpms: pd.DataFrame, target_chromosomes: Tuple[str, ...], model_case: str, for_prediction: bool = True) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    extracted_seqs = {}
    expected_final_size = 2 * (extragenic + intragenic) + 20
    # tpms are absolutely necessary for training, but not for predictions, so can miss if data is for predictions
    if tpms is None and not for_prediction:
        raise ValueError(f"tpms have to be given if \"for_prediction\" is not set to True!")
    
    for values in annotation.values:
        if model_case.lower() == "msr":
            specie, chrom, start, end, strand, gene_id = values
        else:
            chrom, start, end, strand, gene_id = values

        # skip all chromosomes that are not in the target chromosomes. Empty tuple () means, that all chromosomes should be extracted
        if target_chromosomes != () and chrom not in target_chromosomes:
            continue

        seq = extract_gene(genome, extragenic, intragenic, ignore_small_genes, expected_final_size, chrom, start, end, strand)
        append_sequence_prediction(tpms, extracted_seqs, expected_final_size, chrom, gene_id, seq)

    # convert lists to arrays
    for chrom, tuple_ in extracted_seqs.items():
        x, y, gene_ids = tuple_
        x, y, gene_ids = np.array(x), np.array(y), np.array(gene_ids)
        extracted_seqs[chrom] = (x, y, gene_ids)
    return extracted_seqs


def deep_cre(x_train, y_train, x_val, y_val, output_name, model_case, chrom):
    """

    :param x_train: onehot encoded train matrix
    :param y_train: true targets to x_train
    :param x_val: onehot encoded validation matrix
    :param y_val: target values to x_val
    :param output_name: the start of the output file name such as arabidopsis_leaf to create arabidopsis_leaf_output.csv
    :param model_case: model type which can be SSC, SSR, or MSR
    :param chrom: chromosome name
    :return: [accuracy, auROC, auPR]
    """
    input_seq = Input(shape=(x_train.shape[1], x_train.shape[2]))

    # Conv block 1
    conv = Conv1D(filters=64, kernel_size=8, padding='same')(input_seq)
    conv = Activation('relu')(conv)
    conv = Conv1D(filters=64, kernel_size=8, padding='same')(conv)
    conv = Activation('relu')(conv)
    conv = MaxPool1D(pool_size=8, padding='same')(conv)
    conv = Dropout(0.25)(conv)

    # Conv block 2 and 3
    for n_filters in [128, 64]:
        conv = Conv1D(filters=n_filters, kernel_size=8, padding='same')(conv)
        conv = Activation('relu')(conv)
        conv = Conv1D(filters=n_filters, kernel_size=8, padding='same')(conv)
        conv = Activation('relu')(conv)
        conv = MaxPool1D(pool_size=8, padding='same')(conv)
        conv = Dropout(0.25)(conv)

    # Fully connected block
    output = Flatten()(conv)
    output = Dense(128)(output)
    output = Activation('relu')(output)
    output = Dropout(0.25)(output)
    output = Dense(64)(output)
    output = Activation('relu')(output)
    output = Dense(1)(output)
    output = Activation('sigmoid')(output)

    model = Model(inputs=input_seq, outputs=output)
    model.summary()

    time_stamp = get_time_stamp()
    file_name = get_filename_from_path(__file__)
    checkpoint_path = make_absolute_path("saved_models", f"{output_name}_{chrom}_{model_case}_{file_name}_{time_stamp}.h5", start_file=__file__)
    model_chkpt = ModelCheckpoint(filepath=checkpoint_path,
                                  save_best_only=True,
                                  verbose=1)
    early_stop = EarlyStopping(patience=10)
    reduce_lr = ReduceLROnPlateau(patience=5, factor=0.1)
    model.compile(loss='binary_crossentropy', optimizer=Adam(0.0001),
                  metrics=['accuracy', AUC(curve="ROC", name='auROC'), AUC(curve="PR", name='auPR')])
    model.fit(x_train, y_train, batch_size=64, epochs=100, validation_data=(x_val, y_val),
              callbacks=[early_stop, model_chkpt, reduce_lr])

    loaded_model = load_model(checkpoint_path)
    output = loaded_model.evaluate(x_val, y_val)
    return output


def mask_sequences(train_seqs: np.ndarray, val_seqs: np.ndarray, extragenic: int, intragenic: int):
    """masking the start and end codon of the sequence as zeros, according to original paper

    Args:
        train_seqs (np.ndarray): one hot encoded training sequences
        val_seqs (np.ndarray): one hot encoded validation sequences
        extragenic (int): length of the promoter / terminator region that is extracted
        intragenic (int): length of the UTRs that is extracted
    """
    # Masking
    train_seqs[:, extragenic:extragenic + 3, :] = 0
    train_seqs[:, extragenic + (intragenic * 2) + 17:extragenic + (intragenic * 2) + 20, :] = 0
    val_seqs[:, extragenic:extragenic + 3, :] = 0
    val_seqs[:, extragenic + (intragenic * 2) + 17:extragenic + (intragenic * 2) + 20, :] = 0


def append_sequence_training(include_as_validation_gene: bool, include_as_training_gene: bool, expected_final_size, train_seqs, val_seqs, train_targets, val_targets, tpms, gene_id, seq) -> Tuple[int, int]:
    added_val, added_training = 0, 0
    if seq.shape[0] == expected_final_size:
        if include_as_validation_gene:
            val_seqs.append(seq)
            val_targets.append(tpms.loc[gene_id, 'target'])
            added_val = 1
        # train: all species except one 
        elif include_as_training_gene:
            train_seqs.append(seq)
            train_targets.append(tpms.loc[gene_id, 'target'])
            added_training = 1
    return added_val, added_training


def calculate_conditions(val_chromosome, model_case, train_val_split, test_specie, validation_genes, current_val_size, current_train_size, target_val_size, target_train_size, specie, chrom, gene_id):
    if model_case.lower() == "msr":
        include_in_validation_set = specie in test_specie['specie'].values                      #type:ignore
        include_in_training_set = not include_in_validation_set
    elif train_val_split.lower() == "no":
        include_in_validation_set = chrom == val_chromosome
        include_in_training_set = not include_in_validation_set
    else:
        include_in_validation_set = current_val_size < target_val_size and gene_id in validation_genes
        include_in_training_set = current_train_size < target_train_size
    return include_in_validation_set,include_in_training_set


def set_up_validation_genes(genes_picked, pickled_key, model_case):
    if model_case.lower() in ["ssr", "ssc"]:
        with open(genes_picked, 'rb') as handle:
            validation_genes = pickle.load(handle)
            validation_genes = validation_genes[pickled_key]
    else:
        validation_genes = []
    return validation_genes


def load_input_files_training(genome_file_name, annotation_file_name, tpm_file_name, model_case):
    loaded = load_input_files(genome_file_name=genome_file_name, annotation_file_name=annotation_file_name, tpm_counts_file_name=tpm_file_name, model_case=model_case)
    # genome_path = genome_path if os.path.isfile(genome_path) else make_absolute_path("genome", genome_path, start_file=__file__)     
    # tpm_path = tpm_path if os.path.isfile(tpm_path) else make_absolute_path("tpm_counts", tpm_path, start_file=__file__)  # tpm_targets = f"tpm_{p_keys}.csv"
    # annotation_path = annotation_path if os.path.isfile(annotation_path) else make_absolute_path("gene_models", annotation_path, start_file=__file__)  
    # genome = Fasta(filename=genome_path, as_raw=True, read_ahead=10000, sequence_always_upper=True)
    # tpms = pd.read_csv(filepath_or_buffer=tpm_path, sep=',')
    # tpms.set_index('gene_id', inplace=True)
    # annotation = load_annotation_msr(annotation_file_name) if model_case.lower() == "msr" else load_annotation(annotation_file_name)
    return loaded["genome"], loaded["tpms"], loaded["annotation"]


def set_up_train_val_split_variables(annotation: pd.DataFrame):
    # 80 / 20 train-val splitting 
    total_sequences = len(annotation)
    target_val_size = int(total_sequences * 0.2)  # Target size for validation set (20%)
    target_train_size = total_sequences - target_val_size  # Target size for training set (80%)
    return target_val_size, target_train_size


def save_skipped_genes(skipped_genes):
    if skipped_genes:  # This checks if the set/list is not empty
        timestamp = get_time_stamp()
        filename = f'skipped_genes_{timestamp}.txt'
        with open(filename, 'w') as skipped_genes_file:
            for gene in skipped_genes:
                skipped_genes_file.write(f"{gene}\n")
        
        if len(skipped_genes) > 5000:
            print(f"Warning: {len(skipped_genes)} gene IDs were skipped. Please check that the gene name formats are identical in both the GTF and TPM files.")
                
        else:
            print(f"Some gene IDs in the gtf file were not found in TPM counts. Skipped gene IDs have been written to {filename}.")


def extract_genes_training(genome_path: str, annotation_path: str, tpm_path: str, extragenic: int, intragenic: int, genes_picked, pickled_key, val_chromosome,
                model_case, ignore_small_genes, train_val_split, input_filename: Optional[str] = None, test_specie: Optional[pd.DataFrame] = None):
    """
     This function extract sequences from the genome. It implements a gene size aware padding
    :param genome: reference genome from Ensembl Plants database
    :param annot:  gtf file matching the reference genome
    :param tpm_targets: count file target true targets.
    :param extragenic: length of promoter and terminator
    :param intragenic: length of 5' and 3' UTR
    :param genes_picked: pickled file containing genes to filter into validation set. For gene family splitting
    :param val_chromosome: validation chromosome
    :param model_case: model type which can be SSC, SSR or MSR
    :param pickled_key: key to pickled file name
    :param ignore_small_genes: filter genes smaller than 1000 bp
    :param train_val_split: create a training dataset with 80% of genes across all chromosomes and 20% of genes in the validation dataset
    :return: [one_hot train set, one_hot val set, train targets, val targets]
    """
    if model_case.lower() == "msr":
        if test_specie is None:
            raise ValueError("test specie parameter necessary for msr training!")
    
    genome, tpms, annotation = load_input_files_training(genome_path, annotation_path, tpm_path, model_case)
    ssc_training = model_case.lower() == "ssc"        
    validation_genes = set_up_validation_genes(genes_picked, pickled_key, model_case)
    expected_final_size = 2*(extragenic + intragenic) + 20

    # random shuffle annot values to generate 3iterations of train val split 
    # the order of annot_values and pickle file decide which gene goes into training or validation
    if model_case.lower() in ["ssr", "ssc"] and train_val_split.lower() == "yes":
        annotation = annotation.sample(frac=1, random_state=42).reset_index(drop=True)

    #only relevant for train_val_split == "yes"
    current_val_size, current_train_size = 0, 0
    target_val_size, target_train_size = set_up_train_val_split_variables(annotation=annotation)
        
    train_seqs, val_seqs, train_targets, val_targets = [], [], [], []
    skipped_genes = [] 
    for specie, chrom, start, end, strand, gene_id in annotation.values:
        if gene_id not in tpms.index:
            skipped_genes.append(gene_id)
            continue
            
        include_in_validation_set, include_in_training_set = calculate_conditions(val_chromosome, model_case, train_val_split, test_specie,
                                                                                  validation_genes, current_val_size, current_train_size,
                                                                                  target_val_size, target_train_size, specie, chrom, gene_id)

        seq = extract_gene(genome=genome, extragenic=extragenic, intragenic=intragenic, ignore_small_genes=ignore_small_genes,
                            expected_final_size=expected_final_size, chrom=chrom, start=start, end=end, strand=strand, ssc_training=ssc_training,
                            val_chromosome=val_chromosome)
        added_val, added_train = append_sequence_training(include_as_validation_gene=include_in_validation_set, include_as_training_gene=include_in_training_set, train_targets=train_targets,
                                                          expected_final_size=expected_final_size, train_seqs=train_seqs, val_seqs=val_seqs, val_targets=val_targets, tpms=tpms, gene_id=gene_id, seq=seq)
        current_val_size += added_val
        current_train_size += added_train
                    

    if train_val_split.lower() == "yes":
        # check if desired 80/20 split is reached 
        if current_val_size < target_val_size:
            raise ValueError(f"Validation set is not 20%. Current size: {current_val_size} genes, "
                                f"Target size: {target_val_size} genes. Total genes in pickle file: {len(validation_genes)}. "
                                f"(Only genes from pickle file can be in the validation set.)")

    save_skipped_genes(skipped_genes)
    
    train_seqs, val_seqs, train_targets, val_targets  = np.array(train_seqs), np.array(val_seqs), np.array(train_targets), np.array(val_targets)
    print(train_seqs.shape, val_seqs.shape)
    if train_seqs.size == 0 or val_seqs.size == 0:
        if model_case.lower() == "msr":
            raise TerminationError("Validation sequences are empty. Terminating MSR run!")
        raise ValueError("Validation sequences or training sequences are empty.")

    mask_sequences(train_seqs=train_seqs, val_seqs=val_seqs, extragenic=extragenic, intragenic=intragenic)
    return train_seqs, train_targets, val_seqs, val_targets


def balance_dataset(x, y):
    """
    This function randomly down samples the majority class to balance the dataset
    :param x: one-hot encoded set
    :param y: true targets
    :return: returns a balance set
    """
    # Random down sampling to balance data
    low_train, high_train = np.where(y == 0)[0], np.where(y == 1)[0]
    min_class = min([len(low_train), len(high_train)])
    selected_low_train = np.random.choice(low_train, min_class, replace=False)
    selected_high_train = np.random.choice(high_train, min_class, replace=False)
    x_train = np.concatenate([
        np.take(x, selected_low_train, axis=0),
        np.take(x, selected_high_train, axis=0)
    ], axis=0)
    y_train = np.concatenate([
        np.take(y, selected_low_train, axis=0),
        np.take(y, selected_high_train, axis=0)
    ], axis=0)
    x_train, y_train = shuffle(x_train, y_train, random_state=42)#type:ignore
    return x_train, y_train


def train_deep_cre(genome_path: str, annotation_path: str, tpm_path: str, upstream: int, downstream: int, genes_picked, val_chromosome, output_name,
                   model_case: str, pickled_key: Optional[str], ignore_small_genes: bool, train_val_split,  test_specie: Optional[pd.DataFrame] = None,
                   input_filename: Optional[str] = None):
    train_seqs, train_targets, val_seqs, val_targets = extract_genes_training(genome_path, annotation_path, tpm_path, upstream, downstream,
                                                                   genes_picked, pickled_key, val_chromosome, model_case, ignore_small_genes,
                                                                   train_val_split=train_val_split, test_specie=test_specie,
                                                                   input_filename=input_filename)
    x_train, y_train = balance_dataset(train_seqs, train_targets)
    x_val, y_val = balance_dataset(val_seqs, val_targets)
    output = deep_cre(x_train=x_train,
                      y_train=y_train,
                      x_val=x_val,
                      y_val=y_val,
                      output_name=output_name,
                      model_case=model_case,
                      chrom=val_chromosome)
    return output



def parse_args():
    parser = argparse.ArgumentParser(
                        prog='deepCRE',
                        description="""
                        This script performs the deepCRE training. We assume you have the following three directories:
                        tmp_counts (contains your counts files), genome (contains the genome fasta files),
                        gene_models (contains the gtf files)
                        """)
    parser.add_argument('--input',
                        help="""
                        For model case SSR/SSC: This is a six column csv file with entries: species, genome, gtf, tpm, output name,
                        number of chromosomes and pickle_key. \n 
                        For model case MSR: This is a five column csv file with entries: species, genome, gtf, tpm, output name.
                        """, required=True)
    parser.add_argument('--pickle', help="path to pickle file. Necessary for SSR and SSC training.", required=False)
    parser.add_argument('--model_case', help="Can be SSC, SSR or MSR", required=True, choices=["msr", "ssr", "ssc", "both"])
    parser.add_argument('--ignore_small_genes', help="Ignore small genes, can be yes or no", required=False, choices=["yes", "no"], default="yes")
    parser.add_argument('--train_val_split', help="For SSR /SSC training: Creates a training/validation dataset with 80%/20% of genes, can be yes or no", required=False, choices=["yes", "no"], default="no")


    args = parser.parse_args()
    return args


def train_msr(data: pd.DataFrame, input_file_name: str, failed_trainings: List[Tuple], file_name: str, args):
    print(f'Multi species Training: ---------------------\n')
    ignore_small_genes = args.ignore_small_genes.lower() == "yes"

    naming = "_".join([specie[:3] for specie in data['specie'].unique()])
    # generate concat files

    tpm_path = combine_tpms(data=data, input_filename=input_file_name)
    genome_path = combine_fasta(data=data)
    annotation_path = combine_annotations(data=data)

    results_genome = []

    for i, specie in enumerate(data['specie'].unique()):
        try:
            test_specie = data.copy()
            test_specie = test_specie[test_specie['specie'] == specie]
            train_specie = data.copy()
            train_specie = train_specie[train_specie['specie'] != specie]

            print(f'Training on species: {train_specie["specie"].unique()}')
            print(f'Testing on specie: {test_specie["specie"].unique()}')

            output_name = test_specie['output'].values[0]
            print(f"Output name for training: {output_name}")

            results = train_deep_cre(
                genome_path=genome_path,
                annotation_path=annotation_path,
                tpm_path=tpm_path,
                upstream=1000,
                downstream=500,
                genes_picked=args.pickle,
                val_chromosome="model",
                output_name=output_name,
                model_case=args.model_case,
                pickled_key=None,
                ignore_small_genes=ignore_small_genes,
                train_val_split=args.train_val_split,
                test_specie=test_specie,
                input_filename=input_file_name,
            ) 
            results_with_info = {
                    'loss': results[0],
                    'accuracy': results[1],
                    'auROC': results[2],
                    'auPR': results[3],
                    'test_specie': test_specie['specie'].values[0],
                }
                
            results_genome.append(results_with_info)
            print(f"Results for genome: {genome_path}, validation species: {test_specie['specie'].values[0]}: {results}")
                                                
        except TerminationError as e:
            raise e
        except Exception as e:
            raise e
            print(e)
            failed_trainings.append((output_name, i, e))

    results_genome = pd.DataFrame(results_genome, columns=['test_specie','loss', 'accuracy', 'auROC', 'auPR'])
    save_file = make_absolute_path('results', f"{naming}_{input_file_name}_msr_{file_name}_{get_time_stamp()}.csv", start_file=__file__)
    results_genome.to_csv(path_or_buf=save_file, index=False)
    print(results_genome.head())
        
    result_summary(failed_trainings=failed_trainings, input_length=len(data), script=get_filename_from_path(__file__))


def train_ssr_ssc(data: pd.DataFrame, args, failed_trainings: List[Tuple], file_name: str):
    ignore_small_genes = args.ignore_small_genes.lower() == "yes"
    for genome, gtf, tpm_counts, output_name, chromosomes_file, pickled_key in data.values:
        try:
            print(f'Single species Training on genome: ---------------------\n')
            print(genome)
            print('\n------------------------------\n')
            results_genome = []
            
            # Original chromosome-based split
            if args.train_val_split.lower() == 'no': 
                chromosomes = pd.read_csv(filepath_or_buffer=f'genome/{chromosomes_file}', header=None).values.ravel().tolist()
                for i, val_chrom in enumerate(chromosomes):
                    print(f"Using chromosome {val_chrom} as validation chromosome")
                    results = train_deep_cre(genome_path=genome, annotation_path=gtf, tpm_path=tpm_counts, upstream=1000,
                                                downstream=500, genes_picked=args.pickle, val_chromosome=str(val_chrom),
                                                output_name=output_name, model_case=args.model_case, pickled_key=pickled_key,
                                                ignore_small_genes=ignore_small_genes, train_val_split=args.train_val_split)
                    results_genome.append(results)
                    print(f"Results for genome: {genome}, chromosome: {val_chrom}: {results}")
                
            
            # New random gene-based split
            elif args.train_val_split.lower() == 'yes':
                chromosomes=[1,2,3]
                print(f"Using random 80/20 gene-based split for validation")
                for i, val_chrom in enumerate(chromosomes):
                    results = train_deep_cre(
                                    genome_path=genome,
                                    annotation_path=gtf,
                                    tpm_path=tpm_counts,
                                    upstream=1000,
                                    downstream=500,
                                    genes_picked=args.pickle,
                                    val_chromosome=val_chrom,  # 3 iterations
                                    output_name=output_name,
                                    model_case=args.model_case,
                                    pickled_key=pickled_key,
                                    ignore_small_genes=ignore_small_genes,
                                    train_val_split=args.train_val_split
                                )
                    
                    results_genome.append(results)
                    print(f"Results for genome: {genome}, iteration: {val_chrom}: {results}")
                
            results_genome = pd.DataFrame(results_genome, columns=['loss', 'accuracy', 'auROC', 'auPR'])
            save_file = make_absolute_path('results', f"{output_name}_{args.model_case}_{file_name}_ssr_{get_time_stamp()}.csv", start_file=__file__)
            results_genome.to_csv(path_or_buf=save_file, index=False)
            print(results_genome.head())

        except Exception as e:
            print(e)
            failed_trainings.append((output_name, i, e))

    result_summary(failed_trainings=failed_trainings, input_length=len(data), script=get_filename_from_path(__file__))


def main():
    args = parse_args()
    model_case = args.model_case 
    input_file_name = args.input.split('.')[0]

    dtypes = {0: str, 1: str, 2: str, 3: str, 4: str, 5: str, 6: str} if model_case.lower() == "msr" else {0: str, 1: str, 2: str, 3: str, 4: str, 5: str}
    names = ['specie','genome', 'gtf', 'tpm', 'output', "chroms", "p_key"] if model_case.lower() == "msr" else ['genome', 'gtf', 'tpm', 'output', 'chroms', 'p_key']
    data = pd.read_csv(args.input, sep=',', header=None, dtype=dtypes, names = names)

    expected_columns = len(names)
    if data.shape[1] != expected_columns:
        raise Exception("Input file incorrect. Your input file must contain 7 columns and must be .csv")
    
    
    model_cases = ["ssr", "ssc"] if args.model_case == "both" else [args.model_case]
    print(data.head())

    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    training_results_path = make_absolute_path('results', "training", start_file=__file__)
    models_path = make_absolute_path("saved_models", start_file=__file__)
    if not os.path.exists(training_results_path):
        os.makedirs(training_results_path)
    if not os.path.exists(models_path):
        os.makedirs(models_path)

    file_name = get_filename_from_path(__file__)
    failed_trainings = []
    for model_case in model_cases:
        
        # MSR training 
        if model_case.lower() == "msr":
            train_msr(data=data, input_file_name=input_file_name, failed_trainings=failed_trainings, file_name=file_name, args=args)
        
        if model_case.lower() in ["ssr", "ssc"]:
            train_ssr_ssc(data=data, args=args, failed_trainings=failed_trainings, file_name=file_name)

if __name__ == "__main__":
    # main()
    parsed = ParsedTrainingInputs.parse("inputs.json")
    print(parsed)
    print(parsed.run_infos[0])
    # TODO: test everything
        # especially gene extraction for all cases, ESPECIALL train_vla_split
    #TODO: inputs.json should work
    #TODO: make sure all models / inputs / outputs are found correctly
    #TODO: unify methods
    #TODO: talk about 80/20 split in general; and MSR in general
    #TODO: consistenly use ModelCase enum