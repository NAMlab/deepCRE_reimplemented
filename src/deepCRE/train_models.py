from __future__ import annotations
import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from tensorflow.keras.layers import Dropout, Dense, Input, Conv1D, Activation, MaxPool1D, Flatten               #type:ignore
from tensorflow.keras.optimizers import Adam                                                                    #type:ignore
from tensorflow.keras import Model                                                                              #type:ignore
from tensorflow.keras.models import load_model                                                                  #type:ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau                        #type:ignore
from tensorflow.keras.metrics import AUC                                                                        #type:ignore
import pickle
import numpy as np
from pyfaidx import Fasta
from sklearn.utils import shuffle
import re
import sys

from deepCRE.parsing import ParsedInputs, RunInfo, ModelCase
from deepCRE.utils import get_filename_from_path, get_time_stamp, one_hot_encode, make_absolute_path, result_summary, combine_annotations, combine_fasta, combine_tpms, load_input_files


class TerminationError(Exception):
    pass


def find_newest_model_path(output_name: str, model_case: ModelCase, val_chromosome: str = "", model_path: str = "") -> Dict[str, str]:
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
        regex_string = f"^{output_name}_(.+)_{model_case}_train_models_\d+_\d+\.h5$"                                                                    #type:ignore
    else:
        regex_string = f"^{output_name}_{val_chromosome}_{model_case}_train_models_\d+_\d+\.h5$"                                                        #type:ignore
        
    regex = re.compile(regex_string)
    #print(regex)
    candidate_models = [model for model in os.listdir(path_to_models)]
    fitting_models = {}
    for candidate in candidate_models:
        match = regex.match(candidate)

        if match:
            # group 1 is the "(.+)" part of the regex, so the name of the validation chromosome for the model
            if val_chromosome:
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


def append_sequence_prediction(tpms: Optional[pd.DataFrame], extracted_seqs: Dict[str, Tuple[List[np.ndarray], List[int], List[str]]],
                               expected_final_size: int, chrom: str, gene_id: str, sequence_to_append: np.ndarray) -> None:
    if sequence_to_append.shape[0] == expected_final_size:
        extracted_tuple = extracted_seqs.get(chrom, ())
        if extracted_tuple == ():
            x, y, gene_ids = [], [], []
        else:
            x = extracted_tuple[0]                      #type:ignore
            y = extracted_tuple[1]                      #type:ignore
            gene_ids = extracted_tuple[2]               #type:ignore
        if tpms is None:
            y.append("NA")                              #type:ignore
        else:
            try:
                y.append(tpms.loc[gene_id, 'target'])       #type:ignore
            except KeyError:
                # if the gene is not in the TPM file, it is not used for predictions
                return
        x.append(sequence_to_append)
        gene_ids.append(gene_id)
        extracted_seqs[chrom] = (x, y, gene_ids)


def extract_genes_prediction(genome: Fasta, annotation: pd.DataFrame, extragenic: int, intragenic: int, ignore_small_genes: bool, tpms: Optional[pd.DataFrame],
                             target_chromosomes: Tuple[str, ...]) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    extracted_seqs = {}
    expected_final_size = 2 * (extragenic + intragenic) + 20
    # tpms are absolutely necessary for training, but not for predictions, so can miss if data is for predictions
    for values in annotation.values:
        if len(annotation.columns) == 6:
            specie, chrom, start, end, strand, gene_id = values
            # use specie in place of chromosome because for msr case genes need to be ordered by species, not chromosome
        else:
            chrom, start, end, strand, gene_id = values
            specie = None

        # skip all chromosomes that are not in the target chromosomes. Empty tuple () means, that all chromosomes should be extracted
        if target_chromosomes != () and chrom not in target_chromosomes:
            continue

        seq = extract_gene(genome, extragenic, intragenic, ignore_small_genes, expected_final_size, chrom, start, end, strand)
        if specie is not None:
            chrom = specie
        append_sequence_prediction(tpms=tpms, extracted_seqs=extracted_seqs, expected_final_size=expected_final_size, chrom=chrom, gene_id=gene_id, sequence_to_append=seq)

    # convert lists to arrays
    for chrom, tuple_ in extracted_seqs.items():
        x, y, gene_ids = tuple_
        x, y, gene_ids = np.array(x), np.array(y), np.array(gene_ids)
        extracted_seqs[chrom] = (x, y, gene_ids)
    return extracted_seqs


def deep_cre(x_train, y_train, x_val, y_val, output_name, model_case, chrom, time_stamp: str, test: bool = False):
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

    file_name = get_filename_from_path(__file__)
    potential_folder = os.path.dirname(output_name)
    if os.path.exists(potential_folder) and not os.path.isfile(potential_folder):
        checkpoint_path = output_name
    else:
        checkpoint_path = make_absolute_path("saved_models", f"{output_name}_{chrom}_{model_case}_{file_name}_{time_stamp}.h5", start_file=__file__)
    model_chkpt = ModelCheckpoint(filepath=checkpoint_path,
                                  save_best_only=True,
                                  verbose=1)
    early_stop = EarlyStopping(patience=3) if test else EarlyStopping(patience=10)
    reduce_lr = ReduceLROnPlateau(patience=3, factor=0.1) if test else ReduceLROnPlateau(patience=5, factor=0.1)
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


def calculate_conditions(val_chromosome, model_case, train_val_split: bool, test_specie, validation_genes, current_val_size, current_train_size, target_val_size, target_train_size, specie, chrom, gene_id):
    if model_case == ModelCase.MSR:
        include_in_validation_set = specie == test_specie                      #type:ignore
        include_in_training_set = not include_in_validation_set
    elif not train_val_split:
        include_in_validation_set = chrom == val_chromosome and gene_id in validation_genes
        include_in_training_set = chrom != val_chromosome
    else:
        include_in_validation_set = current_val_size < target_val_size and gene_id in validation_genes
        include_in_training_set = current_train_size < target_train_size
    return include_in_validation_set,include_in_training_set


def set_up_validation_genes(genes_picked, pickled_key, model_case):
    genes_picked = genes_picked if os.path.isfile(genes_picked) else make_absolute_path(genes_picked, start_file=__file__)
    if model_case in [ModelCase.SSR, ModelCase.SSC]:
        with open(genes_picked, 'rb') as handle:
            validation_genes = pickle.load(handle)
            validation_genes = validation_genes[pickled_key]
    else:
        validation_genes = []
    return validation_genes


def load_input_files_training(genome_file_name, annotation_file_name, tpm_file_name, model_case):
    loaded = load_input_files(genome_file_name=genome_file_name, annotation_file_name=annotation_file_name, tpm_counts_file_name=tpm_file_name, model_case=str(model_case))
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


def save_skipped_genes(skipped_genes, time_stamp: str):
    if skipped_genes:  # This checks if the set/list is not empty
        file_name = f'skipped_genes_{time_stamp}.txt'
        file_name = make_absolute_path("results", "training", file_name, start_file=__file__)
        with open(file_name, 'w') as skipped_genes_file:
            for gene in skipped_genes:
                skipped_genes_file.write(f"{gene}\n")
        
        if len(skipped_genes) > 5000:
            print(f"Warning: {len(skipped_genes)} gene IDs were skipped. Please check that the gene name formats are identical in both the GTF and TPM files. Skipped gene IDs have been written to {file_name}.")
        else:
            print(f"Some gene IDs in the gtf file were not found in TPM counts. Skipped gene IDs have been written to {file_name}.")


def extract_genes_training(genome_path: str, annotation_path: str, tpm_path: str, extragenic: int, intragenic: int, genes_picked, pickled_key, val_chromosome,
                model_case, ignore_small_genes, train_val_split, time_stamp: str, test_specie: Optional[pd.DataFrame] = None):
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
    if model_case == ModelCase.MSR:
        if test_specie is None:
            raise ValueError("test specie parameter necessary for msr training!")
    
    genome, tpms, annotation = load_input_files_training(genome_path, annotation_path, tpm_path, model_case)
    ssc_training = model_case == ModelCase.SSC        
    validation_genes = set_up_validation_genes(genes_picked, pickled_key, model_case)
    expected_final_size = 2*(extragenic + intragenic) + 20

    # random shuffle annot values to generate 3iterations of train val split 
    # the order of annot_values and pickle file decide which gene goes into training or validation
    if model_case in [ModelCase.SSC, ModelCase.SSR] and train_val_split:
        annotation = annotation.sample(frac=1, random_state=42).reset_index(drop=True)

    #only relevant for train_val_split == "yes"
    current_val_size, current_train_size = 0, 0
    target_val_size, target_train_size = set_up_train_val_split_variables(annotation=annotation)
        
    train_seqs, val_seqs, train_targets, val_targets = [], [], [], []
    skipped_genes = [] 
    if len(annotation.columns) == 5:
        annotation["species"] = "specie unknown"
        annotation = annotation[['species', 'Chromosome', 'Start', 'End', 'Strand', 'gene_id']]
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
                    

    if train_val_split:
        # check if desired 80/20 split is reached 
        if current_val_size < target_val_size:
            raise ValueError(f"Validation set is not 20%. Current size: {current_val_size} genes, "
                             f"Target size: {target_val_size} genes. Total genes in pickle file: {len(validation_genes)}. "
                             f"(Only genes from pickle file can be in the validation set.)")

    save_skipped_genes(skipped_genes, time_stamp=time_stamp)
    
    train_seqs, val_seqs, train_targets, val_targets  = np.array(train_seqs), np.array(val_seqs), np.array(train_targets), np.array(val_targets)
    print(train_seqs.shape, val_seqs.shape)
    if train_seqs.size == 0 or val_seqs.size == 0:
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


def train_deep_cre(genome_path: str, annotation_path: str, tpm_path: str, extragenic: int, intragenic: int, genes_picked, val_chromosome, output_name,
                   model_case: str, ignore_small_genes: bool, train_val_split: bool, time_stamp: str,  test_specie: Optional[pd.DataFrame] = None, pickled_key: Optional[str] = None, test: bool = False):
    train_seqs, train_targets, val_seqs, val_targets = extract_genes_training(genome_path, annotation_path, tpm_path, extragenic, intragenic,
                                                                   genes_picked, pickled_key, val_chromosome, model_case, ignore_small_genes,
                                                                   train_val_split=train_val_split, test_specie=test_specie, time_stamp=time_stamp)
    x_train, y_train = balance_dataset(train_seqs, train_targets)
    x_val, y_val = balance_dataset(val_seqs, val_targets)
    output = deep_cre(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, output_name=output_name,
                      model_case=model_case, chrom=val_chromosome, time_stamp=time_stamp, test=test)
    return output



def parse_args():
    parser = argparse.ArgumentParser(
                        prog='deepCRE',
                        description="""
                        This script performs the deepCRE training.""")
    parser.add_argument('--input', "-i",
                        help="""json file containing the required input parameters. possible parameters can be found in the readme.md file.""", required=True)

    args = parser.parse_args()
    return args


def run_msr(species_info: List[Dict[str, Any]], general_info: Dict[str, Any], time_stamp: str, test: bool = False) -> List[Dict[str, Any]]:
    #msr stuff
    species: List[str] = [specie["species_name"] for specie in species_info]
    naming = "_".join([specie.replace(" ", "").replace(".", "") for specie in species])

    # generate concat files
    tpm_path = combine_tpms(species_data=species_info, naming=naming)
    genome_path = combine_fasta(species_data=species_info, naming=naming)
    annotation_path = combine_annotations(species_data=species_info, naming=naming)
    combined_results = []

    for test_specie_info in species_info:
        train_species = [specie for specie in species if specie != test_specie_info["species_name"]]
        print(f'Training on species: {train_species}')
        print(f'Testing on specie: {test_specie_info["species_name"]}')

        output_name = general_info['output_name']
        print(f"Output name for training: {output_name}")

        results = train_deep_cre(
            genome_path=genome_path, annotation_path=annotation_path, tpm_path=tpm_path, extragenic=general_info["extragenic"],
            intragenic=general_info["intragenic"], genes_picked=test_specie_info["pickle_file"], val_chromosome=test_specie_info["species_name"],
            output_name=general_info["output_name"], model_case=general_info["model_case"], ignore_small_genes=general_info["ignore_small_genes"],
            train_val_split=general_info["train_val_split"], test_specie=test_specie_info["species_name"], time_stamp=time_stamp, test=test
        ) 
        results_with_info = {
            'loss': results[0],
            'accuracy': results[1],
            'auROC': results[2],
            'auPR': results[3],
            'test': test_specie_info['species_name'],
        }
            
        combined_results.append(results_with_info)
        print(f"Results for genome: {genome_path}, validation species: {test_specie_info['species_name']}:\n{results}")
    return combined_results


def get_chromosomes(species_info: List[Dict[str, Any]]) -> List[str]:
    chromosomes = species_info[0]["chromosomes"]
    if not chromosomes:
        raise ValueError("For SSR/SSC training, a list of chromosomes must be provided, if train_val_split isnt True!")
    return chromosomes


def run_ssr(species_info: List[Dict[str, Any]], general_info: Dict[str, Any], time_stamp: str, test: bool = False) -> List [Dict[str, Any]]:
    specie_info = species_info[0]
    print(f'Single species Training on genome:\n------------------------------\n')
    print(specie_info["genome"])
    print('\n------------------------------\n')
    if general_info["train_val_split"]:
        print(f"Using random 80/20 gene-based split for validation")
        chromosomes=[1,2,3]
    else:
        chromosomes = get_chromosomes(species_info)
    combined_results = []
    for val_chrom in chromosomes:
        print(f"Using chromosome {val_chrom} as validation chromosome")
        results = train_deep_cre(genome_path=specie_info["genome"], annotation_path=specie_info["annotation"], tpm_path=specie_info["targets"], extragenic=general_info["extragenic"],
                                        intragenic=general_info["intragenic"], genes_picked=specie_info["pickle_file"], val_chromosome=val_chrom,
                                        output_name=general_info["output_name"], model_case=general_info["model_case"], pickled_key=specie_info["pickle_key"],
                                        ignore_small_genes=general_info["ignore_small_genes"], train_val_split=general_info["train_val_split"], time_stamp=time_stamp, test=test)
        results_with_info = {
            'loss': results[0],
            'accuracy': results[1],
            'auROC': results[2],
            'auPR': results[3],
            'test': val_chrom,
        }
        combined_results.append(results_with_info)
        print(f"Results for genome: {specie_info['genome']}, chromosome: {val_chrom}: {results}")
    return combined_results


def train_models(inputs: ParsedInputs, failed_trainings: List[Tuple[str, int, Exception]], input_length: int, test: bool = False):
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    training_results_path = make_absolute_path('results', "training", start_file=__file__)
    models_path = make_absolute_path("saved_models", start_file=__file__)
    os.makedirs(training_results_path, exist_ok=True)
    os.makedirs(models_path, exist_ok=True)

    file_name = get_filename_from_path(__file__)
    time_stamp = get_time_stamp()
    run_info: RunInfo
    for i, run_info in enumerate(inputs):             #type:ignore
        try:
            print(i, run_info)
            gen_info = run_info.general_info
            spec_info = run_info.species_info
            if run_info.is_msr():
                results = run_msr(species_info=spec_info, general_info=gen_info, time_stamp=time_stamp, test=test)
            else:
                results = run_ssr(species_info=spec_info, general_info=gen_info, time_stamp=time_stamp, test=test)
            results = pd.DataFrame(results, columns=['test', 'loss', 'accuracy', 'auROC', 'auPR'])
            save_file = os.path.join(training_results_path, f"{gen_info['output_name']}_{file_name}_{gen_info['model_case']}_{time_stamp}.csv")
            results.to_csv(path_or_buf=save_file, index=False)
            print(results.head())
        except TerminationError as e:
            print(run_info)
            raise e
        except Exception as e:
            print(e)
            print(run_info)
            failed_trainings.append((run_info.general_info["output_name"], i, e))
    result_summary(failed_trainings=failed_trainings, input_length=input_length, script=get_filename_from_path(__file__))
    return failed_trainings


def parse_input_file(input_file: str) -> Tuple[ParsedInputs, List[Tuple[str, int, Exception]], int]:
    possible_general_parameters = {
        "genome": None,
        "annotation": None,
        "targets": None,
        "output_name": None,
        "chromosomes": "",
        "pickle_key": None,
        "pickle_file": "validation_genes.pickle",
        "model_case": None,
        "ignore_small_genes": True,
        "train_val_split": False,
        "extragenic": 1000,
        "intragenic": 500
    }

    possible_species_parameters = {
        "genome": None,
        "annotation": None,
        "targets": None,
        "chromosomes": "",
        "pickle_key": None,
        "pickle_file": "validation_genes.pickle",
        "species_name": None,
    }
    inputs, failed_trainings, input_length = ParsedInputs.parse(input_file, possible_general_parameters=possible_general_parameters, possible_species_parameters=possible_species_parameters, multiple_species_required_msr=True)
    inputs = inputs.replace_both()
    return inputs, failed_trainings, input_length


def main():
    args = parse_args()
    inputs, failed_trainings, input_length = parse_input_file(args.input)
    train_models(inputs, failed_trainings,  input_length)

if __name__ == "__main__":
    main()