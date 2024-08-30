import datetime
import os.path
from typing import Any, Dict
import pickle
from importlib import reload
import numpy as np
from tensorflow.keras.layers import Dropout, Dense, Input, Conv1D, Activation, MaxPool1D, Flatten#type:ignore
from tensorflow.keras.optimizers import Adam#type:ignore
from tensorflow.keras import Model#type:ignore
from tensorflow.keras.models import load_model#type:ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau#type:ignore
from tensorflow.keras.metrics import AUC#type:ignore
from pyfaidx import Fasta
import pyranges as pr
import pandas as pd
from sklearn.utils import shuffle
from deeplift.dinuc_shuffle import dinuc_shuffle # type: ignore
import shap# type: ignore
import h5py# type: ignore
import modisco# type: ignore


def one_hot_encode(sequence: str,
                   alphabet: str = 'ACGT',
                   neutral_alphabet: str = 'N',
                   neutral_value: Any = 0,
                   dtype=np.float32) -> np.ndarray:
    """
    One-hot encode sequence. This function expects a nucleic acid sequences with 4 bases: ACGT.
    It also assumes that unknown nucleotides within the sequence are N's.
    :param sequence: nucleotide sequence

    :return: 4 x L one-hot encoded matrix
    """
    def to_uint8(string):
        return np.frombuffer(string.encode('ascii'), dtype=np.uint8)
    hash_table = np.zeros((np.iinfo(np.uint8).max, len(alphabet)), dtype=dtype)
    hash_table[to_uint8(alphabet)] = np.eye(len(alphabet), dtype=dtype)
    hash_table[to_uint8(neutral_alphabet)] = neutral_value
    hash_table = hash_table.astype(dtype)
    return hash_table[to_uint8(sequence)]


def deep_cre(x_train, y_train, x_val, y_val, output_name, model_case, chrom):
    """

    :param x_train: onehot encoded train matrix
    :param y_train: true targets to x_train
    :param x_val: onehot encoded validation matrix
    :param y_val: target values to x_val
    :param output_name: the start of the output file name such as arabidopsis_leaf to create arabidopsis_leaf_output.csv
    :param model_case: model type which can be SSC, SSR
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

    model_chkpt = ModelCheckpoint(filepath=f"saved_models/{model_case}_{output_name}_model_{chrom}.h5",
                                  save_best_only=True,
                                  verbose=1)
    early_stop = EarlyStopping(patience=10)
    reduce_lr = ReduceLROnPlateau(patience=5, factor=0.1)
    model.compile(loss='binary_crossentropy', optimizer=Adam(0.0001),
                  metrics=['accuracy', AUC(curve="ROC", name='auROC'), AUC(curve="PR", name='auPR')])
    model.fit(x_train, y_train, batch_size=64, epochs=100, validation_data=(x_val, y_val),
              callbacks=[early_stop, model_chkpt, reduce_lr])

    loaded_model = load_model(f"saved_models/{model_case}_{output_name}_model_{chrom}.h5")
    output = loaded_model.evaluate(x_val, y_val)
    return output


def extract_seq(genome, annot, tpm_targets, upstream, downstream, genes_picked, pickled_key, val_chromosome,
                model_case, ignore_small_genes):
    """
     This function extract sequences from the genome. It implements a gene size aware padding
    :param genome: reference genome from Ensembl Plants database
    :param annot:  gtf file matching the reference genome
    :param tpm_targets: count file target true targets.
    :param upstream: length of promoter and terminator
    :param downstream: length of 5' and 3' UTR
    :param genes_picked: pickled file containing genes to filter into validation set. For gene family splitting
    :param val_chromosome: validation chromosome
    :param model_case: model type which can be SSC, SSR
    :param pickled_key: key to pickled file name
    :param ignore_small_genes: filter genes smaller than 1000 bp
    :return: [one_hot train set, one_hot val set, train targets, val targets]
    """
    genome = Fasta(filename=f"genome/{genome}", as_raw=True, read_ahead=10000, sequence_always_upper=True)
    tpms = pd.read_csv(filepath_or_buffer=f"tpm_counts/{tpm_targets}", sep=',')
    tpms.set_index('gene_id', inplace=True)
    print(tpms.head())
    if annot.endswith('.gtf'):
        annot = pr.read_gtf(f=f"gene_models/{annot}", as_df=True)
        annot = annot[annot['gene_biotype'] == 'protein_coding']
        annot = annot[annot['Feature'] == 'gene']
        annot = annot[['Chromosome', 'Start', 'End', 'Strand', 'gene_id']]
    else:
        annot = pr.read_gff3(f'gene_models/{annot}', as_df=True)
        annot = annot[annot['Feature'] == 'gene']
        annot = annot[['Chromosome', 'Start', 'End', 'Strand', 'ID']]
    expected_final_size = 2*(upstream + downstream) + 20

    train_seqs, val_seqs, train_targets, val_targets = [], [], [], []
    for chrom, start, end, strand, gene_id in annot.values:#type:ignore
        gene_size = end - start
        extractable_downstream = downstream if gene_size//2 > downstream else gene_size//2
        prom_start, prom_end = start - upstream, start + extractable_downstream
        term_start, term_end = end - extractable_downstream, end + upstream

        promoter = one_hot_encode(genome[chrom][prom_start:prom_end])
        terminator = one_hot_encode(genome[chrom][term_start:term_end])
        extracted_size = promoter.shape[0] + terminator.shape[0]
        central_pad_size = expected_final_size - extracted_size

        if model_case.lower() == "ssc" and chrom != val_chromosome:
            np.random.shuffle(promoter)
            np.random.shuffle(terminator)

        pad_size = 20 if ignore_small_genes.lower() == 'yes' else central_pad_size

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

        with open(genes_picked, 'rb') as handle:
            validation_genes = pickle.load(handle)
            validation_genes = validation_genes[pickled_key]
        if seq.shape[0] == expected_final_size:
            if chrom == val_chromosome:
                if gene_id in validation_genes:
                    val_seqs.append(seq)
                    val_targets.append(tpms.loc[gene_id, 'target'])
            else:
                train_seqs.append(seq)
                train_targets.append(tpms.loc[gene_id, 'target'])

    train_seqs, val_seqs = np.array(train_seqs), np.array(val_seqs)
    train_targets, val_targets = np.array(train_targets), np.array(val_targets)
    print(train_seqs.shape, val_seqs.shape)
    # Masking
    train_seqs[:, upstream:upstream + 3, :] = 0
    train_seqs[:, upstream + (downstream * 2) + 17:upstream + (downstream * 2) + 20, :] = 0
    val_seqs[:, upstream:upstream + 3, :] = 0
    val_seqs[:, upstream + (downstream * 2) + 17:upstream + (downstream * 2) + 20, :] = 0
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


def train_deep_cre(genome, annot, tpm_targets, upstream, downstream, genes_picked, val_chromosome, output_name,
                   model_case, pickled_key, ignore_small_genes):
    train_seqs, train_targets, val_seqs, val_targets = extract_seq(genome, annot, tpm_targets, upstream, downstream,
                                                                   genes_picked, pickled_key, val_chromosome,
                                                                   model_case, ignore_small_genes)
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


def predict(genome, annot, tpm_targets, upstream, downstream, val_chromosome, ignore_small_genes,
            output_name, model_case):
    genome = Fasta(filename=f"genome/{genome}", as_raw=True, read_ahead=10000, sequence_always_upper=True)
    tpms = pd.read_csv(filepath_or_buffer=f"tpm_counts/{tpm_targets}", sep=',')
    tpms.set_index('gene_id', inplace=True)
    if annot.endswith('.gtf'):
        annot = pr.read_gtf(f=f"gene_models/{annot}", as_df=True)
        annot = annot[annot['gene_biotype'] == 'protein_coding']
        annot = annot[annot['Feature'] == 'gene']
        annot = annot[['Chromosome', 'Start', 'End', 'Strand', 'gene_id']]
    else:
        annot = pr.read_gff3(f'gene_models/{annot}', as_df=True)
        annot = annot[annot['Feature'] == 'gene']
        annot = annot[['Chromosome', 'Start', 'End', 'Strand', 'ID']]
    expected_final_size = 2 * (upstream + downstream) + 20

    x, y, gene_ids = [], [], []
    for chrom, start, end, strand, gene_id in annot.values:#type:ignore
        gene_size = end - start
        extractable_downstream = downstream if gene_size // 2 > downstream else gene_size // 2
        prom_start, prom_end = start - upstream, start + extractable_downstream
        term_start, term_end = end - extractable_downstream, end + upstream

        promoter = one_hot_encode(genome[chrom][prom_start:prom_end])
        terminator = one_hot_encode(genome[chrom][term_start:term_end])
        extracted_size = promoter.shape[0] + terminator.shape[0]
        central_pad_size = expected_final_size - extracted_size

        pad_size = 20 if ignore_small_genes.lower() == 'yes' else central_pad_size

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

        if seq.shape[0] == expected_final_size:
            x.append(seq)
            y.append(tpms.loc[gene_id, 'target'])
            gene_ids.append(gene_id)

    x, y, gene_ids = np.array(x), np.array(y), np.array(gene_ids)

    # Masking
    x[:, upstream:upstream + 3, :] = 0
    x[:, upstream + (downstream * 2) + 17:upstream + (downstream * 2) + 20, :] = 0

    model = load_model(f"saved_models/{model_case}_{output_name}_model_{val_chromosome}.h5")
    pred_probs = model.predict(x).ravel()
    return x, y, pred_probs, gene_ids, model


def get_time_stamp() -> str:
    """creates a time stamp for the current time

    Returns:
        str: string in the format date_time
    """
    return datetime.datetime.now().strftime("%y%m%d_%H%M%S")


def get_filename_from_path(path: str) -> str:
    """takes a path and returns the name of the file it leads to

    Args:
        path (str): path to a file

    Returns:
        str: name of the file
    """
    if not os.path.isfile(path):
        raise ValueError("path must lead to a file!")
    file_name = os.path.splitext(os.path.basename(path))[0]
    return file_name



def make_absolute_path(*steps_on_path, start_file: str = "") -> str:
    """creates an absoulte path from a starting location

    Args:
        start_file (str, optional): file from which the path starts. Defaults to "".
        steps_on_path (str, optional): arbitrary number of folders with an optionally file at the end which will be appended to the start path.

    Returns:
        str: absolute version of the path
    """
    if start_file == "":
        start_file = __file__
    start_folder = os.path.dirname(os.path.abspath(start_file))
    result_path = os.path.join(start_folder, *steps_on_path)
    return result_path


def load_input_files(genome_file_name: str = "", annotation_file_name: str = "", tpm_counts_file_name: str = "") -> Dict[str, pd.DataFrame]:
    """loads input files and returns them in a Dict

    Args:
        genome_file_name (str, optional): file name of the genome, saved in the subfolder \"genome\". Defaults to "".
        annotation_file_name (str, optional): file name of the annotation, saved in the subfolder \"gene_models\". Defaults to "".
        tpm_counts_file_name (str, optional):  file name of the tpm counts, saved in the subfolder \"tpm_counts\". Defaults to "".

    Raises:
        ValueError: _description_

    Returns:
        Dict[str, pd.DataFrame]: _description_
    """
    if genome_file_name == "" and annotation_file_name == "" and tpm_counts_file_name == "":
        raise ValueError("at least one of the file names must be given!")
    results = {}

    if genome_file_name != "":
        genome_path = make_absolute_path("genome", genome_file_name, start_file=__file__)
        genome = Fasta(filename=genome_path, as_raw=True, read_ahead=10000, sequence_always_upper=True)
        results["genome"] = genome

    if annotation_file_name != "":
        annotation_path = make_absolute_path("gene_models", annotation_file_name, start_file=__file__)
        annotation = pr.read_gtf(f=annotation_path, as_df=True)
        annotation = annotation[annotation['gene_biotype'] == 'protein_coding']
        annotation = annotation[annotation['Feature'] == 'gene']
        annotation = annotation[['Chromosome', 'Start', 'End', 'Strand', 'gene_id']]
        # annot = annot[annot['Chromosome'] == val_chromosome]
        results["annotation"] = annotation

    if tpm_counts_file_name != "":
        tpm_path = make_absolute_path("tpm_counts", tpm_counts_file_name, start_file=__file__)
        tpms = pd.read_csv(filepath_or_buffer=tpm_path, sep=',')
        tpms.set_index('gene_id', inplace=True)
        results["tpms"] = tpms

    return results