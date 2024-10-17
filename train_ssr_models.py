import argparse
import os
from typing import Dict, List, Tuple
import pandas as pd
from utils import get_filename_from_path, get_time_stamp, one_hot_encode, make_absolute_path, load_annotation, result_summary
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


def extract_genes(genome: pd.DataFrame, annotation: pd.DataFrame, extragenic: int, intragenic: int, ignore_small_genes: bool, tpms, target_chromosomes: Tuple[str, ...], for_prediction: bool = False) -> Dict[str, Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]]:
    extracted_seqs = {}
    expected_final_size = 2 * (extragenic + intragenic) + 20
    # tpms are absolutely necessary for training, but not for predictions, so can miss if data is for predictions
    if tpms is None and not for_prediction:
        raise ValueError(f"tpms have to be given if \"for_prediction\" is not set to True!")
    for chrom, start, end, strand, gene_id in annotation.values:#type:ignore
        # skip all chromosomes that are not in the target chromosomes. Empty tuple () means, that all chromosomes should be extracted
        if target_chromosomes != () and chrom not in target_chromosomes:
            continue

        gene_size = end - start
        extractable_downstream = intragenic if gene_size // 2 > intragenic else gene_size // 2
        prom_start, prom_end = start - extragenic, start + extractable_downstream
        term_start, term_end = end - extractable_downstream, end + extragenic

        promoter = one_hot_encode(genome[chrom][prom_start:prom_end])
        terminator = one_hot_encode(genome[chrom][term_start:term_end])
        extracted_size = promoter.shape[0] + terminator.shape[0]
        central_pad_size = expected_final_size - extracted_size

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

        if seq.shape[0] == expected_final_size:
            extracted_tuple = extracted_seqs.get(chrom, ())
            if extracted_tuple == ():
                x, y, gene_ids = [], [], []
            else:
                x = extracted_tuple[0]
                y = extracted_tuple[1]
                gene_ids = extracted_tuple[2]
            x.append(seq)
            # tpms check for for_prediction happened earlier
            if tpms is None:
                y.append("NA")
            else:
                y.append(tpms.loc[gene_id, 'target'])
            gene_ids.append(gene_id)
            extracted_seqs[chrom] = (x, y, gene_ids)

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


def extract_seq(genome, annot, tpm_targets, extragenic: int, intragenic: int, genes_picked, pickled_key, val_chromosome,
                model_case, ignore_small_genes):
    """
     This function extract sequences from the genome. It implements a gene size aware padding
    :param genome: reference genome from Ensembl Plants database
    :param annot:  gtf file matching the reference genome
    :param tpm_targets: count file target true targets.
    :param extragenic: length of promoter and terminator
    :param intragenic: length of 5' and 3' UTR
    :param genes_picked: pickled file containing genes to filter into validation set. For gene family splitting
    :param val_chromosome: validation chromosome
    :param model_case: model type which can be SSC, SSR
    :param pickled_key: key to pickled file name
    :param ignore_small_genes: filter genes smaller than 1000 bp
    :return: [one_hot train set, one_hot val set, train targets, val targets]
    """
    genome_path = make_absolute_path("genome", genome, start_file=__file__)
    tpm_path = make_absolute_path("tpm_counts", tpm_targets, start_file=__file__)
    annotation_path = make_absolute_path("gene_models", annot, start_file=__file__)
    genome = Fasta(filename=genome_path, as_raw=True, read_ahead=10000, sequence_always_upper=True)
    tpms = pd.read_csv(filepath_or_buffer=tpm_path, sep=',')
    tpms.set_index('gene_id', inplace=True)
    annot = load_annotation(annotation_path)
    expected_final_size = 2*(extragenic + intragenic) + 20

    train_seqs, val_seqs, train_targets, val_targets = [], [], [], []
    for chrom, start, end, strand, gene_id in annot.values:#type:ignore
        gene_size = end - start
        extractable_downstream = intragenic if gene_size//2 > intragenic else gene_size//2
        prom_start, prom_end = start - extragenic, start + extractable_downstream
        term_start, term_end = end - extractable_downstream, end + extragenic

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
                        This is a six column csv file with entries: genome, gtf, tpm, output name,
                        number of chromosomes and pickle_key.""", required=True)
    parser.add_argument('--pickle', help="path to pickle file", required=True)
    parser.add_argument('--model_case', help="determines whether a model or a control is trained.", required=True, type=str, choices=["SSR", "SSC", "BOTH"])
    parser.add_argument('--ignore_small_genes', help="Ignore small genes, can be yes or no", required=True)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    data = pd.read_csv(args.input, sep=',', header=None,
                    dtype={0: str, 1: str, 2: str, 3: str, 4: str, 5: str},
                    names=['genome', 'gtf', 'tpm', 'output', 'chroms', 'p_key'])
    ignore_small_genes = args.ignore_small_genes.lower() == "yes"
    model_cases = ["SSR", "SSC"] if args.model_case == "BOTH" else [args.model_case]
    print(data.head())

    if data.shape[1] != 6:
        raise Exception("Input file incorrect. Your input file must contain 6 columns and must be .csv")

    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    result_path = make_absolute_path("results", start_file=__file__)
    models_path = make_absolute_path("saved_models", start_file=__file__)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(models_path):
        os.makedirs(models_path)

    file_name = get_filename_from_path(__file__)
    failed_trainings, passed_trainings = [],[]
    for model_case in model_cases:
        for genome, gtf, tpm_counts, output_name, chromosomes_file, pickled_key in data.values:
            try:
                print(f'Training on genome: ---------------------\n')
                print(genome)
                print('\n------------------------------\n')
                results_genome = []
                chromosomes = pd.read_csv(filepath_or_buffer=f'genome/{chromosomes_file}', header=None).values.ravel().tolist()
                for i, val_chrom in enumerate(chromosomes):
                    print(f"Using chromosome {val_chrom} as validation chromosome")
                    results = train_deep_cre(genome=genome,
                                            annot=gtf,
                                            tpm_targets=tpm_counts,
                                            upstream=1000,
                                            downstream=500,
                                            genes_picked=args.pickle,
                                            val_chromosome=str(val_chrom),
                                            output_name=output_name,
                                            model_case=model_case,
                                            pickled_key=pickled_key,
                                            ignore_small_genes=ignore_small_genes)
                    results_genome.append(results)
                    print(f"Results for genome: {genome}, chromosome: {val_chrom}: {results}")
                results_genome = pd.DataFrame(results_genome, columns=['loss', 'accuracy', 'auROC', 'auPR'])
                save_file = make_absolute_path('results', f"{output_name}_{args.model_case}_{file_name}_{get_time_stamp()}.csv", start_file=__file__)
                results_genome.to_csv(path_or_buf=save_file, index=False)
                print(results_genome.head())

                passed_trainings.append((output_name, i))
            except Exception as e:
                print(e)
                failed_trainings.append((output_name, i, e))

    result_summary(failed_trainings=failed_trainings, passed_trainings=passed_trainings, input_length=len(data), script=get_filename_from_path(__file__))

if __name__ == "__main__":
    main()