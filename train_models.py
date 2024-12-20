import argparse
import os
from typing import Dict, List, Optional, Tuple
import pandas as pd
from utils import get_filename_from_path, get_time_stamp, one_hot_encode, make_absolute_path, load_annotation, load_annotation_msr, combine_files, result_summary
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
import sys


def extract_genes(genome: pd.DataFrame, annotation: pd.DataFrame, extragenic: int, intragenic: int, model_case, ignore_small_genes: bool,train_val_split: bool, tpms, target_chromosomes: Tuple[str, ...], for_prediction: bool = False) -> Dict[str, Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]]:

    extracted_seqs = {}
    expected_final_size = 2 * (extragenic + intragenic) + 20
    # tpms are absolutely necessary for training, but not for predictions, so can miss if data is for predictions
    if tpms is None and not for_prediction:
        raise ValueError(f"tpms have to be given if \"for_prediction\" is not set to True!")
    
    unpack_variables = ("specie", "chrom", "start", "end", "strand", "gene_id") if model_case.lower() == "msr" else ("chrom", "start", "end", "strand", "gene_id")

    for values in annotation.values:
        if len(unpack_variables) == 6:
            specie, chrom, start, end, strand, gene_id = values
        else:
            chrom, start, end, strand, gene_id = values

    #for chrom, start, end, strand, gene_id in annotation.values:#type:ignore
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


def extract_seq(genome, annot, tpm_targets, extragenic: int, intragenic: int, genes_picked, pickled_key, val_chromosome,
                model_case, ignore_small_genes, train_val_split, test_specie: Optional[str] = None):
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
    
    expected_final_size = 2*(extragenic + intragenic) + 20
    train_seqs, val_seqs, train_targets, val_targets = [], [], [], []
    
    if model_case.lower() == "msr":
        if test_specie is None:
            raise ValueError("test specie parameter necessary for msr training!")

        
        # do i even need this anymore? 
        args = parse_args() 
        data = pd.read_csv(args.input, sep=',', header=None,
                    dtype={0: str, 1: str, 2: str, 3: str, 4: str, 5: str, 6: str},
                    names=['specie','genome', 'gtf', 'tpm', 'output', 'chroms', 'p_key'])
        
        # concat all species to one fasta/gtf/tpm file
        p_keys = "_".join(data['p_key'].unique())
        input_filename = args.input.split('.')[0]
        
        # adjust file names to combined ones: 
        genome_path = make_absolute_path("genome", f"genome_{p_keys}.fa", start_file=__file__)     
        tpm_path = make_absolute_path("tpm_counts", f"tpm_{p_keys}_{input_filename}.csv", start_file=__file__)  # tpm_targets = f"tpm_{p_keys}.csv"
        annotation_path = make_absolute_path("gene_models", f"gtf_{p_keys}.csv", start_file=__file__)  
        genome = Fasta(filename=genome_path, as_raw=True, read_ahead=10000, sequence_always_upper=True)
        tpms = pd.read_csv(filepath_or_buffer=tpm_path, sep=',')
        tpms.set_index('gene_id', inplace=True)
        annot = load_annotation_msr(annotation_path)
        #print("Annotation data:")
        #print(annot.head())
        
        skipped_genes = [] 
        validation_genes = []
        for specie, chrom, start, end, strand, gene_id in annot.values:
                    #print(gene_id)
                    gene_size = end - start
                    extractable_downstream = intragenic if gene_size//2 > intragenic else gene_size//2
                    prom_start, prom_end = start - extragenic, start + extractable_downstream
                    term_start, term_end = end - extractable_downstream, end + extragenic

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

                    if gene_id not in tpms.index:
                        skipped_genes.append(gene_id)
                        continue
                        #if len(skipped_genes) > 5000:
                        #    print(f"Warning: More than 5000 gene IDs were skipped. Stopping processing.")
                        #    sys.exit()

                    #if test_specie['specie'].any() and specie in test_specie['specie'].values:
                    if specie in test_specie['specie'].values:
                        validation_genes.append(gene_id)
                        #print(f"val gene added {gene_id}")

                    # val: one species, one chromosome 
                    if seq.shape[0] == expected_final_size:
                        if chrom == val_chromosome and gene_id in validation_genes:        #species not in train species
                            val_seqs.append(seq)
                            val_targets.append(tpms.loc[gene_id, 'target'])
                            #print(f"val_targets: {gene_id}")
                                
                                
                        # train: all species except one 
                        elif specie not in test_specie['specie'].values:
                            train_seqs.append(seq)
                            train_targets.append(tpms.loc[gene_id, 'target'])
                            #print(f"train_targets: {gene_id}")

        
        if skipped_genes:  # This checks if the set/list is not empty
            timestamp = get_time_stamp()
            filename = f'skipped_genes_MSR_{p_keys}_{timestamp}.txt'
            with open(filename, 'w') as skipped_genes_file:
                for gene in skipped_genes:
                    skipped_genes_file.write(f"{gene}\n")
            
            if len(skipped_genes) > 5000:
                print(f"Warning: {len(skipped_genes)} gene IDs were skipped. Please check that the gene name formats are identical in both the GTF and TPM files.")
                 
            else:
                print(f"Some gene IDs in the gtf file were not found in TPM counts. Skipped gene IDs have been written to {filename}.")

        
        
          
    if model_case.lower() in ["ssr", "ssc"]:

        genome_path = make_absolute_path("genome", genome, start_file=__file__)
        tpm_path = make_absolute_path("tpm_counts", tpm_targets, start_file=__file__)
        annotation_path = make_absolute_path("gene_models", annot, start_file=__file__)
        genome = Fasta(filename=genome_path, as_raw=True, read_ahead=10000, sequence_always_upper=True)
        tpms = pd.read_csv(filepath_or_buffer=tpm_path, sep=',')
        tpms.set_index('gene_id', inplace=True)
        annot = load_annotation(annotation_path)

        # chromosome wise train-val splitting, OG code
        if train_val_split.lower() == "no":                             
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

        # 80 / 20 train-val splitting 
        current_val_size = 0
        current_train_size = 0
        
        if train_val_split.lower() == "yes":                             

            # random shuffle annot values to generate 3iterations of train val split 
            # the order of annot_values and pickle file decide which gene goes into training or validation
            shuffled_annot = annot.sample(frac=1, random_state=42).reset_index(drop=True)


            for chrom, start, end, strand, gene_id in shuffled_annot.values:
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

                total_sequences = seq.shape[0]
                target_val_size = int(total_sequences * 0.2)  # Target size for validation set (20%)
                target_train_size = total_sequences - target_val_size  # Target size for training set (80%)
                
                
                if seq.shape[0] == expected_final_size:                 
                                    
                    if current_val_size < target_val_size and gene_id in validation_genes:
                        val_seqs.append(seq)
                        val_targets.append(tpms.loc[gene_id, 'target'])
                        current_val_size += 1


                    elif current_train_size < target_train_size:
                        train_seqs.append(seq)
                        train_targets.append(tpms.loc[gene_id, 'target'])
                        current_train_size += 1 

        # check if desired 80/20 split is reached 
        if current_val_size < target_val_size:
            raise ValueError(f"Validation set is not 20%. Current size: {current_val_size} genes, "
                                f"Target size: {target_val_size} genes. Total genes in pickle file: {len(validation_genes)}. "
                                f"(Only genes from pickle file can be in the validation set.)")

    
    
    train_seqs, val_seqs = np.array(train_seqs), np.array(val_seqs)
    train_targets, val_targets = np.array(train_targets), np.array(val_targets)
    print(train_seqs.shape, val_seqs.shape)
    if train_seqs.size == 0 or val_seqs.size == 0:
        print("Error: Validation sequences or training sequences are empty. Stopping execution.")
        exit(1)

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
                   model_case, pickled_key, ignore_small_genes, train_val_split,  test_specie: Optional[str] = None):
    train_seqs, train_targets, val_seqs, val_targets = extract_seq(genome, annot, tpm_targets, upstream, downstream,
                                                                   genes_picked, pickled_key, val_chromosome,
                                                                   model_case, ignore_small_genes, train_val_split, test_specie)
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
                        This is a seven column csv file with entries: species, genome, gtf, tpm, output name,
                        number of chromosomes and pickle_key.""", required=True)
    parser.add_argument('--pickle', help="path to pickle file", required=True)
    parser.add_argument('--model_case', help="Can be SSC, SSR or MSR", required=True, choices=["msr", "ssr", "ssc", "both"])
    parser.add_argument('--ignore_small_genes', help="Ignore small genes, can be yes or no", required=True, choices=["yes", "no"])
    parser.add_argument('--train_val_split', help="Creates a training/validation dataset with 80%/20% of genes, can be yes or no", required=False, choices=["yes", "no"], default="no")


    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    model_case = args.model_case 
    #input_filename = args.input
    #input_filename = input_filename.split('.')[0]
    input_filename = args.input.split('.')[0]
    #print(input_filename)


    data = pd.read_csv(args.input, sep=',', header=None,
                    dtype={0: str, 1: str, 2: str, 3: str, 4: str, 5: str, 6: str},
                    names=['specie','genome', 'gtf', 'tpm', 'output', 'chroms', 'p_key'])
    ignore_small_genes = args.ignore_small_genes.lower() == "yes"
    print(data.head())

    if data.shape[1] != 7:
        raise Exception("Input file incorrect. Your input file must contain 7 columns and must be .csv")

    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    result_path = make_absolute_path("results", start_file=__file__)
    models_path = make_absolute_path("saved_models", start_file=__file__)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(models_path):
        os.makedirs(models_path)

    file_name = get_filename_from_path(__file__)
    failed_trainings, passed_trainings = [],[]
    
    # MSR training 

    if model_case.lower() == "msr":
        
        print(f'Multi species Training: ---------------------\n')

        p_keys = "_".join(data['p_key'].unique())
        # genreate concat files
        combine_files(data=data, file_type='tpm', file_extension='csv', output_dir='tpm_counts', file_key=p_keys, input_filename=input_filename)
        combine_files(data=data, file_type='genome', file_extension='fa', output_dir='genome', file_key=p_keys)
        combine_files(data=data, file_type='gtf', file_extension='csv', output_dir='gene_models', file_key=p_keys, load_func=load_annotation)
        
        # get concat files
        genome_path = make_absolute_path("genome", f"genome_{p_keys}.fa", start_file=__file__)     
        tpm_path = make_absolute_path("tpm_counts", f"tpm_{p_keys}_{input_filename}.csv", start_file=__file__)  # tpm_targets = f"tpm_{p_keys}.csv"
        annotation_path = make_absolute_path("gene_models", f"gtf_{p_keys}.csv", start_file=__file__)  
        genome = Fasta(filename=genome_path, as_raw=True, read_ahead=10000, sequence_always_upper=True)
        tpms = pd.read_csv(filepath_or_buffer=tpm_path, sep=',')
        tpms.set_index('gene_id', inplace=True)
        annot = load_annotation_msr(annotation_path)

        results_genome = []


        for specie in data['specie'].unique():                                                                     # use this 
            test_specie = data.copy()
            test_specie = test_specie[test_specie['specie'] == specie]
            train_specie = data.copy()
            train_specie = train_specie[train_specie['specie'] != specie]

            print(f'Training on species: {train_specie["specie"].unique()}')
            print(f'Testing on specie: {test_specie["specie"].unique()}')

            
            output_name = "_".join([sp[:3].lower() for sp in train_specie['specie'].unique()])
            print(f"Output name for training: {output_name}")

            #optional: other data directory 
            #chromosomes_file = test_specie['chroms'].values[0]
            #chromosomes = pd.read_csv(filepath_or_buffer=f'../../simon/projects/deepCRE_reimplemented/genome/{chromosomes_file}', header=None).values.ravel().tolist()
            #chromosomes = pd.read_csv(filepath_or_buffer=f'genome/{chromosomes_file}', header=None).values.ravel().tolist()
        

            # get chromosome names from annot data due to renaming of chromosomes 
            test_specie_name = test_specie['specie'].values[0]
            chromosomes = annot[annot['species'] == test_specie_name]['Chromosome'].unique().tolist()
            chromosomes = sorted(chromosomes, key=lambda x: int("".join(filter(str.isdigit, x))))
            #print("sorted Unique chromosomes for the test species:", chromosomes)
        
            # same as SSR training 

            for i, val_chrom in enumerate(chromosomes):
                try: 
                    print(f"Now: using chromosome {val_chrom} from {test_specie['specie'].values[0]} as validation chromosome")
                    results = train_deep_cre(genome=genome,
                                                    annot=annot,
                                                    tpm_targets=tpms,
                                                    upstream=1000,
                                                    downstream=500,
                                                    genes_picked=args.pickle,
                                                    val_chromosome=str(val_chrom),
                                                    output_name=output_name,
                                                    model_case=args.model_case,
                                                    pickled_key=None,
                                                    ignore_small_genes=args.ignore_small_genes,
                                                    train_val_split=args.train_val_split,
                                                    test_specie=test_specie) 

                    results_with_info = {
                        'loss': results[0],
                        'accuracy': results[1],
                        'auROC': results[2],
                        'auPR': results[3],
                        'test_specie': test_specie['specie'].values[0],
                        'chromosome': val_chrom
                    }
                    
                    results_genome.append(results_with_info)
                    print(f"Results for genome: {genome_path}, validation chromosome: {val_chrom}: {results_with_info}")
                                                

                    passed_trainings.append((output_name, i))
                except Exception as e:
                    print(e)
                    failed_trainings.append((output_name, i, e))

    results_genome = pd.DataFrame(results_genome, columns=['test_specie', 'chromosome','loss', 'accuracy', 'auROC', 'auPR'])
    save_file = make_absolute_path('results', f"{p_keys}_{input_filename}_{args.model_case}_{file_name}_{get_time_stamp()}.csv", start_file=__file__)
    results_genome.to_csv(path_or_buf=save_file, index=False)
    print(results_genome.head())
        
    result_summary(failed_trainings=failed_trainings, passed_trainings=passed_trainings, input_length=len(data), script=get_filename_from_path(__file__))

                
                    
            
        

    if model_case.lower() in ["ssr", "ssc"]:

        for specie, genome, gtf, tpm_counts, output_name, chromosomes_file, pickled_key in data.values:
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
                        results = train_deep_cre(genome=genome, annot=gtf, tpm_targets=tpm_counts, upstream=1000,
                                                 downstream=500, genes_picked=args.pickle, val_chromosome=str(val_chrom),
                                                 output_name=output_name, model_case=args.model_case, pickled_key=pickled_key,
                                                 ignore_small_genes=args.ignore_small_genes, train_val_split=args.train_val_split)
                        results_genome.append(results)
                        print(f"Results for genome: {genome}, chromosome: {val_chrom}: {results}")
                    
                
                # New random gene-based split
                if args.train_val_split.lower() == 'yes':
                    chromosomes=[1,2,3]
                    print(f"Using random 80/20 gene-based split for validation")
                    for i, val_chrom in enumerate(chromosomes):
                        results = train_deep_cre(
                                      genome=genome,
                                      annot=gtf,
                                      tpm_targets=tpm_counts,
                                      upstream=1000,
                                      downstream=500,
                                      genes_picked=args.pickle,
                                      val_chromosome=val_chrom,  # 3 iterations
                                      output_name=output_name,
                                      model_case=args.model_case,
                                      pickled_key=pickled_key,
                                      ignore_small_genes=args.ignore_small_genes,
                                      train_val_split=args.train_val_split
                                  )
                        
                        results_genome.append(results)
                        print(f"Results for genome: {genome}, iteration: {val_chrom}: {results}")
                    
                else: 
                    print(f"Invalid input for --train_val_split argument. Please enter 'yes' or 'no'.")
                    break 
                    
                    
                    
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