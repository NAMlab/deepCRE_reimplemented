import os.path
from typing import Any
import pickle
from importlib import reload
import numpy as np
from tensorflow.keras.layers import Dropout, Dense, Input, Conv1D, Activation, MaxPool1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC
from pyfaidx import Fasta
import pyranges as pr
import pandas as pd
from sklearn.utils import shuffle
from deeplift.dinuc_shuffle import dinuc_shuffle
import shap
import h5py
import modisco


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
    for chrom, start, end, strand, gene_id in annot.values:
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
    x_train, y_train = shuffle(x_train, y_train, random_state=42)
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
    for chrom, start, end, strand, gene_id in annot.values:
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


# ----------------------------------------------------------------------------------#
# For Shap and MoDisco
# ----------------------------------------------------------------------------------#

# 1. Shap
def dinuc_shuffle_several_times(list_containing_input_modes_for_an_example, seed=1234):
    assert len(list_containing_input_modes_for_an_example) == 1
    onehot_seq = list_containing_input_modes_for_an_example[0]
    rng = np.random.RandomState(seed)
    to_return = np.array([dinuc_shuffle(onehot_seq, rng=rng) for i in range(50)])

    return [to_return]


def combine_mult_and_diffref(mult, orig_inp, bg_data):
    to_return = []
    for l in range(len(mult)):
        projected_hypothetical_contribs = np.zeros_like(bg_data[l]).astype("float")
        assert len(orig_inp[l].shape) == 2
        for i in range(orig_inp[l].shape[-1]):
            hypothetical_input = np.zeros_like(orig_inp[l]).astype("float")
            hypothetical_input[:, i] = 1.0
            hypothetical_difference_from_reference = (hypothetical_input[None, :, :] - bg_data[l])
            hypothetical_contribs = hypothetical_difference_from_reference * mult[l]
            projected_hypothetical_contribs[:, :, i] = np.sum(hypothetical_contribs, axis=-1)
        to_return.append(np.mean(projected_hypothetical_contribs, axis=0))
    return to_return


def compute_actual_hypothetical_scores(x, model):
    """
    This function computes the actual hypothetical scores given a model.

    :param x: onehot encodings of correctly predicted sequences
    :param model: loaded keras model used for predictions
    :return:
    """
    shap.explainers.deep.deep_tf.op_handlers["AddV2"] = shap.explainers.deep.deep_tf.passthrough
    shap.explainers.deep.deep_tf.op_handlers["FusedBatchNormV3"] = shap.explainers.deep.deep_tf.linearity_1d(0)
    dinuc_shuff_explainer = shap.DeepExplainer(
        (model.input, model.layers[-2].output[:, 0]),
        data=dinuc_shuffle_several_times,
        combine_mult_and_diffref=combine_mult_and_diffref)
    hypothetical_scores = dinuc_shuff_explainer.shap_values(x)
    actual_scores = hypothetical_scores * x
    return actual_scores, hypothetical_scores


def extract_scores(genome, annot, tpm_targets, upstream, downstream, n_chromosome, ignore_small_genes,
                   output_name, model_case):
    """
    This function performs predictions, extracts correct predictions and performs shap computations. This will be
    done iteratively per chromosome.

    :param genome: genome fasta file
    :param annot: gtf annotation file
    :param tpm_targets: targets file; must have a target column
    :param upstream: 1000
    :param downstream: 500
    :param n_chromosome: total number of chromosomes in the species
    :param ignore_small_genes: whether to ignore small genes
    :param output_name: prefix name used to create output files
    :param model_case: SSR, SSC or MSR
    :return: actual scores, hypothetical scores and one hot encodings of correct predictions across the entire genome
    """
    if not os.path.exists('results/shap'):
        os.makedirs('results/shap')
    shap_actual_scores, shap_hypothetical_scores, one_hots_seqs, gene_ids_seqs, preds_seqs = [], [], [], [], []
    chromosomes = pd.read_csv(filepath_or_buffer=f'genome/{n_chromosome}', header=None).values.ravel().tolist()
    for val_chrom in chromosomes:
        x, y, preds, gene_ids, model = predict(genome, annot, tpm_targets, upstream, downstream, str(val_chrom),
                                               ignore_small_genes, output_name, model_case)
        preds = preds > 0.5
        preds = preds.astype(int)
        correct_x, correct_y, correct_gene_ids = [], [], []
        for idx in range(x.shape[0]):
            if preds[idx] == y[idx]:
                correct_x.append(x[idx])
                correct_y.append(y[idx])
                correct_gene_ids.append(gene_ids[idx])

        correct_x = np.array(correct_x)

        # Compute scores
        print(f"Running shap for chromosome -----------------------------------------\n")
        print(f"Chromosome: {val_chrom}: Species: {output_name}\n")
        print(f"Running shap for chromosome -----------------------------------------\n")

        actual_scores, hypothetical_scores = compute_actual_hypothetical_scores(x=correct_x, model=model)
        shap_actual_scores.append(actual_scores)
        shap_hypothetical_scores.append(hypothetical_scores)
        one_hots_seqs.append(correct_x)
        gene_ids_seqs.extend(correct_gene_ids)
        preds_seqs.extend(correct_y)

    shap_actual_scores = np.concatenate(shap_actual_scores, axis=0)
    shap_hypothetical_scores = np.concatenate(shap_hypothetical_scores, axis=0)
    one_hots_seqs = np.concatenate(one_hots_seqs, axis=0)

    h = h5py.File(name=f'results/shap/{output_name}_shap_scores.h5', mode='w')
    h.create_dataset(name='contrib_scores', data=shap_actual_scores)
    pd.DataFrame({'gene_ids': gene_ids_seqs,
                  'preds': preds_seqs}).to_csv(path_or_buf=f'results/shap/{output_name}_shap_meta.csv', index=False)
    h.close()
    return shap_actual_scores, shap_hypothetical_scores, one_hots_seqs


# 2. MoDisco
def modisco_run(contribution_scores, hypothetical_scores, one_hots, output_name):
    if not os.path.exists('results/modisco'):
        os.mkdir('results/modisco')

    save_file = f"results/modisco/{output_name}_modisco.hdf5"
    os.system(f'rm -rf {save_file}')

    print('contributions', contribution_scores.shape)
    print('hypothetical contributions', hypothetical_scores.shape)
    print('correct predictions', one_hots.shape)
    # -----------------------Running modisco----------------------------------------------#

    null_per_pos_scores = modisco.coordproducers.LaplaceNullDist(num_to_samp=5000)
    tfmodisco_results = modisco.tfmodisco_workflow.workflow.TfModiscoWorkflow(
        # Slight modifications from the default settings
        sliding_window_size=15,
        flank_size=5,
        target_seqlet_fdr=0.15,
        seqlets_to_patterns_factory=modisco.tfmodisco_workflow.seqlets_to_patterns.TfModiscoSeqletsToPatternsFactory(
            trim_to_window_size=10,
            initial_flank_to_add=2,
            final_flank_to_add=0,
            final_min_cluster_size=30,
            n_cores=5)
    )(
        task_names=['task0'],
        contrib_scores={'task0': contribution_scores},
        hypothetical_contribs={'task0': hypothetical_scores},
        one_hot=one_hots,
        null_per_pos_scores=null_per_pos_scores)

    reload(modisco.util)
    grp = h5py.File(save_file, "w")
    tfmodisco_results.save_hdf5(grp)
    grp.close()
    print(f"Done with {output_name} Modisco run")


def generate_motifs(genome, annot, tpm_targets, upstream, downstream, ignore_small_genes,
                    output_name, model_case, n_chromosomes):

    actual_scores, hypothetical_scores, one_hots = extract_scores(genome=genome, annot=annot,
                                                                  tpm_targets=tpm_targets,
                                                                  upstream=upstream, downstream=downstream,
                                                                  n_chromosome=n_chromosomes,
                                                                  ignore_small_genes=ignore_small_genes,
                                                                  output_name=output_name,
                                                                  model_case=model_case)

    print("Now running MoDisco --------------------------------------------------\n")
    print(f"Species: {output_name} \n")
    modisco_run(contribution_scores=actual_scores, hypothetical_scores=hypothetical_scores,
                one_hots=one_hots, output_name=output_name)
