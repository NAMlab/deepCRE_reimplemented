# User Manual: DeepCRE

## Table of Contents

- [User Manual: DeepCRE](#user-manual-deepcre)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Installation and Setup](#installation-and-setup)
  - [Framework Overview](#framework-overview)
  - [Script Descriptions and Usage](#script-descriptions-and-usage)
    - [General Input File Formats](#general-input-file-formats)
    - [Data Preprocessing](#data-preprocessing)
      - [Create targets](#create-targets)
      - [Create Validation Genes](#create-validation-genes)
    - [Model Training](#model-training)
      - [Single Species Training Input Files](#single-species-training-input-files)
      - [Multiple Species Training Input Files](#multiple-species-training-input-files)
      - [Training Output Files](#training-output-files)
    - [Prediction](#prediction)
      - [Self Prediction Input Files](#self-prediction-input-files)
      - [Self Prediction Output Files](#self-prediction-output-files)
      - [Cross Prediction Input Files](#cross-prediction-input-files)
      - [Cross Prediction Output Files](#cross-prediction-output-files)
    - [Interpretation](#interpretation)
      - [Interpretation Input Files](#interpretation-input-files)
      - [Interpretation Output Files](#interpretation-output-files)
    - [Motif Extraction](#motif-extraction)
      - [Motif Extraction Input Files](#motif-extraction-input-files)
      - [Motif Extraction Output Files](#motif-extraction-output-files)
  - [Demo Case / Quick Start](#demo-case--quick-start)

## Introduction

This project is the official implementation of the DeepCRE framework, based on the work described
in **Peleke et al. and Peleke et al.** DeepCRE is designed to predict gene activity in plants using
convolutional neural networks on the gene flanking regions. The framework provides an automated
training process with a standardized
model architecture and input format. To train a model for a new species or tissue, all thats
necessary are genomic sequences in a FASTA format, gene annotations as GTF or GFF3, and expression
data from RNA sequencing as CSV. DeepCRE supports
predictions on different species, generates importance scores to interpret the contributions
of specific inputs, and enables motif extraction to identify patterns associated with high or low
gene activity.

## Installation and Setup

To successfully use the DeepCRE framework, first you need to set up an environment with the necessary
packages installed. We recommend the use of conda / miniconda to manage the environment. The following
steps will guide you through the installation process. One caveat to mention is that the installation process was
validated on a Linux system. Using Linux is therefore recommended and the commands in the installation guide
are tailored to that system. Installations of [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
and [conda](https://educe-ubc.github.io/conda.html) are required.\
The DeepCRE framework is built using the TensorFlow library version 2.10. To improve performance, we recommend
installing the GPU version of TensorFlow. If you do not have a GPU available, the CPU version will
work as well. Start off by creating and activating a new conda environment with Python 3.8:

```bash
conda create -n [name] python=3.8
conda activate [name]
```

Next, install the TensorFlow library:

```bash
pip install tensorflow-gpu==2.10.0
pip install tensorflow==2.10.0
```

Install the remaining required packages:

```bash
pip install seaborn pyranges deeplift modisco pyfaidx scikit-learn imbalanced-learn biopython tqdm pyyaml bcbio-gff
```

Finally, install the TensorFlow base package from the conda-forge channel:

```bash
conda install tensorflow-base=2.10.0 -c conda-forge
```

The explanations / interpretation of the output as well as the motif extraction from the models relies on the shap
library. To install it, clone the repository from GitHub to a directory of your choice and install it using pip:

```bash
git clone https://github.com/AvantiShri/shap.git
cd shap
pip install .
```

To install the DeepCRE framework, clone the repository from GitHub to a directory of your choice:

```bash
git clone https://github.com/NAMlab/deepCRE_reimplemented.git
cd deepCRE_reimplemented
pip install -e .
```

To set up the folder structure for the use of deepCRE as well as download some example data, run the
`set_up_example.sh` script:

```bash
chmod +x set_up_example.sh
./set_up_example.sh
```

You're all set! The DeepCRE framework is now installed and ready to use. In the following chapters, we will
discuss the use of the different scripts and functionalities of the framework in detail. If you can't wait and
want to get started right away, example json configuration files are provided in the `src/deepCRE/inputs` folder
and their use is explained in the [examples chapter](#demo-case--quick-start).

## Framework Overview

The DeepCRE framework is organized into scripts for data
preprocessing, model training, and subsequent inference or
analysis.Data preprocessing is the first step, facilitated by
the scripts `create_target_file.py` and `create_validation_genes.py`. These
scripts handle the preparation of input data and require simple
command-line arguments to specify parameters. The resulting outputs
are essential for training the models.  

The core functionalities of DeepCRE are implemented in several
scripts, each serving a distinct purpose. The `train_models.py`
script is used to train DeepCRE models, which are the basis for downstream analysis. Once the models are trained,
additional scripts can be employed for specific tasks. The
`deepcre_crosspredict.py` script enables cross species prediction analysis
across different datasets, while `deepcre_predict.py` facilitates
predictions on the training data set. For interpreting model outputs,
`deepcre_interpret.py` calculates importance scores, and
`deepcre_motifs.py` extracts motifs associated with high or low
gene activity.  

While the preprocessing scripts take
straightforward command-line arguments, the training and
inference scripts require a JSON configuration file to specify the
necessary settings and parameters. More details on the input format
is given in the following chapters.

## Script Descriptions and Usage

### General Input File Formats

The primary input files include the genomes of the species under investigation, provided as FASTA files. Additionally,
annotations of the genomes, formatted as GTF or GFF3 files, are necessary. For tasks such as model training and motif
extraction, target classes derived from expression levels are also required. This means RNA-seq data or a similar
source of gene expression information must be provided.

Except for the readme file and the set_up_example.sh script, all relevant files and folders are
contained in `deepcre_reimplemented/src/deepCRE`. Important folders here are `gene_models`,
`genome` and `tpm_counts`. Genomes should be stored in the `genome` folder, annotations in the
`gene_model` folder and tpm_files and the corresponding targets in the `tpm_counts` folder.
Further important folders are `saved_models` containing the models generated by the training
process and `results` containing all other data that is generated by the scripts.

In addition to these files, the framework relies on a JSON configuration file for every run of its core scripts. The
JSON file serves as a flexible and organized way to define the parameters for each operation. Its general format is a
list, where each entry corresponds to a separate run (e.g., training, inference, or motif extraction). Each entry in
the list is a dictionary, and each dictionary contains the parameters for the run as keys, with the corresponding
settings specified as values. This structure allows users to specify multiple runs in a single file, facilitating
efficient and reproducible workflows.

For each of the scripts, the name of the genome file as well as the name of the corresponding annotation will be
specified for each run in the json config file. Further parameters depend on the different scripts and use cases.

### Data Preprocessing

#### Create targets

Two scripts for data preprocessing are included in the repository. The first one is `create_target_file.py`,
which can take RNA-seq data as a csv file with named columns
and create targets based on that. The file is run using up to three command line parameters:

- `--input_path` or `-i` is the path to the csv file containing the RNA-seq data.
- `--output_path` or `-o` is the path to the output location of the newly generated file.
This parameter does not need to be used, and files are by default saved in the `tpm_counts`
folder under the original filename with an added "_targets".
- `--target_colum` or `t` is the name of the column to be used to calculate the target classes on. By default a
column of the name "logMaxTPM" will be used.

The script expects a column with the name "gene_id" containing the IDs for the genes and another column with the name
specified by `--target_column`. All other columns in the input file will be ignored. The values in the target
column will be divided into an upper quartile, a lower quartile and the rest. All genes in the lower quartile will
be assigned to the low class with a target of 0. All genes in the upper quartile will be assigned the high expression
class and assigned a target value of 1. The remaining genes will get a target value of 2 and will be ignored during
training. The resulting output file will be a csv file with named columns. The 2 contained columns are "gene_id"
and "target".

#### Create Validation Genes

A second file to be used for data preprocessing is `create_validation_genes.py`. This script is used to generate a
pickled dictionary containing for each species a list of genes. Only genes that don't have homologs on other chromosomes
are included in the list, so that during the chromosome wise cross validation, no genes from the training set will also
be included in the validation set as a homolog. The sccript needs a fasta file containing all proteins of a species
and the results of a blast run where all proteins are basted against all proteins of the same species.

The script is run using the following command line parameters:

- `--proteins` or `-p` is the path to the fasta file containing all proteins of the species.
- `--blast_outputs` or `-b` is the path to the file containing the results of the blast run.
the results must be formatted in the blast output format 6.
- `--pickle_key_name` or `-k` is the name of the key under which the list of genes will be saved
in the dictionary in the pickle file.
- `--output` or `-o` is an optional parameter specifying the output path of the pickle file.
By default, the file will be saved in the `src/deepCRE` folder under the name
"validation_genes_<pickle_key_name>.pickle".

### Model Training

#### Single Species Training Input Files

Training models is the core functionality of the deepCRE toolkit and is implemenmted in the
script `src/deepCRE/train_models.py`. Necessary for training are as
mentioned previously a genome, corresponding annotation as well as the targets and validations
genes, that were generated in the preprocessing steps.\
The exact location of these files as well as other parameters are set in a json configuration
file, that is necessary for each run of the script.
An example of such an config file can be found in `src/deepCRE/inputs/arabidopsis_training_demo.json`.
An mentioned preciously, the file is a list of dictionaries, where each dictionary contains the parameters
for a single run. The necessary parameters for the training script are:

- genome: either the full path to the genome file or the name of the genome file in the genome
folder. **Type: string.**
- annotation: either the full path to the annotation file or the name of the annotation file in
the gene_models folder. **Type: string.**
- targets: either the full path to the targets file or the name of the targets file in the
tpm_counts folder. **Type: string.**
- output_name: basis for the name of the saved models and result summary. The output name will
additionally contain "train_models", the validation chromosome for the particular model, the model
case and a time stamp. **Type: string.**
- chromosomes: the names of the chromosomes to be used for validation. Should usually be all
chromosomes that are not mitochondrial or from chloroplasts. **Type: string or List of strings.**
- pickle_key: Key for the species used in the pickle file containing the validation genes. Can be provided directly
as a list or as a file name of a file in the `genome` folder containing the names of the chromosomes in a single
unnamed file of a csv. **Type: string.**
- model_case: can be one of four options. "ssr", "ssc", "msr" or "both" and determines the
type of the model that will be trained. **Type: string.**

Regarding the model case parameter, "ssr" will be the standard case, in which a model will be
trained on a single species. "ssc" trains a model on a single species, but the sequences for
training will be shuffled, so that the model can not learn any patterns. "both" will train a
ssr-model, as well as a ssc-model. "msr" will train a model on multiple species. The inputs
for msr training are somewhat different from the other cases. and will be described in more
detail in the following.

Other optional parameters are not necessary for each run, but can be used to further specify the
training process. These are:

- pickle_file: This option determines the name of the file containing the validation genes. The file will be
looked for in the `src/deepCRE` folder. For deviation from that, use full path instead of just
the file name. **Type: string**, default is "validation_genes.pickle".
- ignore_small_genes: This flag influences how the training data is generated. If set to True, genes
that are smaller than twice the length of the intragenic parameter will be ignored. This is done so that no duplication
of sequences occurs in the extraction process. If set to false, the sequences that are too short will be zero-padded
in the center. **Type: boolean**, default is true.
- extragenic: The number of base pairs to be extracted from the extragenic region, so before the transcription start
site and after the transcription end site. **Type: int**, default is 1000.
- intragenic: The number of base pairs to be extracted from the intragenic region, so between the transcription start
site and the transcription end site. **Type: int**, default is 500.
- train_val_split: Controls how the validation set is created from the provided genetic data. In the default case
(false) chromosome wise chross validation will be used. This means that for each chromosome, a model will be trained
on all other chromosomes and validated on the current one. If set to true, the validation set will be created by
randomly selecting a fraction of the genes. The fraction can be set using the validation_fraction parameter.
**Type: boolean**, default is false.
- validation_fraction: The fraction of genes to be used for validation if train_val_split is set to true. Only genes
in the validation gene file qualify to be used for validation. The number of genes that is necessary for the validation
set is calculated by multiplying the number of genes on the genome with the validation_fraction. Since in the extraction
process genes can be excluded due to their size or them missing in the target file, the fraction of genes in the
validation set compared to the actually used training set can be increased. **Type: float**, default is 0.2.

#### Multiple Species Training Input Files

The deepCRE framework doesnt only allow training on a single species, but also on multiple species. To accomodate for
this, the input format of the json configuration format needs to be extended. For each run, one new parameter is added
called "species_data". This parameter is a list of dictionaries, where each dictionary contains the parameters for a
single species. The parameters that need to be given inside the species_data dictionary are:

- species_name: The name of the species. **Type: string.**
- genome: either the full path to the genome file or the name of the genome file in the genome
folder. **Type: string.**
- annotation: either the full path to the annotation file or the name of the annotation file in
the gene_models folder. **Type: string.**
- targets: either the full path to the targets file or the name of the targets file in the
tpm_counts folder. **Type: string.**
- pickle_key: Key for the species used in the pickle file containing the validation genes.
**Type: string.**

The model case parameter still needs to be set to "msr" to indicate that multiple species are to be trained on.
Chromosomes are not needed to be specified, since in the MSR case, the validation will be executed in a chross
species validation manner. The other optional parameters are still available and can be used to further specify the
training process.

#### Training Output Files

The training process creates multiple output files. The most obvious and important one are the trained models. These
are saved in the `saved_models` folder and are named according to the output_name parameter in the json configuration
file. The models are saved in a h5 format and can be loaded and used for predictions or interpretation. The training
process also creates a summary file, that is saved in the `results/training` folder. The summary file is a csv
containing information about the training process in tabular fashion. For each of the models of the cross validation,
loss, accuracy, auROC (area under the receiver operating characteristic curve) and auPR (area under the precision recall
curve) are saved.  It can be found under `"<output_name>_train_models_<model_case>_<time_stamp>.csv"`.\
The last file that is created by the training process is the file containing all genes that were
skipped in the generation of the dataset, because they were too small or not present in the target file. This file is
called `"skipped_genes_<time_stamp>.csv"` and is saved in the `results/training` folder.

For the msr model case additionally concatenated input files are created. The fasta
and target files for the species are concatenated. To ensure that the chromosomes and genes can clearly be assigned to
the correct species, the species name will be appended to each chromosome and gene id. For the annotations, the format
is changed to a csv file, where the species name is contained as a column, and only genes are contained. The created
files are saved in their respective folders (`"genome_<species1>(_<species2> ... ).fa"` in the `genome` folder,
`"tpm_<species1>(_<species2> ... ).csv"` in the `tpm_counts` folder and `"gtf_<species1>(_<species2> ... ).csv"` in
the `gene_models` folder).

### Prediction

Having trained a model enables making predictions. The deepCRE framework provides two scripts for this purpose. The
`deepcre_predict.py` script is used to make predictions on the training data set, while the `deepcre_crosspredict.py`
script is used to make predictions on a different species. Both scripts require a json configuration file to specify
the necessary parameters.
The thing separating the cross predicion and self prediction is the fact, that the self prediction will use the
models from the chromosome level cross prediction or species level cross predicion in such a way, that each gene will
be predicted by a model, for which this gene was part of the validation set, and not the training set. The cross
prediction will use user specified models to predict all genes presented in a genome.

#### Self Prediction Input Files

The input files for the self prediction script are similar to the input files for the training script. The json
configuration file needs to contain the following parameters:

- genome: either the full path to the genome file or the name of the genome file in the genome
folder. For msr models, this needs to be the concatenated genome file that was generated during the training process
and saved in the `genome` folder. **Type: string.**
- annotation: either the full path to the annotation file or the name of the annotation file in
the gene_models folder. For msr models, this needs to be the concatenated annotation file that was generated during the
training process and saved in the `gene_model` folder. **Type: string.**
- training_output_name: The `output_name` parameter of the training json configuration file needs to be provided here
since the prediction script will look for the correct models to use in the `saved_models` folder. If multiple models
have been trainded with the same input parameters so that they only differ in the time stamp, the most recent models
will be picked. **Type: string.**
- model_case: same as the model case for the training script. **Type: string.**

To provide information on the cross validation, either the chromosomes or the species names also need to be provided,
depending on the model case:

- chromosomes: the names of the chromosomes to be used for validation. Should usually be all
chromosomes that are not mitochondrial or from chloroplasts. Required for ssr model case, will be ignored
for msr models. Can be provided directly as a list or as a file name of a file in the `genome` folder containing the
names of the chromosomes in a single unnamed file of a csv. **Type: string or List of strings.**
- species_data: a list of dictionaries, where each dictionary represents a species. Required for msr model case.
**Type: List of dictionaries.**
  - species_name: name of the species. Required for msr model case. **Type: string.**

More optional parameters are available to further specify the prediction process:

- targets: either the full path to the targets file or the name of the targets file in the
tpm_counts folder. If the file is given, the true class of the genes will be shown in the "true_targets" column of the
output file. Otherwise this column will bve filled with "NA". For msr models, this needs to be the concatenated target
file that was generated during the training process and saved in the `tpm_counts` folder. **Type: string.**
- ignore_small_genes: Will ignore genes based on the same logic as in the training process. **Type: boolean**, default is true.
- extragenic: The number of base pairs to be extracted from the extragenic region, so before the transcription start
site and after the transcription end site. Should be the same as during the training process. **Type: int**,
default is 1000.
- intragenic: The number of base pairs to be extracted from the intragenic region, so between the transcription start
site and the transcription end site. Should be the same as during the training process. **Type: int**, default is 500.

#### Self Prediction Output Files

The self prediction script creates csv output files containing the predictions for each gene containedin the specified
genome (minus small genes depending on ignore_small_genes). The columns are the gene id("genes"), the
model output as a value between 0 and 1 ("pred_probs") and the true target class of the gene ("true_targets"). The
model output can be converted to a binary prediction by setting a threshold at 0.5. All genes with a model output
above 0.5 will be predicted as high expression genes, all genes below the threshold will be predicted as
low expression genes. The files are saved in the `results/predictions` folder. For ssr models, the file is named
`"<training_output_name>_<model_case>_deepcre_predict_<time_stamp>.csv"`. For msr models, one output file is
generated for each species with the name
`"<training_output_name>_<species_name>_<model_case>_deepcre_predict_<time_stamp>.csv"`.

#### Cross Prediction Input Files

The input files for the cross prediction script are similar to the input files for the self prediction script. The json
configuration file needs to contain the following parameters:

- genome: either the full path to the genome file or the name of the genome file in the genome
folder. **Type: string.**
- annotation: either the full path to the annotation file or the name of the annotation file in
the gene_models folder. **Type: string.**
- prediction_models: A list of models to be used for the precitions. File name of the models to be used is enough
if the models are located in the `saved_models` folder. Otherwise use full paths. Can be provided directly as a list or
as a file name of a file in the `saved_models` folder containing the names of the models in a single unnamed file of a
csv. **Type: string List of strings.**

For the name of the output file, either a full path to a desired output file needs to be given, or a base name for the
automatic generation of an output name for the file to be placed in the `results/predictions` folder:

- output_base: The base name for the output file. **Type: string.**
- output_path: The full path to the output file. **Type: string.**

If both output_base and output_path are given, the output_path will be used. If none of the two is given, the script
will throw an error. The other optional parameters are similar to the self prediction script:

- targets: either the full path to the targets file or the name of the targets file in the
tpm_counts folder. If given, the output will contain the correct class in a seperate column. **Type: string.**
- chromosomes: the names of the chromosomes for which preidctions should be made. Can be provided directly as a list
or as a file name of a file in the `genome` folder containing the names of the chromosomes in a single unnamed file
of a csv. **Type: string or List of string**
- ignore_small_genes: Will ignore genes based on the same logic as in the training process. **Type: boolean**, default
is true.
- extragenic: The number of base pairs to be extracted from the extragenic region, so before the transcription start
site and after the transcription end site. Should be the same as during the training process. **Type: int**,
default is 1000.
- intragenic: The number of base pairs to be extracted from the intragenic region, so between the transcription start
site and the transcription end site. Should be the same as during the training process. **Type: int**, default is 500.

It is possible to use the msr models to predict singluar genomes and also ssr models to make predictions on the
concatenated genomes. What is important to watch out for here is that the genome, annotation and potentially the
target files must be properly fitting, so either all files are concatenated or all files are singular. When using the
chromosomes parameter, the names of the chromosomes must be the same as in the concatenated files, meaning that the
species name must be appended to the chromosome name.

#### Cross Prediction Output Files

The cross prediction script creates a single csv output file containing the predictions for each gene contained in the
specified genome (minus small genes depending on ignore_small_genes). The rows of the output file are the different
genes with their ID in column "genes". For each model in prediction_models a column is created containing the model output
as a value between 0 and 1.  The columns are named after the model file names. An additional column with the name
"pred_probs" contains the average of all model outputs for the given gene. The model outputs can be converted to binary
predictions analogously to the self prediction script. The true targets as potentially denoted in the targets file are
contained in the "true_targets" column. The file is either saved according to the output_path, or saved in the
`results/predictions` folder according to the output_base. In this case the file is named after one of the models in the
prediction_models: `"<model_name>_deepcre_crosspredict_<output_base>_<time_stamp>.csv"`.

### Interpretation

#### Interpretation Input Files

The inputs for the interpretation script (`src/deepCRE/deepcre_interpret.py`) are almost the same as for the self
[prediction script](#self-prediction-input-files). The only difference is, that for the interpretation the target
file needs to be provided, since explanations will only be calculated for correct predictions. This is because we
want to learnt about biological patterns, and not wrong patterns learned by the neural network. The json
configuration file therefore needs to contain the following parameters:

- genome
- annotation
- targets
- training_output_name
- model_case

And depending on the model case:

- chromosomes (for ssr models)
- species_data with species names (for msr models)

The optional parameters are the same as for the self [prediction script](#self-prediction-input-files):

- ignore_small_genes
- extragenic
- intragenic

#### Interpretation Output Files

The interpretation script creates multiple output files, which are located in the `results/shap`
folder. The first file is called `"<training_output_name>_deepcre_interpret_<time_stamp>.h5"` and contains
two datasets. The file can be read in using the h5py package in python in the following way:

```python
import h5py

h5_file = h5py.File("src/deepCRE/results/shap/arabidopsis_deepcre_interpret_241017_170839.h5")
shap_scores = h5_file["contrib_scores"]
hypothetical_scores = h5_file["hypothetical_contrib_scores"]
```

The dataset `contrib_scores` contains the shap scores for each gene in the genome. The dataset
`hypothetical_contrib_scores` contains the hypothetical shap scores for each gene in the genome. The hypothetical
scores represent what the shap scores would look like if another base was present at a given position. For a more
detailed explanation, refer to the [shap repository](https://github.com/kundajelab/shap) and the
[shap paper](https://proceedings.neurips.cc/paper_files/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf).
Both datasets have the shape (number of genes, number of base pairs, 4), with the default of the number of base pairs
being 3020, and 4 representing the four nucleic bases in a one hot encoded fashion. The entry at each position
represents the contribution of the base at that position to the prediction of the gene.\
The second file is called `"<training_output_name>_deepcre_interpret_<time_stamp>_shap_meta.csv"`
and contains two columns, namely "gene_ids" and "preds". The gene_ids contain the IDs of the genes and the preds
column contains the predictions of the model for the genes (which also are the true targets, since only correctly
predicted genes are used for the interpretation). The genes in the meta file have the same order as the genes in the
shap scores file, so that the shap scores can be assigned to the correct genes.

### Motif Extraction

#### Motif Extraction Input Files

The inputs for the motif extraction script (`src/deepCRE/deepcre_motifs.py`) are once again almost the same as for
the self [prediction script](#self-prediction-input-files). Just like for the interpretation, the target file needs
to be provided, since motifs and patterns should only be extracted from correctly calculated examples. The motifs
can be calculated using saved explanations from the interpretation script. In case that is not wanted, an additional
parameter can be set to calculate the explanations on the fly. The json configuration file therefore needs to contain
the following parameters:

- genome
- annotation
- targets
- training_output_name
- model_case

And depending on the model case:

- chromosomes (for ssr models)
- species_data with species names (for msr models)

The optional parameters that can be used for this script are:

- ignore_small_genes
- extragenic
- intragenic
- force_interpretations: If set to true, the explanations will be recalculated, even if explanations
for this model have already been calculated. Otherwise, explanations will only be calculated, if they are not
already present. **Type: boolean**, default is false.

#### Motif Extraction Output Files

The motif extraction script creates a single output file located in the `results/motifs` folder. The file is named
`"<training_output_name>_deepcre_motifs_<time_stamp>.hdf5"` and contains the motifs extracted from the explanations.
The file can be read in using the h5py package in python. The general structure of the file is explained in the
[modisco-lite repository](https://github.com/jmschrei/tfmodisco-lite/) and the computational basis in the
[technical note](https://arxiv.org/pdf/1811.00416).

## Demo Case / Quick Start

When setting up the folder structure using the `set_up_example.sh` script, example data for the arabidopsis thaliana
species is downloaded. The data is stored in the `genome`, `gene_models` and `tpm_counts` folders. The data can be
used to run a simple demo case of the capabilities of the deepCRE framework. The first steps will be preparing the
data for training.\
In the `tpm_counts` folder, a file called `arabidopsis_leaf_counts.csv` is stored. This file contains the expression
levels of the genes in the arabidopsis genome from leaf samples of arabidopsis plants. To create the target file, the
`create_target_file.py` script can be used. The script is run using the following command:

```bash
python src/deepCRE/create_target_file.py --input_path src/deepCRE/tpm_counts/arabidopsis_leaf_counts.csv
```

Since the column we are interested in is called "logMaxTPM", we don't need to specify the target_column parameter. The
output file will be saved in the `tpm_counts` folder under the name `arabidopsis_leaf_counts_targets.csv`.\

Next we need to create the validation genes file. To run our script for the generation of the validation genes, we need
the protein sequences of arabidopsis as well as the results of a blast run. The protein sequences are publicly available
and for a blast run, an examplary run could look like this:

```bash
wget https://ftp.ebi.ac.uk/ensemblgenomes/pub/release-52/plants/fasta/arabidopsis_thaliana/pep/Arabidopsis_thaliana.TAIR10.pep.all.fa.gz
gunzip Arabidopsis_thaliana.TAIR10.pep.all.fa.gz
makeblastdb -in Arabidopsis_thaliana.TAIR10.pep.all.fa -dbtype prot -title arabidopsis -parse_seqids -hash_index -out arabidopsis
blastp -db arabidopsis -query Arabidopsis_thaliana.TAIR10.pep.all.fa  -out Blast_ara_to_ara -outfmt 6
```

For more information on using blast, refer to the [blast documentation](https://www.ncbi.nlm.nih.gov/books/NBK279690/).
Alternative software like diamond can be used as well, as long as the output is in the blast output format 6.
`create_validation_genes.py` can then be run using the following command:

```bash
python src/deepCRE/create_validation_genes.py --proteins Arabidopsis_thaliana.TAIR10.pep.all.fa --blast_outputs Blast_ara_to_ara --pickle_key_name ara
```

The output file will be saved in the `src/deepCRE` folder under the name `validation_genes_ara.pickle`. Alternatively a
file name can be specified using the `-o` parameter. A file with the name `validation_genes.pickle` is also already
part of the repository and contains the validation genes for _Arabidopsis thaliana_, _Zea mays_, _Solanum lycopersicum_ and
_Sorghum bicolor_ under the pickle keys "ara", "zea", "sol" and "sor" respectively.\

With the genome, annotation, targets and validation genes prepared, the training process can be started. An example
json configuration file for the training process is provided in `src/deepCRE/inputs/arabidopsis_training_demo.json`.
The file contains the necessary parameters for the training process with no optional parameters. The file can be used
to train a model on the arabidopsis genome. The training process can be started using the following command:

```bash
python src/deepCRE/train_models.py --input src/deepCRE/inputs/arabidopsis_training_demo.json
```

The training process will create a summary file in the `src/deepCRE/results/training` folder and save the models
in the `src/deepCRE/saved_models` folder under the name `"arabidopsis_<1-5>_ssr_train_models_<time_stamp>.h5"`.
Based on the trained models predictions, interpretations and motif extractions can be made. A json configuration file
for all of these processes is provided in `src/deepCRE/inputs/arabidopsis_predict_interpret_extract_demo.json`.
The file contains again only the necessary parameters for the processes. The other scripts can be run using the
following commands:

```bash
python src/deepCRE/deepcre_predict.py --input src/deepCRE/inputs/arabidopsis_predict_interpret_extract_demo.json
python src/deepCRE/deepcre_interpret.py --input src/deepCRE/inputs/arabidopsis_predict_interpret_extract_demo.json
python src/deepCRE/deepcre_motifs.py --input src/deepCRE/inputs/arabidopsis_predict_interpret_extract_demo.json
```

Results will be stored in the `src/deepCRE/results` folder in the respective subfolders `"predictions"`, `"shap"` and
`"modisco"`.\

The example data can be used to get a first impression of the capabilities of the deepCRE framework. For more detailed
information on the input parameters and the use of the scripts, refer to the [script descriptions](#script-descriptions-and-usage).

If after reading the full documentation, questions remain, feel free to contact us under the
[contact information](mailto:g.schmitz@f-juelich.de).
