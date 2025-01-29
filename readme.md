# User Manual: DeepCRE

## Table of Contents

- [User Manual: DeepCRE](#user-manual-deepcre)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
    - [Overview](#overview)
  - [Installation and Setup](#installation-and-setup)
    - [Prerequisites](#prerequisites)
    - [Installation Steps](#installation-steps)
      - [simons notes](#simons-notes)
  - [Framework Overview](#framework-overview)
    - [Workflow Summary](#workflow-summary)
  - [Script Descriptions and Usage](#script-descriptions-and-usage)
    - [General Input File Formats](#general-input-file-formats)
    - [Data Preprocessing](#data-preprocessing)
      - [Create targets](#create-targets)
      - [Create Validation Genes](#create-validation-genes)
    - [Model Training](#model-training)
      - [Single Species Training Input Files](#single-species-training-input-files)
      - [Multiple Species Training Input Files](#multiple-species-training-input-files)
      - [Training Output Files](#training-output-files)
    - [Self Predictions](#self-predictions)
      - [Self Prediction Input Files](#self-prediction-input-files)
      - [Self Prediction Output Files](#self-prediction-output-files)
    - [Interpretation](#interpretation)
      - [Interpretation Input Files](#interpretation-input-files)
      - [Interpretation Usage Example](#interpretation-usage-example)
      - [Interpretation Output Files](#interpretation-output-files)
    - [Motif Extraction](#motif-extraction)
      - [Motif Extraction Input Files](#motif-extraction-input-files)
      - [Motif Extraction Usage Example](#motif-extraction-usage-example)
      - [Motif Extraction Output Files](#motif-extraction-output-files)
  - [Examples and Case Studies](#examples-and-case-studies)
    - [End-to-End Workflow Example](#end-to-end-workflow-example)
  - [Troubleshooting and FAQs](#troubleshooting-and-faqs)
  - [Contributing](#contributing)
    - [How to Report Issues](#how-to-report-issues)
    - [How to Contribute](#how-to-contribute)
  - [References](#references)
  - [License](#license)
  - [Glossary](#glossary)
  - [Acknowledgements](#acknowledgements)

## Introduction

### Overview

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

### Prerequisites

environment setup

### Installation Steps

#### simons notes

conda create -p <path> python=3.8

pip install tensorflow-gpu==2.10.0

pip install tensorflow==2.10.0

pip install seaborn pyranges deeplift modisco pyfaidx scikit-learn imbalanced-learn biopython tqdm

“install shap” (clone https://github.com/AvantiShri/shap.git, cd shap, pip install .)

"install deepCRE"

pip install pyyaml

conda install tensorflow-base=2.10.0 -c conda-forge

pip install bcbio-gff

chmod +x set_up_example.sh

./set_up_example.sh

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

### Workflow Summary

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

Two scripts for data preprocessing are included in the repository. The first one is `create_target_file.py`, which can take RNA-seq data as a csv file with named columns
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

no clue how this works actually.

### Model Training

#### Single Species Training Input Files

Training models is the core functionality of the deepCRE toolkit. Necessary for training are as
mentioned previously a genome, corresponding annotation as well as the targets and validations
genes, that were generated in the preprocessing steps.\
The exact location of these files as well as other parameters are set in a json configuration
file, that is necessary for each run of the script.
An example of such an config file can be found in `src/deepCRE/inputs/arabidopsis_training_demo.json`. An mentioned preciously, the file is a list of dictionaries, where each dictionary contains the parameters for a single run. The necessary parameters for the training script are:

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
chromosomes that are not mitochondrial or from chloroplasts. **Type: List of strings.**
- pickle_key: Key for the species used in the pickle file containing the validation genes.
**Type: string.**
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

### Self Predictions

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
for msr models. **Type: List of strings.**
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



### Interpretation

#### Interpretation Input Files

#### Interpretation Usage Example

#### Interpretation Output Files

### Motif Extraction

#### Motif Extraction Input Files

#### Motif Extraction Usage Example

#### Motif Extraction Output Files

## Examples and Case Studies

### End-to-End Workflow Example

## Troubleshooting and FAQs

## Contributing

### How to Report Issues

### How to Contribute

## References

## License

## Glossary

## Acknowledgements
