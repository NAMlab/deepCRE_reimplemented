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
      - [actually done](#actually-done)
  - [Framework Overview](#framework-overview)
    - [Workflow Summary](#workflow-summary)
  - [Script Descriptions and Usage](#script-descriptions-and-usage)
    - [General Input File Formats](#general-input-file-formats)
    - [Folder Structure](#folder-structure)
    - [Data Preprocessing](#data-preprocessing)
    - [Model Training](#model-training)
      - [Training Input Files](#training-input-files)
      - [Training Usage Example](#training-usage-example)
      - [Training Output Files](#training-output-files)
    - [Predictions](#predictions)
      - [Prediction Input Files](#prediction-input-files)
      - [Prediction Usage Example](#prediction-usage-example)
      - [Prediction Output Files](#prediction-output-files)
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

#### actually done

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

### Folder Structure

### Data Preprocessing

### Model Training

#### Training Input Files

#### Training Usage Example

#### Training Output Files

### Predictions

#### Prediction Input Files

#### Prediction Usage Example

#### Prediction Output Files

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
