# deepCRE-core

This repo reimplements deepCRE using shap and modisco. It also upgrades to
tensorflow 2.10.0.

## Required directory structure

Within the current directory containing all python scripts, you should have
the following sub directories:

- genome : Contains the reference genomes you wish to train deepCRE on.
- gene_models: Contains the reference gtf files matching the genomes.
- tmp_counts: Contains target files. This file must have at least two columns named exactly: gene_id and target.

  The target column should be the already assigned true labels. This reimplementation allows
  you the freedom to assign true labels as you see fit.

## Examples

I have provided two example input files:

example_input.csv: For training. This is an example of how your input training file
should look. You can add more rows, where each row will be one species of interest.
This file contains primarily 6 columns:

genome| gtf | tpm_target | output_name |name of chromosomes file | pickle_key_id |

example_predict_input.csv: For predictions, modisco, shap runs. his file contains
the following columns:

genome| gtf | tpm_target | output_name |name of chromosomes file |

An example of the chromosomes file is uploaded in the genome subdirectory. This file must
always be in the genome subdirectory. Every new species requires its own chromosome file.
This file is a single column csv file that has the name of each chromosome on a separate
line.

## Usage

To train models use the following commands

```shell
python train_ssr_models.py --input example_input.csv --pickle validation_genes.pickle --model_case SSR --ignore_small_genes yes
```

```shell
python train_ssr_models.py --input example_input.csv --pickle validation_genes.pickle --model_case SSC --ignore_small_genes yes
```

To run predictions use

```shell
python deepcre_predict.py --input example_predict_input.csv --model_case SSR --ignore_small_genes yes
```

To compute shap importance scores. You also get a ..shap_meta.csv file for genes and predictions
This file is a one-to-one match with scores in the ..shap_score.h5 output

```shell
python deepcre_interprete.py --input example_predict_input.csv --model_case SSR --ignore_small_genes yes
```

To run modisco

```shell
python deepcre_motifs.py --input example_predict_input.csv --model_case SSR --ignore_small_genes yes
```
