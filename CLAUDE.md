# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

deepCRE trains CNNs that predict binary gene expression class (high/low) from gene *flanking regions*
(promoter + 5'UTR + [padding] + 3'UTR + terminator), then explains them with SHAP and TF-MoDISco.
Reimplementation of [Peleke et al. 2024](https://www.nature.com/articles/s41467-024-47744-0).
`readme.md` is the user manual and documents every JSON input parameter — read it before changing
input parsing or output naming.

## Environment

Python **3.8** with TensorFlow **2.10** (pinned in `pyproject.toml`: `requires-python = ">=3.8, <3.9"`).
Note the shell's default `python` may be a newer Anaconda Python — activate the project conda env first.
Install per `readme.md` (conda env, `pip install tensorflow==2.10.0`, `pip install -e .`, the
kundajelab/AvantiShri SHAP fork, then `./set_up_example.sh` to create `genome/`, `gene_models/`,
`tpm_counts/` and download the *Arabidopsis* demo data).

Dependencies are **not** declared in `pyproject.toml` (no `[project.dependencies]`) — the readme's
`pip install` list is the source of truth: seaborn, pyranges, deeplift, modisco, pyfaidx,
scikit-learn, imbalanced-learn, biopython, tqdm, pyyaml, bcbio-gff, h5py.

## Commands

All commands are run **from the repository root** (paths inside the code are resolved relative to
`src/deepCRE/`, but the tests use root-relative paths).

```bash
# preprocessing (plain CLI args)
python src/deepCRE/create_target_file.py -i src/deepCRE/tpm_counts/arabidopsis_leaf_counts.csv
python src/deepCRE/create_validation_genes.py -p <proteins.fa> -b <blast_out_fmt6> -k <pickle_key>

# core scripts (all take a single --input / -i JSON file)
python src/deepCRE/train_models.py        -i src/deepCRE/inputs/arabidopsis_training_demo.json
python src/deepCRE/deepcre_predict.py    -i src/deepCRE/inputs/arabidopsis_predict_interpret_extract_demo.json
python src/deepCRE/deepcre_crosspredict.py -i <config.json>
python src/deepCRE/deepcre_interpret.py  -i <config.json>
python src/deepCRE/deepcre_motifs.py     -i <config.json>

# standalone sequence extraction (CLI args, not JSON)
python src/deepCRE/extract_sequences.py -f <genome.fa> -a <annotation.gtf> -o out.fa
```

### Tests

Unit tests live in `test_folder/` and use `unittest`; they are run with pytest:

```bash
python -m pytest test_folder/test_gene_extraction.py test_folder/test_cross_predictions.py
python -m pytest test_folder/test_gene_extraction.py::TestExtractGene::test_extract_gene_plus_strand_adds_central_padding
```

Do **not** run bare `pytest` on `test_folder/`: `test.py` imports tensorflow and defines
module-level `test_*` functions that pytest will collect but which are actually manual scripts
requiring downloaded genomes and previously trained models.

Integration tests are driven by `test_folder/test.py`'s `input_integration_tests()` (invoked via
`python test_folder/test.py [--modisco]`), which sweeps directories of JSON configs and calls each
script's `parse_input_file` + runner with `test=True`. Its hardcoded folder paths
(`src/deepCRE/inputs/training`, …) are stale — the config directories actually live under
`src/deepCRE/inputs/tests/{training,prediction,cross_prediction,interpretation,motives}`. Fix the
paths (or pass the right ones) before relying on it.

`test=True` threads through to `deep_cre()`, where it shortens `EarlyStopping`/`ReduceLROnPlateau`
patience so integration runs finish quickly.

## Architecture

### Two-layer input parsing (`parsing.py`)

Every core script takes a JSON file containing a **list** of run dicts; each becomes a `RunInfo`
with two dicts:

- `general_info` — settings shared by the whole run (`output_name`, `model_case`, `extragenic`, …)
- `species_info` — a *list* of per-species dicts, one entry even in the single-species case

Each script declares its own `possible_general_parameters` / `possible_species_parameters` dicts
inside its `parse_input_file()`; these double as **the parameter whitelist and the defaults table**.
Adding a new config option means adding it there (and documenting it in `readme.md`).

Key parsing semantics, all in `RunInfo`:
- Keys present in *both* parameter dicts (e.g. `genome`, `targets`, `chromosomes`) are read from
  `general_info` and used as defaults for every species, then deleted from `general_info`.
  So after parsing, `general_info` and `species_info` are disjoint.
- Parameters whose default is `None` are *required*; parameters with a non-`None` default are optional.
- `chromosomes` and `prediction_models` accept either a list or a filename of a single-column
  headerless CSV (looked up in `genome/` and `saved_models/` respectively).
- `model_case: "both"` is expanded by `ParsedInputs.replace_both()` into two runs (SSR + SSC).
- `ModelCase` enum: `SSR` (single species), `SSC` (single species, shuffled sequences → no
  sequence context), `MSR` (multi-species, cross-*species* validation), `BOTH`.

Runs are independent: each script loops over `RunInfo`s, catches per-run exceptions into
`failed_runs`, and prints a tally via `utils.result_summary()`. Only `TerminationError`
(`train_models.py`) aborts the whole script.

### Path resolution (`utils.make_absolute_path`)

Nearly all paths are resolved **relative to `src/deepCRE/`**, via
`make_absolute_path(..., start_file=__file__)`. The convention used everywhere (`get_input_file_path`,
`load_input_files`) is: *if the given string is an existing file, use it as a path; otherwise treat it
as a bare filename inside the conventional subfolder*. The conventional subfolders under
`src/deepCRE/` are:

| folder | contents |
|---|---|
| `genome/` | FASTA genomes, chromosome-name CSVs |
| `gene_models/` | GTF/GFF3 annotations (and MSR combined `gtf_*.csv`) |
| `tpm_counts/` | expression/target CSVs (`gene_id`, `target`) |
| `saved_models/` | trained `.h5` models, model-name CSVs |
| `results/training`, `results/predictions`, `results/shap`, `results/modisco` | outputs |
| `inputs/` | JSON run configs |

All of these except `inputs/` and the `results/*` `.gitkeep`s are gitignored.

### Model discovery is filename-convention-based — be careful

`train_models.deep_cre()` writes checkpoints as
`{output_name}_{val_chromosome}_{model_case}_{basename(train_models.py)}_{timestamp}.h5`,
and `find_newest_model_path()` re-discovers them with the regex
`^{output_name}_(.+)_{model_case}_train_models_\d+_\d+\.h5$`, picking the lexicographically last
(= newest timestamp) per validation chromosome. Both sides use
`get_filename_from_path(__file__)` / the literal `train_models`, so **renaming `train_models.py`
silently breaks prediction and interpretation.** Same pattern for
`deepcre_interpret.find_newest_interpretation_results()`.

This is why downstream scripts take `training_output_name` + `model_case` (+ chromosomes/species)
rather than explicit model paths — the exception is `deepcre_crosspredict.py`, which takes an
explicit `prediction_models` list.

### Sequence extraction and the 3020-bp input

`extragenic=1000`, `intragenic=500` by default. A gene yields
`promoter(1000) + 5'-flank(500) + zeros(20) + 3'-flank(500) + terminator(1000)` = **3020 bp**,
one-hot encoded to `(3020, 4)`. Minus-strand genes are reverse-complemented. Start/stop codon
positions are zero-masked identically in `train_models.mask_sequences()` and in
`deepcre_predict.predict_self()` / `deepcre_crosspredict.predict_other()` — if you change the
layout, all three must change together.

Two parallel extraction implementations exist and should not be confused:
- `train_models.extract_gene` / `extract_genes_prediction` — the pipeline path; keyed by chromosome,
  applies `ignore_small_genes` (skip genes shorter than `2 * intragenic`, else centre-zero-pad) and
  filters against the targets table.
- `extract_sequences.py` — standalone CLI for dumping flanking regions as FASTA or one-hot arrays,
  with its own `CENTRAL_PADDING = 20`, GFF3/GTF attribute handling and genes-of-interest filtering.

### Validation-set construction

`validation_genes.pickle` (in `src/deepCRE/`) maps a `pickle_key` (`"ara"`, `"zea"`, `"sol"`, `"sor"`)
to genes with no homologs on other chromosomes, so chromosome-wise cross-validation doesn't leak
homologs between train and validation. Two modes, selected by `train_val_split`:
- `False` (default): leave-one-chromosome-out cross-validation over `chromosomes`.
- `True`: random gene-level split of size `validation_fraction`, restricted to pickle-listed genes.

MSR is the third variant: `utils.combine_fasta`/`combine_tpms`/`combine_annotations` concatenate
per-species files into `genome/genome_<species...>.fa`, `tpm_counts/tpm_<species...>.csv` and
`gene_models/gtf_<species...>.csv`, appending `_<species_name>` to every chromosome and gene ID so
they stay attributable; validation then holds out one whole species. These combined files are cached
— the combine functions return early if the file already exists, so **stale combined files silently
shadow changed inputs**. MSR annotations are read by `load_annotation_msr()` (space-separated CSV,
6 columns, species column first), not `load_annotation()`.

### Interpretation chain

`deepcre_interpret.py` computes SHAP contribution + hypothetical-contribution scores (via the
kundajelab SHAP fork's `DeepExplainer` with dinucleotide-shuffled references) for **correctly
predicted genes only**, hence the mandatory `targets` file. It writes an `.h5` with datasets
`contrib_scores` and `hypothetical_contrib_scores` of shape `(n_genes, 3020, 4)` plus a
`*_shap_meta.csv` whose row order matches the h5. `deepcre_motifs.py` consumes those scores (or
recomputes them when `force_interpretations` is true) and runs TF-MoDISco into `results/modisco/`.

## Conventions in this codebase

- Google-style docstrings with `Args:`/`Returns:`/`Raises:` on essentially every function — match this.
- Type hints throughout; `#type:ignore` comments are used where pyranges/keras stubs are wrong.
- All outputs carry a `utils.get_time_stamp()` suffix (`%y%m%d_%H%M%S`); nothing is overwritten,
  and "newest" is determined by lexicographic sort of these stamps.
- Untracked scratch files (`test.fa`, `output.fa`, `goi.json`, various JSON configs) accumulate in
  `src/deepCRE/`; don't assume everything there is part of the package.

