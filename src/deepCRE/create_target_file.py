import argparse
import os
import pandas as pd
import numpy as np
import os

from deepCRE.utils import make_absolute_path


def parse_args():
    parser = argparse.ArgumentParser(description="Convert RNA seq data into targets for training.")
    parser.add_argument("--input_path", "-i", help="path to the RNA seq data.", required=True)
    parser.add_argument("--output_path", "-o", help="path to output file.", default="")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    tpm_path = args.input_path
    if args.output_path:
        output_path = args.output_path
    else:
        base_name = os.path.splitext(os.path.basename(tpm_path))[0]
        output_path = make_absolute_path("tpm_counts", f"{base_name}_targets.csv", start_file=__file__)

    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Read the current TPM counts file
    tpm_counts = pd.read_csv(tpm_path)
    true_targets = []

    # Calculate targets based on logMaxTPM
    for log_count in tpm_counts['logMaxTPM'].values:
        if log_count <= np.percentile(tpm_counts['logMaxTPM'], 25):
            true_targets.append(0)
        elif log_count >= np.percentile(tpm_counts['logMaxTPM'], 75):
            true_targets.append(1)
        else:
            true_targets.append(2)

    tpm_counts['target'] = true_targets
    tpm_counts = tpm_counts[['gene_id', 'target']]
    tpm_counts.to_csv(path_or_buf=output_path, index=False)
    print(f"Processed {tpm_path}:")
    print(tpm_counts.head())  # Print the first few rows of the processed DataFrame

if __name__ == "__main__":
    main()
