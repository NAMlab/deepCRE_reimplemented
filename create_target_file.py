import pandas as pd
import numpy as np
from utils import make_absolute_path


def main():
    tpm_path = make_absolute_path("tpm_couns", "arabidopsis_counts.csv", start_file=__file__)
    output_path = make_absolute_path("tpm_couns", "arabidopsis_targets.csv", start_file=__file__)

    tpm_counts = pd.read_csv(tpm_path)
    true_targets = []

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
    print(tpm_counts.head())


if __name__ == "__main__":
    main()