import pandas as pd
import numpy as np
import os

def main():
    # Define the directory containing the input files and the output directory
    tpm_dir = "../../simon/projects/deepCRE_reimplemented/tpm_counts"
    output_dir = "tpm_counts"

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # List of exact filenames to process
    filenames = [
        "Atha-D0h-Mn0Q50.8_expr_class.csv",
        "Anem-D0h-Mn0Q50.8_expr_class.csv",
        "Ahal-D0h-Mn0Q50.8_expr_class.csv",
        "Alyr-D0h-Mn0Q50.8_expr_class.csv"
    ]

    # Iterate through the specified filenames
    for filename in filenames:
        tpm_path = os.path.join(tpm_dir, filename)
        output_path = os.path.join(output_dir, f"{filename.split('.')[0]}_targets.csv")  # Create output filename

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

        # Add the target column to the DataFrame
        tpm_counts['target'] = true_targets
        # Keep only the 'gene_id' and 'target' columns
        tpm_counts = tpm_counts[['gene_id', 'target']]
        # Save the modified DataFrame to a CSV file
        tpm_counts.to_csv(path_or_buf=output_path, index=False)
        print(f"Processed {filename}:")
        print(tpm_counts.head())  # Print the first few rows of the processed DataFrame

if __name__ == "__main__":
    main()
