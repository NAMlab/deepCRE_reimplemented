import os
import pickle
from typing import Tuple
import pandas as pd
from Bio import SeqIO
import re
import argparse
from deepCRE.utils import make_absolute_path


def parse_args():
    parser = argparse.ArgumentParser(
                        prog='create validation genes',
                        description="This script can be used to create a list of validation genes for a given proteome.")
    parser.add_argument('--proteins', "-p", help="Path to Proteome file", required=True)
    parser.add_argument('--blast_outputs', "-b", help="Path to BLAST output file. Needs to be in outfmt 6", required=True)
    parser.add_argument('--pickle_key_name', "-k", help="Key for the pickle dictionary", required=True)
    parser.add_argument('--output', "-o", help="Path to an output file. By default will create a file in the src/deepCRE folder.", required=False, default="")

    args = parser.parse_args()
    return args


def get_save_path(output_path: str, pickle_key: str):
    if not output_path:
        pickle_file_path = make_absolute_path(f"validation_genes_{pickle_key}.pickle", start_file=__file__)

    # Remove any existing pickle file
    if os.path.exists(pickle_file_path):
        raise FileExistsError(f"File already exists at {pickle_file_path}. Please remove it or provide a different output path.")
    return pickle_file_path


def get_queries(description: str) -> Tuple[str, str]:
    if "gene:" in description:
        gene_query = "gene:"
    elif "gene=" in description:
        gene_query = "gene="
    else:
        gene_query = ""
    
    if "chromosome:" in description:
        chrom_query = "chromosome:"
    elif "seq_id=" in description:
        chrom_query = "seq_id="
    else:
        chrom_query = ""
    return gene_query, chrom_query


def main():
    args = parse_args()
    proteome_path = args.proteins
    blast_output_path = args.blast_outputs
    pickle_key_name = args.pickle_key_name # Key for the pickle dictionary
    output_path = args.output
    print(proteome_path, blast_output_path, pickle_key_name, output_path)

    pickle_file_path = get_save_path(output_path=output_path, pickle_key=pickle_key_name)

    # Dictionary to store validation genes
    validation_genes = {}

    if os.path.exists(blast_output_path):
        # Parse protein sequences and build a DataFrame
        info = []
        for rec in SeqIO.parse(proteome_path, 'fasta'):
            description = rec.description
            gene_query, chrom_query = get_queries(description)
            if not (gene_query and chrom_query):
                continue
            protein_id = description.split(' ')[0]  # Extract protein_id
            gene_match = re.search(f'{gene_query}([^ ]+)', description)
            chrom_match = re.search(f'{chrom_query}([^ ]+)', description)
            gene_id = gene_match.group(1) if gene_match else None
            chrom = chrom_match.group(1) if chrom_match else None
            if chrom is not None and chrom.count(":") == 4:
                chrom = chrom.split(":")[1]
            info.append([gene_id, protein_id, chrom])

        info = pd.DataFrame(info, columns=['gene_id', 'protein_id', 'chrom'])
        info.index = info.protein_id.tolist()  #type:ignore

        # Read BLAST output
        blast_out = pd.read_csv(
            blast_output_path,
            sep='\t',
            names=[
                'qseqid', 'sseqid', 'pident', 'length', 'mismatch',
                'gap_open', 'qstart', 'qend', 'sstart', 'send', 'evalue',
                'bitscore'
            ]
        )

        # Filter BLAST output for valid protein IDs
        valid_ids = [seq_id for seq_id in blast_out.qseqid.tolist() if seq_id in info.index]
        blast_out = blast_out[blast_out['qseqid'].isin(valid_ids)]

        if not blast_out.empty:
            # Map gene and chromosome information
            blast_out['qgene'] = info.loc[blast_out.qseqid.tolist(), 'gene_id'].values
            blast_out['sgene'] = info.loc[blast_out.sseqid.tolist(), 'gene_id'].values
            blast_out['qchrom'] = info.loc[blast_out.qseqid.tolist(), 'chrom'].values
            blast_out['schrom'] = info.loc[blast_out.sseqid.tolist(), 'chrom'].values

            # Apply filtering criteria
            blast_out = blast_out[blast_out['evalue'] < 0.001]
            blast_out = blast_out[blast_out['bitscore'] >= 50]
            blast_out = blast_out[~blast_out['schrom'].isin(['Pt', 'Mt'])]
            blast_out = blast_out[~blast_out['qchrom'].isin(['Pt', 'Mt'])]

            # Identify validation genes
            val_set = []
            for gene_grp in blast_out.groupby('qgene'):
                if len(gene_grp[1]['qchrom'].unique()) == 1:
                    val_set.append(gene_grp[0])

            # Save flat list of validation genes for this output
            validation_genes[pickle_key_name] = val_set
    else:
        print(f"BLAST output file not found at {blast_output_path}")

    # Save the dictionary with the flattened list of gene IDs
    with open(pickle_file_path, 'wb') as pickle_file:
        pickle.dump(validation_genes, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Validation genes saved to {pickle_file_path}")

if __name__ == "__main__":
    main()