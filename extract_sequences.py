import argparse
import os
from pyfaidx import Fasta
from BCBio import GFF
from Bio import SeqFeature, SeqRecord
import numpy as np
import pandas as pd
import math

from typing import Optional, Tuple, List

CENTRAL_PADDING = 20

def onehot(seq):
    code = {'A': [1, 0, 0, 0],
            'C': [0, 1, 0, 0],
            'G': [0, 0, 1, 0],
            'T': [0, 0, 0, 1],
            'N': [0, 0, 0, 0]}
    encoded = np.zeros(shape=(len(seq), 4))
    for i, nt in enumerate(seq):
        if nt in ['A', 'C', 'G', 'T']:
            encoded[i, :] = code[nt]
        else:
            encoded[i, :] = code['N']
    return encoded


def find_genes(annotation_path: str, gene_name_attribute: str, feature_type_filter: List[str]) -> pd.DataFrame:
    chromosomes, starts, ends, strands, gene_ids = [], [], [], [], []
    weird_strand_counter = 0
    with open(annotation_path, "r") as file:
        filter = dict(gff_type=feature_type_filter)
        rec: SeqRecord.SeqRecord
        for rec in GFF.parse(file, limit_info=filter):
            feat: SeqFeature.SeqFeature
            for feat in rec.features:
                # extracted_seq = feat.extract(fasta_obj[rec.id])
                strand = feat.location.strand #type:ignore
                if strand not in [-1, 1]:
                    # print(f"no proper strand available for {feat.id} on {rec.id}")
                    weird_strand_counter += 1
                    # nothing should be appended, if one of the columns cant be filled properly. 
                    continue
                strand = "+" if strand == 1 else "-"
                chromosomes.append(rec.id)
                starts.append(feat.location.start) #type:ignore
                ends.append(feat.location.end) #type:ignore
                strands.append(strand)
                # gene_id = feat.id.replace("gene:", "")  # Clean up gene_id
                gene_id = feat.qualifiers[gene_name_attribute][0] + "_" + feat.type
                gene_id = gene_id.replace("gene:", "")
                # print(f"{gene_name_attribute}: {gene_id}")
                gene_ids.append(gene_id)

    print(f"{weird_strand_counter} entries missing due to unclear strand (only \"+\" and \"-\" allowed).")
    return pd.DataFrame(data={
        "chromosome": chromosomes,
        "start": starts,
        "end": ends,
        "strand": strands,
        "gene_id": gene_ids
    })


def find_start_end(start: int, end: int, intragenic: int, extragenic: int, strand: str) -> Tuple[int, int, int, int, int]:
    # if gene length is less than 1000, the intragenic regions from TTS and TSS will overlap
    gene_length = abs(end - start)
    if gene_length < 2 * intragenic:
        # adjust length to be extracted so it does not overlap, and add padding to the center,
        # so that the extracted sequence maintains the same length and the position of the TSS
        # and TTS remain.
        longer_intra_gene = math.ceil(gene_length / 2)
        shorter_intra_gene = math.floor(gene_length / 2)
        additional_padding = 2 * intragenic - gene_length
    else:
        longer_intra_gene = shorter_intra_gene = intragenic
        additional_padding = 0

    if strand == '+':
        prom_start, prom_end = start - extragenic, start + longer_intra_gene
        term_start, term_end = end - shorter_intra_gene, end + extragenic

    else:
        prom_start, prom_end = end - longer_intra_gene, end + extragenic
        term_start, term_end = start - extragenic, start + shorter_intra_gene
    return prom_start, prom_end, term_start, term_end, additional_padding


def extract_seq(fasta: Fasta, genes: pd.DataFrame, intragenic: int = 500, extragenic: int = 1000) -> Tuple[List[np.ndarray], List[str]]:
    encoded_train_seqs, train_ids = [], []
    lists = encoded_train_seqs, train_ids
    for chrom, start, end, strand, gene_id in genes.values:
        vals = find_start_end(start=start, end=end, intragenic=intragenic, extragenic=extragenic, strand=strand)
        prom_start, prom_end, term_start, term_end, additional_padding = vals
        append_sequences(prom_start=prom_start, prom_end=prom_end, term_start=term_start, term_end=term_end,
                            central_padding=CENTRAL_PADDING, additional_padding=additional_padding, chrom=chrom,
                            gene_id=gene_id, lists=lists, fasta=fasta, intragenic=intragenic, extragenic=extragenic)

    return encoded_train_seqs, train_ids


def append_sequences(prom_start, prom_end, term_start, term_end, central_padding, additional_padding,
                        lists, chrom, gene_id, fasta, intragenic, extragenic) -> None:
    encoded_train_seqs, train_ids = lists
    if prom_start > 0 and term_start > 0:
        if prom_start < prom_end and term_start < term_end:
            direction = 1
        elif prom_start > prom_end and term_start > term_end:
            direction = -1
        encoded_seq = np.concatenate([onehot(fasta[chrom][prom_start:prom_end])[::direction, ::direction],
                                      np.zeros(shape=(central_padding + additional_padding, 4)),
                                      onehot(fasta[chrom][term_start:term_end])[::direction, ::direction]])
        if encoded_seq.shape[0] == 2 * (extragenic + intragenic) + central_padding:
            encoded_train_seqs.append(encoded_seq)
            train_ids.append(gene_id)


def extract_string(fasta_obj: Fasta, gene_df: pd.DataFrame, intragenic: int, extragenic: int, output_file: str, central_padding: int):
    genes_seqs = {}
    # Debug: Check number of genes being processed
    print(f"Number of genes to process: {len(gene_df)}")
    genes_too_close_to_sequence_edge_counter = 0
    for chrom, start, end, strand, gene_id in gene_df.values:
        # print(f"Processing gene {gene_id} on chromosome {chrom} from {start} to {end} (strand {strand})")  # Debugging info
        vals = find_start_end(start=start, end=end, intragenic=intragenic, extragenic=extragenic, strand=strand)
        prom_start, prom_end, term_start, term_end, additional_padding = vals
        if prom_start > 0 and term_start > 0:
            if prom_start < prom_end and term_start < term_end:
                direction = 1
            elif prom_start > prom_end and term_start > term_end:
                direction = -1
            else:
                raise  ValueError("Issue..")
            promoter = fasta_obj[chrom][prom_start:prom_end:direction]
            terminator = fasta_obj[chrom][term_start:term_end:direction]
            padding = (central_padding + additional_padding) * "N"
            if strand == "+":
                gene_flanks = "".join(promoter) + padding + "".join(terminator)
                header = f"{chrom}_{gene_id}:{start}-{end}"  # Normal order for positive strand
            else:
                gene_flanks = "".join(terminator) + padding + "".join(promoter)
                header = f"{chrom}_{gene_id}:{end}-{start}"  # Invert start and end
            genes_seqs[header] = gene_flanks
        else:
            genes_too_close_to_sequence_edge_counter += 1
    
    print(f"{genes_too_close_to_sequence_edge_counter} genes are too close to the edge of the sequence to extract a full length gene flanking region!")
    if genes_seqs:
        print(f"Writing {len(genes_seqs)} sequences to output file: {output_file}")
        with open(output_file, "w") as f:
            for header, sequence in genes_seqs.items():
                f.write(f">{header}\n{sequence}\n")
        print(f"Finished writing output to {output_file}")
    else:
        print("No sequences were extracted!")




def extract_gene_flanking_regions(fasta_path: str, annotation_path: str, output_path: str, extract_one_hot_flag: bool,
                                  extract_string_flag: bool, extragenic: int, intragenic: int, gene_name_attribute: str,
                                  feature_type_filter: List[str]) -> Optional[Tuple[List, List]]:
    if not (extract_one_hot_flag or extract_string_flag):
        raise  ValueError("either extraction  as string or one hot coded np array is necessary!")
    fasta_obj = Fasta(fasta_path, as_raw=True, sequence_always_upper=True, read_ahead=10000)
    gene_df = find_genes(annotation_path=annotation_path, gene_name_attribute=gene_name_attribute, feature_type_filter=feature_type_filter)
    if extract_string_flag:
        extract_string(fasta_obj, gene_df, intragenic=intragenic, extragenic=extragenic, output_file=output_path, central_padding=CENTRAL_PADDING)
    if extract_one_hot_flag:
        sequences, gene_ids = extract_seq(fasta=fasta_obj, genes=gene_df, intragenic=intragenic, extragenic=extragenic)
        return sequences, gene_ids


def parse_args():
    parser = argparse.ArgumentParser(description="Extract gene flanking regions from FASTA and annotation files.")
    parser.add_argument("--fasta_path", "-f", type=str, required=True, help="path to the fasta file of the genome to extract the gene flanking regions from.")
    parser.add_argument("--annotation_path", "-a", type=str, required=True, help="path to the genome annotation.")
    parser.add_argument("--output_path", "-o", type=str, required=True, help="name / path of the file you want to save the extracted gene glanking regions to be saved at.")
    parser.add_argument("--intragenic", "-i", type=int, default=500, help="length of the sequence extracted inside the gene (downstream of TSS and upstream of TTS).")
    parser.add_argument("--extragenic", "-e", type=int, default=1000, help="length of the sequence extracted outside the gene (upstream of TSS and downstream of TTS).")
    parser.add_argument("--overwrite", "-ow", type=str, choices=["true", "false"], default="false", help="allows to overwrite an existing file for the output.")
    parser.add_argument("--gene_name_attribute", "-g", type=str, default="gene_id", help="name of the attribute that contains the name of the gene. Can be found in the last column of both gff3 and gtf files.")
    parser.add_argument("--feature_type", "-ft", type=str, default="gene", help="type of features for which gene flanking reagions are supposed to be extracted. Will filter entries based on column 3 in gff3 (\"type\") and gff/gtf (\"feature\"). Multiple values can be separated by semicolons.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    feature_type_filter = args.feature_type.split(";")
    if os.path.isfile(args.output_path):
        if args.overwrite == "false":
            raise ValueError(f"file {args.output_path} already exists!")
    extract_gene_flanking_regions(fasta_path=args.fasta_path, annotation_path=args.annotation_path, output_path=args.output_path,
                                  extract_string_flag=True, extract_one_hot_flag=False, extragenic=args.extragenic,
                                  intragenic=args.intragenic, gene_name_attribute=args.gene_name_attribute, feature_type_filter=feature_type_filter)


if __name__ == "__main__":
    main()
    # extract_gene_flanking_regions(fasta_path="/home/gernot/Data/gene_stuff/01_Ae_spelt_3B.fasta",
    #                               annotation_path="/home/gernot/Data/gene_stuff/Final_transcriptome_stringTie.gtf",
    #                               output_path="test_sanaa_00.fa", extract_string_flag=True, extract_one_hot_flag=False, extragenic=1000, intragenic=500, gene_name_attribute="gene_id",
    #                               feature_type_filter=["transcript"])

