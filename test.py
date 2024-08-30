import os
import re
import pandas as pd
import unittest
import pyranges as pr

from train_ssr_models import extract_genes
from utils import load_input_files, make_absolute_path, get_time_stamp
from deepcre_predict import find_newest_model_path, predict_other, predict_self

def compare_lists(list1, list2) -> bool:
    if len(list1) != len(list2):
        print(f"length of lists doesnt fit ({len(list1)}, {len(list2)})")
        return False
    
    list1 = sorted(list1)
    list2 = sorted(list2)
    for item1, item2 in zip(list1, list2):
        if item1 != item2:
            print(f"{item1} != {item2}")
            return False
    return True
    

def test_compare_old_new():
    new = pd.read_csv("results/predictions/arabidopsis_deepcre_predict_240820_172417.csv", header=0)
    old = pd.read_csv("results/predictions/arabidopsis_deepcre_predict_240819_173613.csv", header=0)
    if not len(old) == len(new):
        raise ValueError("legths of compared dataframes arent equal!")
    comparison = old.compare(new)
    if not comparison.empty:
        print(comparison)
        raise ValueError("DFs arent the same!")
    

def test_gene_dist():
    annotation_path = "gene_models/Arabidopsis_thaliana.TAIR10.52.gtf"
    annot = pr.read_gtf(f=annotation_path, as_df=True)
    annot = annot[annot['gene_biotype'] == 'protein_coding']
    annot = annot[annot['Feature'] == 'gene']
    annot = annot[['Chromosome', 'Start', 'End', 'Strand', 'gene_id']]
    full_list = []
    list_longer_1000 = []
    for chrom, start, end, strand, gene_id in annot.values:                                                     #type:ignore
        if chrom in ["2"]:
            full_list.append(gene_id)
            if end - start >= 1000:
                list_longer_1000.append(gene_id)
    ignore_short = pd.read_csv("results/predictions/arabidopsis_deepcre_predict_240820_172417.csv", header=0)
    include_short = pd.read_csv("results/predictions/arabidopsis_deepcre_predict_240820_180424.csv", header=0)
    short_list = ignore_short["genes"].values
    long_list = include_short["genes"].values
    print(len(list_longer_1000))
    if not compare_lists(short_list, list_longer_1000):
        raise ValueError("short lists arent equal!")
    if not compare_lists(long_list, full_list):
        raise ValueError("full lists arent equal!")


def test_regex():
    # regex_string = "^arabidopsis_(\\d+)_SSR_train_ssr_models_\\d+_\\d+\\.h5&"
    regex_string = "^arabidopsis_1_SSR_train_ssr_models_\d+_\d+\.h5$"                                           #type:ignore
    regex = re.compile(regex_string)
    match = regex.match("arabidopsis_1_SSR_train_ssr_models_240816_183905.h5")
    if match: 
        print(match.group(0))
        print(match.group(1))
    else:
        print("no match found")


def test_predict_other():
    data = pd.read_csv("example_predict_input.csv", sep=',', header=None,
                    dtype={0: str, 1: str, 2: str, 3: str, 4: int, 5: str},
                    names=['genome', 'gtf', 'tpm', 'output', 'counts'])

    for genome_file_name, annotation_file_name, tpm_counts_file_name, output_name, num_chromosomes in data.values:
        loaded_input_files = load_input_files(genome_file_name=genome_file_name, annotation_file_name=annotation_file_name, tpm_counts_file_name=tpm_counts_file_name)
        genome = loaded_input_files["genome"]
        annotation = loaded_input_files["annotation"]
        tpms = loaded_input_files["tpms"]
        extragenic = 1000
        intragenic = 500
        ignore_small_genes = "yes"
        extracted_genes = extract_genes(genome=genome, annotation=annotation, extragenic=extragenic, intragenic=intragenic, ignore_small_genes=ignore_small_genes, tpms=tpms, target_chromosomes=())
        results_dfs = []
        for chrom in range(1, num_chromosomes + 1):
            results, _ = predict_other(extragenic=extragenic, intragenic=intragenic, val_chromosome=str(chrom), output_name=output_name,
                                            model_case="SSR", extracted_genes=extracted_genes)
            results_dfs.append(results)
        result = pd.concat(results_dfs)
        only_preds = result.drop(["true_targets", "genes"], axis=1)
        result["pred_probs"] = only_preds.mean(axis=1)
        print(result.head())
        result.to_csv(f"results/predictions/arabidopsis_deepcre_predict_other_{get_time_stamp()}.csv")


def compare_predict_other_self():
    self_name = "arabidopsis_deepcre_predict_240820_175307.csv"
    other_name = "arabidopsis_deepcre_predict_other_240823_105835.csv"
    self_data = pd.read_csv(os.path.join("results", "predictions", self_name))
    other_data = pd.read_csv(os.path.join("results", "predictions", other_name), index_col=0)
    # chrom 1: 6078
    # chrom 2: 3541
    self_data_chrom_1 = self_data["pred_probs"].iloc[:6078]
    self_data_chrom_2 = self_data["pred_probs"].iloc[6078:]
    other_data_chrom_1 = other_data["arabidopsis_1_SSR_train_ssr_models_240816_183905.h5"].iloc[:6078]
    other_data_chrom_2 = other_data["arabidopsis_2_SSR_train_ssr_models_240816_184026.h5"].iloc[6078:]
    print((self_data_chrom_1 == other_data_chrom_1).all())
    self_data_chrom_2.index = other_data_chrom_2.index
    comparison_chrom_2 = (self_data_chrom_2 == other_data_chrom_2)
    print((comparison_chrom_2).all())


class TestDeepCRE(unittest.TestCase):
    
    def test_model_finding(self):
        results = find_newest_model_path(output_name="arabidopsis", model_case="SSR", model_path="test_folder/model_names")
        for key in results:
            self.assertTrue(key in ["1", "2"])
        path_to_models = make_absolute_path("test_folder", "model_names", start_file=__file__)
        self.assertEqual(results["1"], os.path.join(path_to_models, "arabidopsis_1_SSR_train_ssr_models_240822_103323.h5"))
        self.assertEqual(results["2"], os.path.join(path_to_models, "arabidopsis_2_SSR_train_ssr_models_240822_105523.h5"))


if __name__ == "__main__":
    # unittest.main()
    # test_regex()
    # test_predict_other()
    compare_predict_other_self()
    # test_gene_dist()