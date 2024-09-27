import itertools
from typing import Dict, List
import unittest
import pandas as pd
import os
import deepcre_crosspredict as cp


model_names = "arabidopsis_1_SSR_train_ssr_models_240822_103323.h5;arabidopsis_2_SSR_train_ssr_models_240822_105523.h5"
# model_names = "arabidopsis_1_SSR_train_ssr_models_240916_170010.h5;arabidopsis_2_SSR_train_ssr_models_240916_170112.h5"


class TestCrossPredictions(unittest.TestCase):

    def __init__(self):
        super(TestCrossPredictions, self).__init__()
        # TODO: comment this back in, only commented out to reduce number of test cases
        # self.set_up_test_cases()

    def set_up_test_cases(self):
        optional_cols = itertools.product([True, False], repeat=5)
        optional_cols = [list(cols) for cols in optional_cols]
        cols_to_use = [[True] * 4 for _ in range(len(optional_cols))]
        folder_path = os.path.join("test_folder", "test_cross", "test_optional_cols")
        paths = []
        for i in range(len(cols_to_use)):
            cols_to_use[i].extend(optional_cols[i])
            paths.append(self.create_input_dataframe(cols_to_use[i], folder_path=folder_path))
        
        all_there = pd.read_csv(paths[0])
        all_there["intragenic_extraction_length"] = ["asdf"]
        all_there["extragenic_extraction_length"] = ["asdf"]
        all_there["ignore_small_genes"] = ["asdf"]
        all_there["chromosome_selection"] = ["asdf"]
        all_there["target_classes"] = ["asdf"]
        
        all_there.to_csv(os.path.join(folder_path, "all_junk_fail.csv"), index=False)

        all_there["intragenic_extraction_length"] = [""]
        all_there["extragenic_extraction_length"] = [""]
        all_there["ignore_small_genes"] = [""]
        all_there["chromosome_selection"] = [""]
        all_there["target_classes"] = [""]
        
        all_there.to_csv(os.path.join(folder_path, "all_empty_pass.csv"), index=False)

    @staticmethod
    def create_input_dataframe(cols_to_use: List[bool], folder_path: str) -> str:
        assert len(cols_to_use) == 9
        columns = [
            ("genome", "Arabidopsis_thaliana.TAIR10.dna.toplevel.fa"),
            ("gene_model", "Arabidopsis_thaliana.TAIR10.52.gtf"),
            ("model_names", model_names),
            ("subject_species_name", "arabidopsis"),
            ("intragenic_extraction_length", 600),
            ("extragenic_extraction_length", 900),
            ("ignore_small_genes", "no"),
            ("chromosome_selection", "genome/arabidopsis_chroms.csv"),
            ("target_classes", "tpm_counts/arabidopsis_counts.csv"),
        ]
        input_dict = {
            columns[i][0]: [columns[i][1]] for i, include_column in enumerate(cols_to_use) if include_column
        }
        df = pd.DataFrame(input_dict)
        if not os.path.exists(os.path.abspath(folder_path)):
            os.makedirs(folder_path)
        name = "_".join([columns[i][0] for i, include_column in enumerate(cols_to_use) if include_column]) + "_pass.csv"
        path = os.path.join(folder_path, name)
        df.to_csv(path, index=False)
        return path


    def test_cross_pred_integration(self):
        necessary_columns = {
            "genome": ["Arabidopsis_thaliana.TAIR10.dna.toplevel.fa"],
            "gene_model": ["Arabidopsis_thaliana.TAIR10.52.gtf"],
            "model_names": [model_names],
            "subject_species_name": ["arabidopsis"]
        }
        input_path = self.create_test_input(necessary_columns, "base_case.csv")
        input_df = pd.read_csv(input_path)
        cp.run_cross_predictions(input_df)
        self.assertTrue(True)

    def test_input_cases(self):
        folder_path = os.path.join("test_folder", "test_cross", "test_optional_cols")
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if not os.path.isfile(file_path):
                continue
            input_df = pd.read_csv(file_path)
            if "_fail" in file_path:
                self.assertRaises(ValueError, cp.run_cross_predictions, input_df)
            else:
                cp.run_cross_predictions(input_df)
        self.assertTrue(True)


    @staticmethod
    def create_test_input(input_dict: Dict, file_name: str) -> str:
        test_cross_pred_folder = os.path.join("test_folder", "test_cross")
        test_cross_pred_folder = os.path.abspath(test_cross_pred_folder)
        save_path = os.path.join(test_cross_pred_folder, file_name)
        if not os.path.isfile(save_path):
            df = pd.DataFrame(input_dict)
            if not os.path.exists(test_cross_pred_folder):
                os.makedirs(test_cross_pred_folder)
            df.to_csv(save_path, index=False)
        return save_path

    def create_check_input_test_cases(self) -> Dict[str, str]:
        test_input_dicts = []
        test_input_dicts.append(({
            "genome": ["Arabidopsis_thaliana.TAIR10.dna.toplevel.fa"],
            "gene_model": ["Arabidopsis_thaliana.TAIR10.52.gtf"],
            "model_names": [model_names],
            "subject_species_name": ["arabidopsis"]
        }, "base_case.csv"))
        test_input_dicts.append(({
            "genome": ["Arabidopsis_thaliana.TAIR10.dna.toplevel.fa"],
            "model_names": [model_names],
            "subject_species_name": ["arabidopsis"]
        }, "one_column_missing.csv"))
        test_input_dicts.append(({
            "genome": ["Arabidopsis_thaliana.TAIR10.dna.toplevel.fa"],
        }, "one_column_only.csv"))
        test_input_dicts.append(({
            "genome": ["Arabidopsis_thaliana.TAIR10.dna.toplevel.fa"],
            "gene_model": ["Arabidopsis_thaliana.TAIR10.52.gtf"],
            "model_names": [model_names],
            "random_addon_col": [model_names],
            "subject_species_name": ["arabidopsis"]
        }, "addidtional_unused_column.csv"))
        file_paths = {file_name: self.create_test_input(input_dict, file_name) for input_dict, file_name in test_input_dicts}
        return file_paths

    def test_check_input(self):
        paths_dict = self.create_check_input_test_cases()
        paths_dict = {file_name: pd.read_csv(path) for file_name, path in paths_dict.items()}
        self.assertRaises(ValueError, cp.check_input, paths_dict["one_column_missing.csv"])
        self.assertRaises(ValueError, cp.check_input, paths_dict["one_column_only.csv"])
        self.assertRaises(ValueError, cp.check_input, pd.DataFrame())
        self.assertTrue(cp.check_input(paths_dict["addidtional_unused_column.csv"]))
        self.assertTrue(cp.check_input(paths_dict["base_case.csv"]))
        


if __name__ == "__main__":
    # unittest.main()
    stupid_obejct = TestCrossPredictions()
    # stupid_obejct.test_cross_pred_integration()
    # stupid_obejct.test_check_input()
    stupid_obejct.test_input_cases()