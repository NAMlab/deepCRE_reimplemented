from typing import Dict, List
import unittest
import pandas as pd
import os
import deepcre_crosspredict as cp


class TestCrossPredictions(unittest.TestCase):

    def test_cross_pred_integration(self):
        necessary_columns = {
            "genome": ["Arabidopsis_thaliana.TAIR10.dna.toplevel.fa"],
            "gene_model": ["Arabidopsis_thaliana.TAIR10.52.gtf"],
            "model_names": ["arabidopsis_1_SSR_train_ssr_models_240822_103323.h5;arabidopsis_2_SSR_train_ssr_models_240822_105523.h5"],
            "subject_species_name": ["arabidopsis"]
        }
        input_path = self.create_test_input(necessary_columns, "base_case.csv")
        os.system("pwd")
        os.system(f"python deepcre_crosspredict.py -i {input_path}")
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
            "model_names": ["arabidopsis_1_SSR_train_ssr_models_240822_103323.h5;arabidopsis_2_SSR_train_ssr_models_240822_105523.h5"],
            "subject_species_name": ["arabidopsis"]
        }, "base_case.csv"))
        test_input_dicts.append(({
            "genome": ["Arabidopsis_thaliana.TAIR10.dna.toplevel.fa"],
            "model_names": ["arabidopsis_1_SSR_train_ssr_models_240822_103323.h5;arabidopsis_2_SSR_train_ssr_models_240822_105523.h5"],
            "subject_species_name": ["arabidopsis"]
        }, "one_column_missing.csv"))
        test_input_dicts.append(({
            "genome": ["Arabidopsis_thaliana.TAIR10.dna.toplevel.fa"],
        }, "one_column_only.csv"))
        test_input_dicts.append(({
            "genome": ["Arabidopsis_thaliana.TAIR10.dna.toplevel.fa"],
            "gene_model": ["Arabidopsis_thaliana.TAIR10.52.gtf"],
            "model_names": ["arabidopsis_1_SSR_train_ssr_models_240822_103323.h5;arabidopsis_2_SSR_train_ssr_models_240822_105523.h5"],
            "random_addon_col": ["arabidopsis_1_SSR_train_ssr_models_240822_103323.h5;arabidopsis_2_SSR_train_ssr_models_240822_105523.h5"],
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
    stupid_obejct.test_check_input()