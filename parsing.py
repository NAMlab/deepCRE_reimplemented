from __future__ import annotations
import argparse
from enum import Enum, auto
import json
import os
from typing import Any, Dict, Iterable, List, Tuple
from copy import deepcopy

import pandas as pd

from utils import make_absolute_path, read_feature_from_input_dict


possible_general_parameters = {
    "genome": None,
    "annotation": None,
    "targets": None,
    "output_name": None,
    "chromosomes": None,
    "pickle_key": None,
    "pickle_file": "validation_genes.pickle",
    "model_case": None,
    "ignore_small_genes": True,
    "train_val_split": False,
    "extragenic": 1000,
    "intragenic": 500
}

possible_species_parameters = {
    "genome": None,
    "annotation": None,
    "targets": None,
    "chromosomes": None,
    "pickle_key": None,
    "pickle_file": "validation_genes.pickle",
    "species_name": None,
}


class RunInfo:
    general_info: Dict[str, Any]
    species_info: List[Dict[str, Any]]

    def set_up_defaults(self):
        #get general defaults
        defaults = possible_species_parameters.copy()
        #get values from general columns that can be used for species infos
        applicable_general_inputs = list(set(possible_general_parameters.keys()).intersection(set(possible_species_parameters.keys())))
        applicable_general_inputs = {key: self.general_info[key] for key in applicable_general_inputs if key in self.general_info.keys()}
        #overwrite the defaults with info from general_info, where applicable
        defaults.update(applicable_general_inputs)
        applicable_general_inputs = defaults
        #remove key/value pairs that dont have a meaningful default or a value from the general_info, so they dont interfere later
        to_delete = [key for key, value in applicable_general_inputs.items() if value is None]
        for key in to_delete:
            del applicable_general_inputs[key]
        return applicable_general_inputs

    @staticmethod
    def check_general_keys(general_dict: Dict[str, Any]):
        necessary_parameters = list(set(possible_general_parameters).difference(set(possible_species_parameters)))
        missing = [parameter for parameter in necessary_parameters if parameter not in general_dict.keys()]
        if missing:
            raise ValueError(f"Error parsing general info dict! Input parameters {missing} missing!")

    @staticmethod
    def check_species_keys(specie_dict: Dict[str, Any], missing_species_name_ok: bool = False):
        missing = [parameter for parameter in possible_species_parameters.keys() if parameter not in specie_dict.keys()]
        if missing_species_name_ok and missing == ["species_name"]:
            return
        if missing:
            raise ValueError(f"Error parsing specie dict! Input parameters {missing} missing!")
    
    def load_chromosomes(self) -> None:
        for specie_info in self.species_info:
            chromosomes = specie_info["chromosomes"]
            if isinstance(chromosomes, str):
                if os.path.isfile(chromosomes):
                    chromosomes = pd.read_csv(chromosomes).values.ravel().tolist()
                else:
                    chromosomes = pd.read_csv(make_absolute_path("genome", chromosomes, start_file=__file__), header=None).values.ravel().tolist()
            chromosomes = [str(chrom) for chrom in chromosomes]
            specie_info["chromosomes"] = chromosomes


    def parse_species(self, run_dict: Dict[str, Any]):
        #list of parameters in both general and species parameters
        applicable_general_inputs = self.set_up_defaults()
        species_info = []
        for species_dict in run_dict.get("species_data", []):
            curr_specie_dict = {key: read_feature_from_input_dict(species_dict, key) for key in possible_species_parameters.keys() if key in species_dict.keys()}
            #get defaults, then overwrite them with the data that was read in for all keys where data was read in
            curr_general_info = applicable_general_inputs.copy()
            curr_general_info.update(curr_specie_dict)
            #make sure all necessary parameters are filled
            RunInfo.check_species_keys(curr_general_info)
            species_info.append(curr_general_info)
        if not species_info:
            curr_general_info = applicable_general_inputs.copy()
            RunInfo.check_species_keys(curr_general_info, missing_species_name_ok=True)
            species_info.append(curr_general_info)
        self.species_info = species_info
        self.load_chromosomes()

    def parse_general_inputs(self, run_dict: Dict[str, str]):
        #load defaults
        defaults = possible_general_parameters.copy()
        read_data = {key: read_feature_from_input_dict(run_dict, key) for key in possible_general_parameters.keys() if key in run_dict.keys()}
        #overwrite defaults with read data
        defaults.update(read_data)
        general_info = defaults
        #remove empty defaults
        to_delete = [key for key, value in general_info.items() if value is None]
        for key in to_delete:
            del general_info[key]
        general_info["model_case"] = ModelCase.parse(general_info["model_case"])
        #make check
        RunInfo.check_general_keys(general_dict=general_info)
        self.general_info = general_info

    @staticmethod
    def parse(run_dict: Dict[str, Any]):
        run_info_object = RunInfo()
        run_info_object.parse_general_inputs(run_dict=run_dict)
        # load general info first, and use it as defaults for specific species
        run_info_object.parse_species(run_dict)
        if run_info_object.general_info["model_case"] == ModelCase.MSR and len(run_info_object.species_info) < 2:
            raise ValueError(f"Need at least 2 species for MSR training! Only found {len(run_info_object.species_info)}.")
        for species_key in possible_species_parameters.keys():
            if species_key in run_info_object.general_info.keys():
                del run_info_object.general_info[species_key]
        return run_info_object
    
    def is_msr(self) -> bool:
        return self.general_info["model_case"] == ModelCase.MSR
    
    def __str__(self) -> str:
        self.general_info["model_case"] = str(self.general_info["model_case"])
        specie_info_json = json.dumps(self.species_info, indent=2)
        gen_info_json = json.dumps(self.general_info, indent=2)
        self.general_info["model_case"] = ModelCase.parse(self.general_info["model_case"])
        result = "RunInfo(\n  General Info{"
        for gen_info in gen_info_json.split("\n"):
            if gen_info in ["{", "}"]:
                continue
            result += "\n  " + gen_info
        result += "\n  },\n  Species info["
        for species_info in specie_info_json.split("\n"):
            if species_info in ["[", "]"]:
                continue
            result += "\n  " + species_info
        result += "\n  ]\n)"
        return result


class ParsedTrainingInputs:
    run_infos: List[RunInfo]

    def __init__(self):
        self.run_infos = []

    @staticmethod
    def parse(json_file_name: str) -> Tuple[ParsedTrainingInputs, List[Tuple[str, int, Exception]]]:
        json_file_name = json_file_name if os.path.isfile(json_file_name) else make_absolute_path(json_file_name, __file__)
        failed_parsings = []
        parsed_object = ParsedTrainingInputs()
        with open(json_file_name, "r") as f:
            input_list = json.load(f)
        for i, run_dict in enumerate(input_list):
            try:
                curr_run_info = RunInfo.parse(run_dict)
                parsed_object.run_infos.append(curr_run_info)
            except Exception as e:
                print(f"error reading input run number {i}.")
                print(f"error message is: \"{e}\"")
                print(f"the dictionary that was loaded for the run is the following:")
                print(f"{json.dumps(run_dict, indent=2)}")
                failed_parsings.append(("error during parsing!", i, e))
        return parsed_object, failed_parsings
    
    def replace_both(self) -> ParsedTrainingInputs:
        new_object = ParsedTrainingInputs()
        for info in self.run_infos:
            if info.general_info["model_case"] == ModelCase.BOTH:
                info.general_info["model_case"] = ModelCase.SSR
                ssc_version = deepcopy(info)
                ssc_version.general_info["model_case"] = ModelCase.SSC
                new_object.run_infos.append(info)
                new_object.run_infos.append(ssc_version)
            else:
                new_object.run_infos.append(info)
        return new_object
    
    def __iter__(self) -> Iterable[RunInfo]:
        return (info for info in self.run_infos)
    
    def __str__(self) -> str:
        result = "ParsedTrainingInputs["
        for info in self.run_infos:
            for info_line in str(info).split("\n"):
                result += "\n  " + info_line
        result += "\n]"
        return result
    
    def __len__(self) -> int:
        return len(self.run_infos)
            


class ModelCase(Enum):
    MSR = auto()
    SSR = auto()
    SSC = auto()
    BOTH = auto()

    @staticmethod
    def parse(input_string: str) -> ModelCase:
        if input_string.lower() == "msr":
            return ModelCase.MSR
        elif input_string.lower() == "ssr":
            return ModelCase.SSR
        elif input_string.lower() == "ssc":
            return ModelCase.SSC
        elif input_string.lower() == "both":
            return ModelCase.BOTH
        else:
            raise ValueError(f"model case \"{input_string}\" not recognized!")
    
    def __str__(self) -> str:
        if self == ModelCase.MSR:
            return "msr"
        elif self == ModelCase.SSR:
            return "ssr"
        elif self == ModelCase.SSC:
            return "ssc"
        elif self == ModelCase.BOTH:
            return "both"
        else:
            raise NotImplementedError("string method was not implemented for this Variant yet!")


def convert_training_csv_to_json(csv_path: str, output_json_path: str, **command_line_args):
    df = pd.read_csv(csv_path, header=None)
    json_compatible_inputs = []
    if len(df.columns) == 6:
        for i, (genome, gtf, tpm, output, chroms, p_key) in df.iterrows():
            copied = command_line_args.copy()
            read_in = {
                "genome": genome,
                "annotation": gtf,
                "targets": tpm,
                "output_name": output,
                "chromosomes": chroms,
                "pickle_key": p_key,
            }
            copied.update(read_in)
            json_compatible_inputs.append(copied)
    with open(output_json_path, "w") as f:
        json.dump(json_compatible_inputs, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", help="path to the csv file to be converted", required=True, type=str)
    parser.add_argument("--output_path", help="path to the save location of the converted json output", required=True, type=str)
    parser.add_argument("--command_line_args", help="command line arguments that would have been used with this script", required=True, type=str, nargs="+")
    args = parser.parse_args()
    cmd_line_list = args.command_line_args
    cmd_line_args = {}
    for i in range(0, len(cmd_line_list), 2):
        cmd_line_args[cmd_line_list[i]] = cmd_line_list[i+1]
    return args.csv_path, args.output_path, cmd_line_args

if __name__ == "__main__":
    csv, out, args = parse_args()
    convert_training_csv_to_json(csv_path=csv, output_json_path=out, **args)
