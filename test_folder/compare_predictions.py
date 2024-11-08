import os
import re
import pandas as pd


def get_best_model():
    df = pd.read_csv("/home/gernot/Data/deepCRE_website_paper/compiled_results.csv", index_col=0)
    print(df.idxmax())


def get_best_model_val_chrom():
    results_folder = "/home/gernot/Data/deepCRE_website_paper/training_outputs"
    data_dict =  {}
    for file in sorted(os.listdir(results_folder)):
        name = "_".join(file.split("_")[:2])
        results_df = pd.read_csv(os.path.join(results_folder, file))
        results_list = list(results_df["accuracy"])
        data_dict[name] = results_list
    match_list_lengths(data_dict)
    df = pd.DataFrame(data_dict)
    print(df.idxmax() + 1)

def compile_results():
    data_dict = {}
    for res_file_name in ["arabidopsis_leaf_1_SSR_train_ssr_models_241104_110233_deepcre_crosspredict_arabidopsis_leaf_241104_183042.csv", ]:
        name = "_".join(res_file_name.split("_")[:2])
        res_dict = compare_results_crosspred(res_file_name)
        res_list = []
        keys = [int(key) for key in res_dict.keys()]
        keys = sorted(keys)
        keys = [str(key) for key in keys]
        for key in keys:
            res_list.append(res_dict[key])
        data_dict[name] = res_list
    match_list_lengths(data_dict)
    df = pd.DataFrame(data_dict)
    print(df)

def match_list_lengths(data_dict):
    max_length = max([len(list_) for list_ in data_dict.values()])
    for list_ in data_dict.values():
        length_diff = max_length - len(list_)
        to_append = [None for _ in range(length_diff)]
        list_.extend(to_append)



def compare_results_crosspred(file_name: str):
    data = pd.read_csv(f"test_folder/{file_name}")
    data = data.loc[data["true_targets"] != 2]
    length = len(data)
    rename_cols(data)
    results = {}
    for col in data.columns:
        try:
            # this will only continue if column name is numerical, otherwise will just skip iteration of loop
            int(col)
            data[col] = (data[col] >= 0.5).astype("int")
            correct = data[col] == data["true_targets"]
            correct_counts = correct.value_counts()
            correct_count = correct_counts[True]
            results[col] = correct_count / length
        except ValueError:
            pass
    return results



def rename_cols(data):
    regex = "_(\d+)_"       #type:ignore
    regex = re.compile(regex)
    new_cols = []
    for (i, column) in enumerate(data.columns):
        match = regex.search(column)
        if match:
            new_cols.append(match.group(1))
        else:
            new_cols.append(column)
    data.columns = new_cols


def predicionts_equal():
    new = pd.read_csv("arabidopsis_deepcre_predict_240916_170201_fritz_merge.csv")
    old = pd.read_csv("arabidopsis_deepcre_predict_240916_165930_meins.csv")
    older_old = pd.read_csv("arabidopsis_deepcre_predict_240827_144836_same_old.csv")


    print(new.head())
    print(old.head())
    print(older_old.head())

    new["preds"] = new["pred_probs"] > 0.5
    old["preds"] = old["pred_probs"] > 0.5
    older_old["preds"] = older_old["pred_probs"] > 0.5

    res_series = new["preds"] == old["preds"]
    res_series_old = old["preds"] == older_old["preds"]

    print(res_series.head())
    print(res_series.value_counts())

    print(res_series_old.head())
    print(res_series_old.value_counts())


if __name__ == "__main__":
    # predicionts_equal()
    # print(compare_results_crosspred(file_name="arabidopsis_leaf_1_SSR_train_ssr_models_241104_110233_deepcre_crosspredict_arabidopsis_leaf_241104_183042.csv"))
    # compile_results()
    get_best_model_val_chrom()
