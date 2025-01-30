import os
import h5py



def load_interpretation_outputs():
    # load h5 file
    h5_file = h5py.File("src/deepCRE/results/shap/arabidopsis_deepcre_interpret_241017_170839.h5")
    for key, item in h5_file.items():
        print(key)
        print(item.shape)

def recursive_read(obj, depth=0):
    print(depth * " ", obj)
    try:
        keys = obj.keys()
    except AttributeError:
        return
    for key in keys:
        recursive_read(obj[key], depth=depth + 4)


def load_modisco_results():
    # load hdf5 file
    h5_file = h5py.File("src/deepCRE/results/modisco/arabidopsis_deepcre_motifs_241021_174945.hdf5")
    recursive_read(h5_file)


if __name__ == "__main__":
    load_modisco_results()