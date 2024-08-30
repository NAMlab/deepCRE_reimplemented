import pickle

if __name__ == "__main__":
    genes_pickled = "validation_genes.pickle"
    with open(genes_pickled, 'rb') as handle:
            validation_genes = pickle.load(handle)
            print(validation_genes.keys())
            print(type(validation_genes["ara"]))
            print(type(validation_genes["ara"][0]))