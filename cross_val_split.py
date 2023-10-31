import numpy as np



def cross_val_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = np.random.randint(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)