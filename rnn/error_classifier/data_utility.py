import numpy as np


def load_data(file_name, ratio, columns):
    """ Load data from file, select the desired columns (input, target)
    and split in train, test and validation according with the specified ratio

    :param file_name: Name of the file to open
    :type file_name: string
    :param ratio: Ratio for the split
    :type ratio: array_like ([0.8, 0.9])
    :param columns:
    :type columns: array_like ([0, 3])
    :return: Splitted datasets
    :rtype: array_like
    """
    DATA = np.loadtxt(file_name, dtype=bytes, delimiter='\t').astype(str)
    N = DATA.shape[0]
    DATA = DATA[:, columns]

    ratio = (ratio * N).astype(np.int32)
    rng = np.random.RandomState(777)
    ind = rng.permutation(N)
    y_train = DATA[ind[:ratio[0]], 1:]
    y_val = DATA[ind[ratio[0]:ratio[1]], 1:]
    y_test = DATA[ind[ratio[1]:], 1:]

    X_train = DATA[ind[:ratio[0]], 0]
    X_val = DATA[ind[ratio[0]:ratio[1]], 0]
    X_test = DATA[ind[ratio[1]:], 0]

    X_train_lengths = [len(i) for i in X_train]
    X_val_lengths = [len(i) for i in X_val]
    X_test_lengths = [len(i) for i in X_test]

    return X_train, X_train_lengths, X_val, X_val_lengths, X_test, X_test_lengths, y_train, y_val, y_test

def one_hot_encode(data, vocabulary):
    batches = []
    for e in data:
        batch = np.zeros((len(e), len(vocabulary)))
        for i in range(len(e)):
            batch[i, vocabulary.index(e[i])] = 1.0
        batches.append(batch)

    return np.asarray(batches)

def emb_encode(data, vocabulary):
    batches = []
    for e in data:
        batch = np.zeros(len(e))
        for i in range(len(e)):
            batch[i] = vocabulary.index(e[i])
        batches.append(batch)

    return np.asarray(batches)
    #return batches