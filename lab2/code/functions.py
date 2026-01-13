import numpy as np
import pandas as pd

labeled_files = ["O013257.npz", "O013490.npz", "O012791.npz"]

def intodf(file):
    """
    Convert a .npz file into a pandas DataFrame with labeled columns.

    Parameters:
    -----------
    file : str
        Name of the .npz file (e.g., 'O012345.npz').

    Returns:
    --------
    df : pandas.DataFrame
        DataFrame with columns ['Y', 'X', 'NDAI', 'SD', 'CORR', 'RDF', 'RCF', 'RBF', 'RAF', 'RAN', 'label']
        for labeled data, or without 'label' for unlabeled data.

    Notes:
    ------
    Assumes file is in '../data/data_labeled/' or '../data/data_unlabeled/' and 'labeled_files' is defined.
    """
    if file in labeled_files:
        file_path = "../data/data_labeled/" + file
        data = np.load(file_path)
        array = data['arr_0']
        df = pd.DataFrame(array)
        df.columns = ['Y','X','NDAI','SD','CORR','RDF','RCF','RBF','RAF','RAN','label']
    else:
        file_path = "../data/data_unlabeled/" + file
        data = np.load(file_path)
        array = data['arr_0']
        df = pd.DataFrame(array)
        df.columns = ['Y','X','NDAI','SD','CORR','RDF','RCF','RBF','RAF','RAN']
    return df