import numpy as np
from torch.utils.data import Subset


def uniform_split(dataset, split_ratio):
    """

    Rozdeli dataset na trenovacie a testovacie (alebo validacne) v urcitom pomere. Z kazdej triedy zoberie
        posledne vzorky, kde pocet vzoriek zavisi od pomeru rozdelenia.

    Args:
        dataset (torch.utils.data.Dataset): DataSet class a vsetky jeho varianty, tj. aj ImageFolder.
        split_ratio (_type_): _description_

    Returns:
        [torch.utils.data.Subset, torch.utils.data.Subset] : Dva subsety (dedia od DataSet) rozdelene v pozadovanom pomere
    """
    class_counts = count_class_samples(dataset)
    limit_indices = _get_classes_limit_indices(class_counts)
    indices = _get_indices_array(limit_indices)
    indices_big, indices_small = _split_dataset_indices(indices, split_ratio)
    subset_big = Subset(dataset, _concat_indices_into_list(indices_big))
    subset_small = Subset(dataset, _concat_indices_into_list(indices_small))

    return subset_big, subset_small


def count_class_samples(dataset):
    """Funkcia spocita pocet vzoriek patriacich do kazdej triedy. Pre rychlost pouzivat 'cached=True', no plati to iba na dataset
        architektury. Ak by to z nejakeho dovodu nefungovalo, treba switchnut spat na 'cached=False'.

    Args:
        dataset: (torchvision.datasets.ImageFolder): Dataset obrazkov nacitany cez ImageFolder classu.

    Returns:
        dict: Dictionary kde je ku kazdemu indexu (ktory reprezentuje triedu) priradeny int - pocet vzoriek
                v triede
    """

    class_counts = {}
    for _, label in dataset:
        label = int(label)
        if label not in class_counts:
            class_counts[label] = 1
        else:
            class_counts[label] += 1

    return class_counts


def _get_classes_limit_indices(class_counts):
    """Funkcia vypocita limitne indexy pre kazdu triedu v datasete

    Args:
        class_counts (dict): Dictionary s poctami vzoriek v kazdej triede (z funkcie '__count_classs_samples')

    Returns:
        ndarray: Pole kde indexy reprezentuju cislo triedy a hodnoty prestavuju limitny index triedy
    """
    counts = list(class_counts.values())
    idxs = np.zeros(len(counts))

    for idx, count in enumerate(counts):
        if idx == 0:
            idxs[idx] = count - 1
        else:
            idxs[idx] = np.sum(counts[idx::-1]) - 1

    return idxs


def _get_indices_array(idxs):
    """Funkcia vytvori polia indexov jednotlivych tried

    Args:
        idxs (ndarray): Pole kde indexy reprezentuju cislo triedy a hodnoty prestavuju limitny index triedy

    Returns:
        dict: Dictionary kde kazdy key je cislo triedy a value tohto key je numpy pole indexov, ktore
                do tejto triedy patria
    """
    indices = {}
    for i in range(len(idxs)):
        if i == 0:
            indices[str(i)] = np.arange(0, idxs[i] + 1, 1)
        else:
            indices[str(i)] = np.arange(idxs[i - 1] + 1, idxs[i] + 1, 1)

    return indices


def _split_dataset_indices(indices, split_ratio):
    """Rozdeli dataset podla zvoleneho pomeru

    Args:
        indices (dict): Dictionary s indexmi classov
        split_ratio (float [0-1]): Pomer rozdelenia datasetu

    Returns:
        dict, dict: Vrati orezany dataset a zaroven odrezok z datasetu
    """

    selection = {}
    split_indices = {}
    for i in indices:
        working_arr = indices[str(i)]
        num_to_select = int(len(working_arr) * split_ratio)
        selection[str(i)] = working_arr[-num_to_select:]
        split_indices[str(i)] = working_arr[:-num_to_select]

    return split_indices, selection


def _concat_indices_into_list(indices):
    """Pretvori dict na 1D list kvoli vstupu do torch.data.utils.Subset

    Args:
        indices (dict): Dict s indexmi classov

    Returns:
        List: 1D List s indexmi classov
    """
    concatenated_array = []
    for _, value in indices.items():
        concatenated_array.extend(value)

    return list(map(int, concatenated_array))
