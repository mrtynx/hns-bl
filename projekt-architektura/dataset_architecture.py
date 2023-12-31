import numpy as np
import torch
import pickle
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.datasets import ImageFolder

ARCHITECTURAL_COUNTS = {
    0: 392,
    1: 362,
    2: 364,
    3: 405,
    4: 566,
    5: 615,
    6: 456,
    7: 315,
    8: 424,
    9: 312,
    10: 278,
    11: 480,
    12: 335,
    13: 280,
    14: 381,
    15: 331,
    16: 523,
    17: 417,
    18: 382,
    19: 343,
    20: 321,
    21: 720,
    22: 301,
    23: 331,
    24: 455,
}


def get_classes_weights():
    global ARCHITECTURAL_COUNTS
    total = _sum_count_samples(ARCHITECTURAL_COUNTS)
    weights = []
    for val in ARCHITECTURAL_COUNTS.values():
        weights.append(total / val)

    weights = np.array(weights, dtype=np.float32)

    return weights / np.max(weights)


def get_random_split_arch_dataset(root_path, transform, split, seed):
    dataset = ImageFolder(root=root_path, transform=transform)

    train_split = int(np.ceil(len(dataset) * split))
    test_split = int(np.fix(len(dataset) * (1 - split)))

    print(f"Train size : {train_split}, Test size : {test_split}")
    train_dataset, test_dataset = random_split(
        dataset=dataset,
        lengths=[train_split, test_split],
        generator=torch.Generator().manual_seed(seed),
    )

    return train_dataset, test_dataset


def get_architectural_dataset(root_path, transform, batch_sz, test, val=0.0):
    dataset = ImageFolder(root=root_path, transform=transform)
    train, _test = _create_split_subset(dataset, test)

    dataloader_test = DataLoader(
        _test, batch_size=batch_sz, shuffle=False, num_workers=8, pin_memory=True
    )

    if val != 0.0:
        train, _val = _create_split_subset(train, val)
        dataloader_val = DataLoader(
            _val, batch_size=batch_sz, shuffle=False, num_workers=8, pin_memory=True
        )
        dataloader_train = DataLoader(
            train, batch_size=batch_sz, shuffle=True, num_workers=8, pin_memory=True
        )
        return dataloader_train, dataloader_test, dataloader_val

    else:
        dataloader_train = DataLoader(
            train, batch_size=batch_sz, shuffle=True, num_workers=8, pin_memory=True
        )
        return dataloader_train, dataloader_test


def _create_split_subset(dataset, split_ratio):
    """

    Rozdeli dataset na trenovacie a testovacie (alebo validacne) v urcitom pomere. Z kazdej triedy zoberie
        posledne vzorky, kde pocet vzoriek zavisi od pomeru rozdelenia.

    Args:
        dataset (torch.utils.data.Dataset): DataSet class a vsetky jeho varianty, tj. aj ImageFolder.
        split_ratio (_type_): _description_

    Returns:
        [torch.utils.data.Subset, torch.utils.data.Subset] : Dva subsety (dedia od DataSet) rozdelene v pozadovanom pomere
    """
    class_counts = count_class_samples(dataset, cached=True)
    limit_indices = _get_classes_limit_indices(class_counts)
    indices = _get_indices_array(limit_indices)
    indices_big, indices_small = _split_dataset_indices(indices, split_ratio)
    subset_big = Subset(dataset, _concat_indices_into_list(indices_big))
    subset_small = Subset(dataset, _concat_indices_into_list(indices_small))

    return subset_big, subset_small


def count_class_samples(dataset, cached=False):
    """Funkcia spocita pocet vzoriek patriacich do kazdej triedy. Pre rychlost pouzivat 'cached=True', no plati to iba na dataset
        architektury. Ak by to z nejakeho dovodu nefungovalo, treba switchnut spat na 'cached=False'.

    Args:
        dataset: (torchvision.datasets.ImageFolder): Dataset obrazkov nacitany cez ImageFolder classu.

    Returns:
        dict: Dictionary kde je ku kazdemu indexu (ktory reprezentuje triedu) priradeny int - pocet vzoriek
                v triede
    """

    if cached:
        global ARCHITECTURAL_COUNTS
        return ARCHITECTURAL_COUNTS
    else:
        class_counts = {}
        for _, label in dataset:
            if label not in class_counts:
                class_counts[label] = 1
            else:
                class_counts[label] += 1

        return class_counts


def _sum_count_samples(class_counts):
    total = 0
    for val in class_counts.values():
        total += val

    return total


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
