import torch


class EarlyStopping:
    def __init__(self, patience=7, delta=0):
        """
        Args:
            patience (int): Number of epochs with no improvement to wait before stopping training.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in validation loss to qualify as improvement.
            path (str): Path to save the best model checkpoint.
        """
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, metric):
        score = -metric

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
