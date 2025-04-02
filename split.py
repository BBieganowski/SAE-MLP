# custom_split.py
import numpy as np
from sklearn.model_selection import BaseCrossValidator

class PurgedGroupTimeSeriesSplit(BaseCrossValidator):
    """
    Custom purged group time series split that leaves a gap between
    train and validation (useful for time series).
    """
    def __init__(self, n_splits=5, group_gap=31):
        self.n_splits = n_splits
        self.group_gap = group_gap

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        unique_groups = np.unique(groups)
        n_samples_per_fold = len(unique_groups) // self.n_splits
        for i in range(self.n_splits):
            start = i * n_samples_per_fold
            if i < self.n_splits - 1:
                stop = (i+1) * n_samples_per_fold
            else:
                stop = len(unique_groups)
            val_groups = unique_groups[start:stop]
            # Purge logic
            train_groups = unique_groups[:max(0, start - self.group_gap)]
            yield (
                np.where(np.isin(groups, train_groups))[0],
                np.where(np.isin(groups, val_groups))[0]
            )
