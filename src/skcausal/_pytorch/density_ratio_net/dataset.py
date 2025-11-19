import torch
import torch.nn as nn
from typing import List
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from skcausal._pytorch.utils.data import _force_numeric_to_default_dtype


class SyntheticDatasetResampleOnce(Dataset):
    """Dataset with synthetic and real data.

    For the same batch, we see X with different t.
    """

    def __init__(
        self,
        X,
        t,
        method="uniform",
        binary_dims=None,
        dummy_sets=None,
        mutually_excludent_sets=None,
        **kwargs
    ):
        self.method = method
        self.binary_dims = binary_dims
        self.dummy_sets = dummy_sets
        self.mutually_excludent_sets = mutually_excludent_sets
        if method not in ["uniform", "balanced"]:
            raise ValueError("Invalid method")

        X = _force_numeric_to_default_dtype(X)
        t = _force_numeric_to_default_dtype(t).reshape((-1, 1))

        X_synth = X
        t_synth = _generate_t_synth(
            t,
            method=method,
            binary_dims=binary_dims,
            dummy_sets=dummy_sets,
            mutually_excludent_sets=mutually_excludent_sets,
        )

        self.X = X
        self.t = t
        self.X_synth = X_synth
        self.t_synth = t_synth

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        return self.X[idx], self.t[idx], self.X_synth[idx], self.t_synth[idx]


class SyntheticDatasetShuffled(SyntheticDatasetResampleOnce):
    """Dataset with synthetic and real data shuffled randomly"""

    def __init__(
        self,
        X,
        t,
        method="uniform",
        binary_dims=None,
        dummy_sets=None,
        mutually_excludent_sets=None,
        **kwargs
    ):
        super().__init__(X, t, method, binary_dims, dummy_sets, mutually_excludent_sets)

        shuffle_idx = torch.randperm(self.X_synth.shape[0])
        self.X_synth = _force_numeric_to_default_dtype(self.X_synth[shuffle_idx])
        self.t_synth = _force_numeric_to_default_dtype(self.t_synth[shuffle_idx])


class SyntheticDatasetResampleEachBatch(SyntheticDatasetResampleOnce):

    def __init__(
        self,
        X,
        t,
        method="uniform",
        binary_dims=None,
        dummy_sets=None,
        mutually_excludent_sets=None,
        n_repeats=1,
        **kwargs
    ):

        self.n_repeats = n_repeats
        super().__init__(X, t, method, binary_dims, dummy_sets, mutually_excludent_sets)

    def __getitem__(self, idx):
        X, t, X_synth, _ = super().__getitem__(idx)

        # Repeat X_synth self.n_repeats times
        X_synth = X_synth.repeat(self.n_repeats, 1)
        t_synth = t.repeat(self.n_repeats, 1)

        t_synth = _generate_t_synth(
            t_synth,
            method=self.method,
            binary_dims=self.binary_dims,
            dummy_sets=self.dummy_sets,
            mutually_excludent_sets=self.mutually_excludent_sets,
            t_samples=self.t,
        )
        return X, t, X_synth, t_synth


def _generate_t_synth(
    t, method, binary_dims, dummy_sets, mutually_excludent_sets, t_samples=None
):

    if method == "uniform":
        t_synth = _random_uniform_with_excludent_columns(
            t,
            binary_columns=binary_dims,
            dummy_sets=dummy_sets,
            mutually_excludend_sets=mutually_excludent_sets,
        )
    elif method == "balanced":
        if t_samples is None:
            # Permutation of t
            t_synth = t[torch.randperm(t.shape[0])]
        else:
            t_synth = t_samples[torch.randint(0, t_samples.shape[0], (t.shape[0],))]
    else:
        raise ValueError("Invalid loss method")
    return t_synth


def _random_uniform_with_excludent_columns(
    t: torch.Tensor,
    binary_columns: List[int],
    dummy_sets: List[List[int]],
    mutually_excludend_sets: List[List[int]],
):
    """
    Generates a vector similar to `t`, but with random values along dimensions,
      respecting the support of each column
    and columns that are mutually exclusive (such as dummies).
    """
    t_synth = torch.rand_like(t)

    if binary_columns is not None:
        for dim in binary_columns:
            t_synth[:, dim] = torch.round(t_synth[:, dim])

    if dummy_sets is not None:
        for dummy_set in dummy_sets:
            # Create tensor of possible values for those dummy columns.
            # For example, for 2 dummies (3 categories), possible values would be
            # [[0, 0], [0, 1], [1, 0]]
            possible_values = torch.eye(len(dummy_set) + 1)[:, 1:]
            t_synth[:, dummy_set] = possible_values[
                torch.randint(0, possible_values.shape[0], (t_synth.shape[0],))
            ]

    if mutually_excludend_sets is not None:
        for cols in mutually_excludend_sets:
            mask = torch.zeros((t_synth.shape[0], len(cols)))
            idx_to_set_to_1 = torch.randint(0, len(cols), (1,))
            mask[:, idx_to_set_to_1] = 1

            t_synth[:, cols] = mask * t_synth[:, cols]

    return t_synth
