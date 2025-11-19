import torch
import torch.nn as nn
from typing import List


def random_uniform_with_excludent_columns(
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


class _BaseSyntheticTreatmentLoss(nn.Module):

    def __init__(
        self,
        loss_method,
        n_repeats=1,
        binary_dims=None,
        dummy_sets=None,
        mutually_excludent_sets=None,
    ):

        super().__init__()

        self.loss_method = loss_method
        self.n_repeats = n_repeats
        self.binary_dims = binary_dims
        self.dummy_sets = dummy_sets
        self.mutually_excludent_sets = mutually_excludent_sets

        assert loss_method in ["balanced", "uniform"]

    def generate_t_synth(self, t):
        return _generate_t_synth(
            t,
            self.loss_method,
            self.binary_dims,
            self.dummy_sets,
            self.mutually_excludent_sets,
        )


class LogisticRandomLoss(nn.Module):
    def __init__(
        self,
        loss_method,
        n_repeats=1,
        binary_dims=None,
        dummy_sets=None,
        mutually_excludent_sets=None,
    ):

        super().__init__(
            loss_method,
            n_repeats=n_repeats,
            binary_dims=binary_dims,
            dummy_sets=dummy_sets,
            mutually_excludent_sets=mutually_excludent_sets,
        )

    def forward(self, model, x, t):

        x_synth = x.repeat(self.n_repeats, 1)
        t_synth = self.generate_t_synth(t.repeat(self.n_repeats, 1))

        weight_denominator = model(x, t)
        weight_numerator = model(x_synth, t_synth)

        # 1 / (w + 1) = p_den(x,t) / (p_num(x,t) + p_den(x,t))
        proba_1 = 1 / (weight_denominator + 1)
        # 1 / (w + 1) = p_num(x,t) / (p_num(x,t) + p_den(x,t))
        proba_0 = weight_numerator / (weight_numerator + 1)

        loss_1 = -torch.log(proba_1).mean()
        loss_0 = -torch.log(proba_0).mean()

        return (loss_0 + loss_1) / 2


class LogisticLoss(nn.Module):
    def __init__(
        self,
        l2_penalty=0,
        l2_inv_penalty=0,
    ):

        self.l2_penalty = l2_penalty
        self.l2_inv_penalty = l2_inv_penalty
        super().__init__()

    def forward(self, model, x, t, x_synth, t_synth):

        # Ptarget/Psource | Psource
        weight_denominator = model(x, t)

        if x_synth.dim() > 2:
            x_synth = x_synth.reshape((-1, x_synth.shape[-1]))
            t_synth = t_synth.reshape((-1, t_synth.shape[-1]))

        # Ptarget/Psource | Ptarget
        weight_numerator = model(x_synth, t_synth)

        # 1 / (w + 1) = p_den(x,t) / (p_num(x,t) + p_den(x,t))
        proba_source = 1 / (weight_denominator + 1)
        # w / (w + 1) = p_num(x,t) / (p_num(x,t) + p_den(x,t))
        proba_target = weight_numerator / (weight_numerator + 1)

        loss_l2 = self.l2_penalty * (weight_denominator.mean() - 1) ** 2
        loss_l2_inv = self.l2_inv_penalty * (1 / weight_numerator.mean() - 1) ** 2

        loss_target = torch.mean(-torch.log(proba_source))
        loss_source = torch.mean(-torch.log(proba_target))
        return loss_source + loss_target + loss_l2 + loss_l2_inv


def _generate_t_synth(t, loss_method, binary_dims, dummy_sets, mutually_excludent_sets):
    if loss_method == "uniform":
        t_synth = random_uniform_with_excludent_columns(
            t,
            binary_columns=binary_dims,
            dummy_sets=dummy_sets,
            mutually_excludend_sets=mutually_excludent_sets,
        )
    elif loss_method == "balanced":
        t_synth = t[torch.randperm(t.shape[0])]
    else:
        raise ValueError("Invalid loss method")
    return t_synth
