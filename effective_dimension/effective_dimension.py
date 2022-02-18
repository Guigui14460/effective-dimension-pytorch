from typing import Union

from nngeometry.metrics import FIM
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

from .fisher_matrix import get_fisher_matrix_eigenvalues, get_fisher_matrix
from .utils import count_parameters


def effective_dimension(fisher_matrix: FIM, dimensions: int, training_dataset_size: int, gamma: float = 1, normalized: bool = True) -> float:
    """
    Computes the Effective dimension.

    Parameters
    ----------
        fisher_matrix: nngeometry.metrics.FIM
            Fisher information matrix object
        dimensions: int
            Number of trainable parameters
        training_dataset_size: int
            Size of the training dataset
        gamma: float (optional, default = 1)
            A constant in (2 * pi * log(training_dataset_size), 1]
        normalized: bool (optional, default = True)
            If activated, the effective dimension is normalized, else the default effective dimension is returned

    Returns
    -------
    effective_dimension: float
        default computed number if normalized parameter is False, normalized if normalized parameter is True

    Notes
    -----
    You can go see the [Abbas et al.] paper on arXiv (https://arxiv.org/abs/2112.04807).
    """
    c = 2 * np.pi * np.log(training_dataset_size)
    assert gamma <= 1 and gamma > c / training_dataset_size

    kappa = gamma * training_dataset_size / c

    eigs = get_fisher_matrix_eigenvalues(fisher_matrix)
    trace = fisher_matrix.trace().detach().cpu().numpy()
    normalized_fisher_eigs = dimensions * eigs / trace
    
    numerator = np.sum(np.log(1 + kappa * normalized_fisher_eigs))
    ed = numerator / np.log(kappa)
    
    if normalized:
        return ed / dimensions
    return ed


def get_effective_dimension(model: nn.Module, dataloader: data.DataLoader, model_output_size: int, 
                            training_dataset_size: int, device: Union[str, torch.device] = 'cpu', 
                            variant: str = 'classif_logits', gamma: float = 1, normalized: bool = True) -> float:
    """
    Computes the Effective dimension.

    Parameters
    ----------
        model: nn.Module
            Model to compute Fisher matrix
        dataloader: data.DataLoader
            Training set to compute Fisher matrix
        model_output_size: int
            Number of outputs of the model
        training_dataset_size: int
            Size of the training dataset
        device: Union[str, torch.device] (optional, default = 'cpu')
            Target device for the returned matrix
        variant: str (optional, default = 'classif_logits')
            Variant to use depending on how you interpret your function. Possible choices are:
                - 'classif_logits' when using logits for classification
                - 'regression' when using a gaussian regression model
        gamma: float (optional, default = 1)
            A constant in (2 * pi * log(training_dataset_size), 1]
        normalized: bool (optional, default = True)
            If activated, the effective dimension is normalized, else the default effective dimension is returned

    Returns
    -------
    effective_dimension: float
        default computed number if normalized parameter is False, normalized if normalized parameter is True

    """
    d = count_parameters(model)
    fisher = get_fisher_matrix(model, dataloader, model_output_size, device=device, variant=variant)
    ed = effective_dimension(fisher, d, training_dataset_size, gamma=gamma, normalized=normalized)
    return ed
