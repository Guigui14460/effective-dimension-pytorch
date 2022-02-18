from itertools import product
from typing import Union

from nngeometry.metrics import FIM
from nngeometry.object import PMatKFAC
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data


def get_fisher_matrix(model: nn.Module, dataloader: data.DataLoader, model_output_size: int, 
                      device: Union[str, torch.device] = 'cpu', variant: str = 'classif_logits') -> FIM:
    """
    Gets the Fisher information matrix object.

    Parameters
    ----------
        model: nn.Module
            Model to compute Fisher matrix
        dataloader: data.DataLoader
            Training set to compute Fisher matrix
        model_output_size: int
            Number of outputs of the model
        device: Union[str, torch.device] (optional, default = 'cpu')
            Target device for the returned matrix
        variant: str (optional, default = 'classif_logits')
            Variant to use depending on how you interpret your function. Possible choices are:
                - 'classif_logits' when using logits for classification
                - 'regression' when using a gaussian regression model

    Returns
    -------
    fisher_matrix: nngeometry.metrics.FIM
        the computed Fisher matrix
    """
    return FIM(model, dataloader, PMatKFAC, model_output_size, device=device, variant=variant)


def get_fisher_matrix_eigenvalues(fisher_information_matrix: FIM) -> np.ndarray:
    """
    Gets all the eigenvalues of the K-FAC approximated Fisher matrix by computing the eigenvalues of
    a, g for every layer and exploiting the tensor product and block diagonal structure of the approximation.

    Parameters
    ----------
        fisher_information_matrix: nngeometry.metrics.FIM
            The Fisher information matrix object to get the eigenvalues
    
    Returns
    -------
    eigs: np.ndarray
        eigenvalues of the Fisher information matrix
    """
    s = fisher_information_matrix.generator.layer_collection.numel()
    full_ls = []
    for layer_id, layer in fisher_information_matrix.generator.layer_collection.layers.items():
        a, g = fisher_information_matrix.data[layer_id]
        evals_a, _ = torch.symeig(a)
        evals_a = torch.nan_to_num(evals_a)
        evals_g, _ = torch.symeig(g)
        evals_g = torch.nan_to_num(evals_g)
        full_ls.append([np.absolute(evals_a.detach().cpu().numpy()), np.absolute(evals_g.detach().cpu().numpy())])
    eigs = []
    contract_tuple = lambda t: t[0] * t[1]
    for ls in full_ls:
        eigs += list(map(contract_tuple, product(ls[0], ls[1])))
    return np.array(np.absolute(eigs))
