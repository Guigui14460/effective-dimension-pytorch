import numpy as np
import torch.nn as nn


def generate_random(num_inputs: int, *input_size: int):
    """
    Generate noise.

    Parameters
    -----------
        num_inputs: int
            first dimension
        input_size: int
            other dimensions

    Returns
    -------
    np.ndarray
        numpy array of shape (num_inputs, input_size[0], input_size[1], ... input_size[n-1])
    """
    return np.random.normal(0, 1, size=(num_inputs, *input_size))


def count_parameters(model: nn.Module) -> int:
    """
    Counts the number of learneable parameters in a neural network model.

    Parameters
    ----------
        model: nn.Module
            model to count learneable parameters

    Returns
    -------
    int
        number of learneable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
