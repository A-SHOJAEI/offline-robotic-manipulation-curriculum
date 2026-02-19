"""Neural network utility functions."""

import torch.nn as nn


def get_activation(name: str) -> nn.Module:
    """Get activation function by name.

    Args:
        name: Name of activation function ('relu', 'tanh', 'elu', 'leaky_relu').

    Returns:
        Activation module instance.

    Raises:
        ValueError: If activation name is not recognized.
    """
    activations = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "elu": nn.ELU(),
        "leaky_relu": nn.LeakyReLU(),
    }

    activation = activations.get(name.lower())
    if activation is None:
        raise ValueError(
            f"Unknown activation '{name}'. Available: {list(activations.keys())}"
        )

    return activation
