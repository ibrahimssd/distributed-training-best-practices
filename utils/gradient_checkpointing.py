"""Gradient checkpointing wrappers."""

import torch.nn as nn


def enable_gradient_checkpointing(model: nn.Module) -> nn.Module:
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return model
