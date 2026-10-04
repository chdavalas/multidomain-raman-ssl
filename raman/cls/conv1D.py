import lightning as L
import logging
import torch.nn as nn
import torch.nn.functional as F
from raman.backbones.conv1d import Conv1DModel
from .cls_lit import BaseLit

logger = logging.getLogger(__name__)

class Conv1DLit(BaseLit):
    def __init__(
        self,
        num_classes=192,
        input_channels=2,
        lr=1e-4,
        weight_decay=0.000001,
        input_sequence_length=1570,
        regression=False,
    ):
        super().__init__(
            num_classes=num_classes,
            input_channels=input_channels,
            lr=lr,
            weight_decay=weight_decay,
            input_sequence_length=input_sequence_length,
            regression=regression, 
        )

    def _build_model(self, **kwargs) -> nn.Module:
        # BaseLit determines output_dim (1 if regression, num_classes if not)
        output_dim = kwargs.get("output_dim") 
        input_channels = kwargs.get("input_channels")
        input_sequence_length = kwargs.get("input_sequence_length")
        
        return Conv1DModel(
            output_dim=output_dim,
            input_channels=input_channels,
            input_sequence_length=input_sequence_length,
        )