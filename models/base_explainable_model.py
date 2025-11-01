from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class ExplainableVulnerabilityModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_embedding_layer(self) -> nn.Module:
        pass

    @abstractmethod
    def get_reference_input_ids(self, input_ids: torch.Tensor, tokenizer) -> torch.Tensor:
        pass

    @abstractmethod
    def lig_forward(self, inputs: torch.Tensor, attention_mask: torch.Tensor = None):
        pass

    @abstractmethod
    def get_input_embeddings(self, input_ids: torch.Tensor):
        pass

    @abstractmethod
    def get_vuln_prediction(self, inputs_ids: torch.Tensor, **kwargs):
        pass