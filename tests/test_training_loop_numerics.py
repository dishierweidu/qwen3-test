import pytest
import torch
from torch.utils.data import DataLoader

from qwen3_omni_pretrain.training.loop import NonFiniteTrainingError, train_one_epoch


class FiniteLossInfiniteLogitsModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        loss = self.weight.square()
        logits = torch.full((x.shape[0], 2), float("inf"), device=x.device)
        return {"loss": loss, "logits": logits}


class MetadataRejectingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.seen_x = None

    def forward(self, x):
        self.seen_x = x.detach().clone()
        return {"loss": self.weight.square(), "logits": x * self.weight}


def test_nonfinite_logits_raise_before_optimizer_step():
    model = FiniteLossInfiniteLogitsModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = model.weight.detach().clone()
    dataloader = DataLoader(
        [{"x": torch.tensor([1.0]), "_sample_ids": ["sample-7"]}],
        batch_size=None,
    )

    with pytest.raises(NonFiniteTrainingError, match="sample-7"):
        train_one_epoch(model, dataloader, optimizer, None, torch.device("cpu"))

    assert torch.equal(model.weight.detach(), before)


def test_metadata_keys_are_not_forwarded_to_model():
    model = MetadataRejectingModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    dataloader = DataLoader(
        [{"x": torch.tensor([2.0]), "_sample_ids": ["ok"]}],
        batch_size=None,
    )

    loss = train_one_epoch(
        model, dataloader, optimizer, None, torch.device("cpu")
    )

    assert loss == pytest.approx(1.0)
    assert model.seen_x.tolist() == [2.0]


class FiniteOutputInfiniteGradientModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        loss = torch.sqrt(self.weight)
        return {"loss": loss, "logits": torch.zeros((x.shape[0], 2))}


def test_nonfinite_gradient_raises_before_optimizer_step():
    model = FiniteOutputInfiniteGradientModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    dataloader = DataLoader(
        [{"x": torch.tensor([1.0]), "_sample_ids": ["grad-bad"]}],
        batch_size=None,
    )

    with pytest.raises(NonFiniteTrainingError, match="gradient"):
        train_one_epoch(model, dataloader, optimizer, None, torch.device("cpu"))

    assert model.weight.item() == 0.0
