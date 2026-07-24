"""Historical piecewise learning-rate behavior separated from root utilities."""

from dataclasses import dataclass

from torch import optim


@dataclass(frozen=True)
class PiecewiseConstantLR:
    learning_rates: tuple
    milestones: tuple

    def __post_init__(self):
        if len(self.learning_rates) != len(self.milestones) + 1:
            raise ValueError(
                "learning_rates must contain one more value than milestones"
            )

    def __call__(self, epoch):
        index = 0
        for milestone in self.milestones:
            if epoch < milestone:
                break
            index += 1
        return self.learning_rates[index]

    def apply(self, optimizer, epoch):
        learning_rate = self(epoch)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        return learning_rate


def build_optimizer(parameters, training_config):
    if training_config.optimizer != "adam":
        raise ValueError("the canonical pipeline currently supports Adam")
    return optim.Adam(
        parameters,
        lr=training_config.learning_rates[0],
        weight_decay=0,
    )
