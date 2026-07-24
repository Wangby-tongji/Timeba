"""Small, explicit training operations shared by CLI and integration tests."""

import math

import torch


def _assert_finite_tensor(name, tensor):
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"{name} contains NaN or Inf")


def train_step(components, batch, epoch):
    """Run forward, historical Loss, backward, LR update, and Adam step."""
    model = components.model
    optimizer = components.optimizer
    model.train()
    optimizer.zero_grad()

    output = model(dict(batch))
    loss_out = components.loss(output, dict(batch))
    loss = loss_out["loss"]
    _assert_finite_tensor("loss", loss)
    loss.backward()

    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            _assert_finite_tensor(f"gradient {name}", parameter.grad)

    learning_rate = components.lr_schedule.apply(optimizer, epoch)
    if not math.isfinite(learning_rate):
        raise FloatingPointError("learning rate is NaN or Inf")
    optimizer.step()

    return {
        "output": output,
        "loss": loss.detach(),
        "cls_loss": loss_out["cls_loss"].detach(),
        "reg_loss": loss_out["reg_loss"].detach(),
        "num_cls": loss_out["num_cls"],
        "num_reg": loss_out["num_reg"],
        "learning_rate": learning_rate,
    }
