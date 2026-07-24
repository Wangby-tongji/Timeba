# Copyright (c) 2020 Uber Technologies, Inc.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Canonical full Timeba architecture.

This module preserves the historical four-stage 64/128/256/512 architecture
and its PyTorch attribute names.  Dataset selection, feature selection,
training configuration, and evaluation are intentionally outside this module.
"""

from typing import Dict, List, Tuple

import torch
from torch import Tensor, nn

from .blocks import Conv1d, MambaBlock, Unet1d
from .interaction import A2A
from .prediction import PredNet


HIDDEN_STAGES = (64, 128, 256, 512)
ACTOR_DIM = 512
ACTOR2ACTOR_DISTANCE = 100.0


def gpu(data):
    """
    Transfer tensor in `data` to gpu recursively
    `data` can be dict, list or tuple
    """
    if isinstance(data, list) or isinstance(data, tuple):
        data = [gpu(x) for x in data]
    elif isinstance(data, dict):
        data = {key: gpu(_data) for key, _data in data.items()}
    elif isinstance(data, torch.Tensor):
        data = data.contiguous().cuda(non_blocking=True)
    return data


class Timeba(nn.Module):
    """Dataset-independent canonical full Timeba model."""

    def __init__(self, input_dim, pred_len, num_modes=6):
        super(Timeba, self).__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if pred_len <= 0:
            raise ValueError("pred_len must be positive")
        if num_modes <= 0:
            raise ValueError("num_modes must be positive")

        self.config = {
            "n_actor": ACTOR_DIM,
            "actor2actor_dist": ACTOR2ACTOR_DISTANCE,
            "num_preds": int(pred_len),
            "num_mods": int(num_modes),
        }

        # Attribute names and registration order are checkpoint-sensitive.
        self.actor_net = ActorNet(self.config, input_dim=int(input_dim))
        self.a2a = A2A(self.config)
        self.pred_net = PredNet(self.config)

    def forward(self, data: Dict) -> Dict[str, List[Tensor]]:
        actors, actor_idcs = actor_gather(gpu(data["feats"]))
        actor_ctrs = gpu(data["ctrs"])
        actors = self.actor_net(actors)
        actors = self.a2a(actors, actor_idcs, actor_ctrs)

        out = self.pred_net(actors, actor_idcs, actor_ctrs)
        rot, orig = gpu(data["rot"]), gpu(data["orig"])
        for i in range(len(out["reg"])):
            out["reg"][i] = torch.matmul(out["reg"][i], rot[i]) + orig[
                i
            ].view(1, 1, 1, -1)
        return out


def actor_gather(actors: List[Tensor]) -> Tuple[Tensor, List[Tensor]]:
    batch_size = len(actors)
    num_actors = [len(x) for x in actors]
    actors = [x.transpose(1, 2) for x in actors]
    actors = torch.cat(actors, 0)

    actor_idcs = []
    count = 0
    for i in range(batch_size):
        idcs = torch.arange(count, count + num_actors[i])
        actor_idcs.append(idcs)
        count += num_actors[i]
    return actors, actor_idcs


class ActorNet(nn.Module):
    def __init__(self, config, input_dim):
        super(ActorNet, self).__init__()
        self.config = config
        norm = "GN"
        ng = 1

        n_in = input_dim
        n_out = list(HIDDEN_STAGES)
        blocks = [MambaBlock, MambaBlock, MambaBlock, MambaBlock]
        num_blocks = [1, 1, 1, 1]

        groups = []
        for i in range(len(num_blocks)):
            group = []
            if i == 0:
                group.append(
                    blocks[i](n_in, n_out[i], norm=norm, ng=ng)
                )
            else:
                group.append(
                    blocks[i](
                        n_in,
                        n_out[i],
                        stride=2,
                        norm=norm,
                        ng=ng,
                    )
                )
            groups.append(nn.Sequential(*group))
            n_in = n_out[i]
        self.groups = nn.ModuleList(groups)

        n = config["n_actor"]
        lateral = []
        for i in range(len(n_out)):
            lateral.append(
                Conv1d(n_out[i], n, norm=norm, ng=ng, act=False)
            )
        self.lateral = nn.ModuleList(lateral)

        # Preserve all four registered entries, including the unused last one.
        self.Unet = nn.ModuleList()
        for i in range(len(n_out)):
            self.Unet.append(Unet1d(n_out[i], 512))

        self.output = MambaBlock(n, n, norm=norm, ng=ng)

    def forward(self, actors: Tensor) -> Tensor:
        out = actors
        outputs = []
        for i in range(len(self.groups)):
            out = self.groups[i](out)
            outputs.append(out)

        out = self.lateral[-1](outputs[-1])
        for i in range(len(self.groups) - 2, -1, -1):
            out = self.Unet[i](out, outputs[i])

        out = self.output(out)[:, :, -1]
        return out
