# Copyright (c) 2020 Uber Technologies, Inc.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by the Timeba authors on 2026-07-25 during repository cleanup.
# Changes include extraction, modularization, and integration into the
# canonical Timeba reference implementation.

"""Multimodal trajectory prediction head from the historical full model."""

from typing import Dict, List

import torch
from torch import Tensor, nn

from .blocks import Linear, LinearRes, LinearRes2


class PredNet(nn.Module):
    """
    Final motion forecasting with Linear Residual block
    """

    def __init__(self, config):
        super(PredNet, self).__init__()
        self.config = config
        norm = "GN"
        ng = 1

        n_actor = config["n_actor"]
        pred = []
        for i in range(config["num_mods"]):
            pred.append(
                nn.Sequential(
                    LinearRes(n_actor, n_actor, norm=norm, ng=ng),
                    nn.Linear(n_actor, 2 * config["num_preds"]),
                )
            )
        self.pred = nn.ModuleList(pred)
        self.att_dest = AttDest(n_actor)
        self.cls = LinearRes2(n_actor, 1)

    def forward(
        self,
        actors: Tensor,
        actor_idcs: List[Tensor],
        actor_ctrs: List[Tensor],
    ) -> Dict[str, List[Tensor]]:
        preds = []
        for i in range(len(self.pred)):
            preds.append(self.pred[i](actors))
        reg = torch.cat([x.unsqueeze(1) for x in preds], 1)
        reg = reg.view(reg.size(0), reg.size(1), -1, 2)

        for i in range(len(actor_idcs)):
            idcs = actor_idcs[i]
            ctrs = actor_ctrs[i].view(-1, 1, 1, 2)
            reg[idcs] = reg[idcs] + ctrs

        dest_ctrs = reg[:, :, -1].detach()
        feats = self.att_dest(actors, torch.cat(actor_ctrs, 0), dest_ctrs)
        cls = self.cls(feats).view(-1, self.config["num_mods"])

        cls, sort_idcs = cls.sort(1, descending=True)
        row_idcs = torch.arange(len(sort_idcs)).long().to(sort_idcs.device)
        row_idcs = (
            row_idcs.view(-1, 1).repeat(1, sort_idcs.size(1)).view(-1)
        )
        sort_idcs = sort_idcs.view(-1)
        reg = reg[row_idcs, sort_idcs].view(
            cls.size(0), cls.size(1), -1, 2
        )

        out = dict()
        out["cls"], out["reg"] = [], []
        for i in range(len(actor_idcs)):
            idcs = actor_idcs[i]
            ctrs = actor_ctrs[i].view(-1, 1, 1, 2)
            out["cls"].append(cls[idcs])
            out["reg"].append(reg[idcs])
        return out


class AttDest(nn.Module):
    def __init__(self, n_agt: int):
        super(AttDest, self).__init__()
        norm = "GN"
        ng = 1

        self.dist = nn.Sequential(
            nn.Linear(2, n_agt),
            nn.GELU(),
            Linear(n_agt, n_agt, norm=norm, ng=ng),
        )
        self.agt = Linear(2 * n_agt, n_agt, norm=norm, ng=ng)

    def forward(
        self,
        agts: Tensor,
        agt_ctrs: Tensor,
        dest_ctrs: Tensor,
    ) -> Tensor:
        n_agt = agts.size(1)
        num_mods = dest_ctrs.size(1)

        dist = (agt_ctrs.unsqueeze(1) - dest_ctrs).view(-1, 2)
        dist = self.dist(dist)
        agts = (
            agts.unsqueeze(1).repeat(1, num_mods, 1).view(-1, n_agt)
        )
        agts = torch.cat((dist, agts), 1)
        agts = self.agt(agts)
        return agts
