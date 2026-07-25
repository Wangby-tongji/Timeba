# Copyright (c) 2020 Uber Technologies, Inc.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by the Timeba authors on 2026-07-25 during repository cleanup.
# Changes include extraction, modularization, and integration into the
# canonical Timeba reference implementation.

"""Actor-to-actor interaction modules from the historical full model."""

from math import gcd
from typing import List

import torch
from torch import Tensor, nn


class A2A(nn.Module):
    """
    The actor to actor block performs interactions among actors.
    """

    def __init__(self, config):
        super(A2A, self).__init__()
        self.config = config

        n_actor = config["n_actor"]
        att = []
        for i in range(2):
            att.append(Att(n_actor, n_actor))
        self.att = nn.ModuleList(att)

    def forward(
        self,
        actors: Tensor,
        actor_idcs: List[Tensor],
        actor_ctrs: List[Tensor],
    ) -> Tensor:
        for i in range(len(self.att)):
            actors = self.att[i](
                actors,
                actor_idcs,
                actor_ctrs,
                actors,
                actor_idcs,
                actor_ctrs,
                self.config["actor2actor_dist"],
            )
        return actors


class EncodeDist(nn.Module):
    def __init__(self, n, linear=True):
        super(EncodeDist, self).__init__()

        block = [nn.Linear(2, n), nn.ReLU(inplace=True)]
        if linear:
            block.append(nn.Linear(n, n))
        self.block = nn.Sequential(*block)

    def forward(self, dist):
        x, y = dist[:, :1], dist[:, 1:]
        dist = torch.cat(
            (
                torch.sign(x) * torch.log(torch.abs(x) + 1.0),
                torch.sign(y) * torch.log(torch.abs(y) + 1.0),
            ),
            1,
        )

        dist = self.block(dist)
        return dist


class Linear_dev(nn.Module):
    def __init__(self, n_in, n_out, norm="GN", ng=32, act=True):
        super(Linear_dev, self).__init__()
        assert norm in ["GN", "BN", "SyncBN"]

        self.linear = nn.Linear(n_in, n_out, bias=False)
        if norm == "GN":
            self.norm = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == "BN":
            self.norm = nn.BatchNorm1d(n_out)
        else:
            exit("SyncBN has not been added!")

        self.relu = nn.GELU()
        self.act = act

    def forward(self, x):
        out = self.linear(x)
        out = self.norm(out)
        if self.act:
            out = self.relu(out)
        return out


class Att(nn.Module):
    """
    Attention block to pass context nodes information to target nodes
    This is used to Actor2Map, Actor2Actor, Map2Actor and Map2Map
    """

    def __init__(self, n_agt, n_ctx):
        super(Att, self).__init__()
        norm = "GN"
        ng = 1

        self.dist = nn.Sequential(
            nn.Linear(2, n_ctx),
            nn.ReLU(inplace=True),
            Linear_dev(n_ctx, n_ctx, norm=norm, ng=ng),
        )
        self.query = Linear_dev(n_agt, n_ctx, norm=norm, ng=ng)
        self.GRN = GatedResdualNetwork(n_agt)
        self.agt = nn.Linear(n_agt, n_agt, bias=False)
        self.norm = nn.GroupNorm(gcd(ng, n_agt), n_agt)
        self.norm2 = nn.GroupNorm(gcd(ng, n_agt), n_agt)
        self.linear = nn.Sequential(
            nn.Linear(n_agt, n_agt * 4),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(n_agt * 4, n_agt),
        )
        self.relu = nn.GELU()

    def forward(
        self,
        agts,
        agt_idcs,
        agt_ctrs,
        ctx,
        ctx_idcs,
        ctx_ctrs,
        dist_th,
    ):
        res = agts
        if len(ctx) == 0:
            agts = self.agt(agts)
            agts = self.relu(agts)
            agts = self.linear(agts)
            agts += res
            agts = self.relu(agts)
            return agts

        batch_size = len(agt_idcs)
        hi, wi = [], []
        hi_count, wi_count = 0, 0
        for i in range(batch_size):
            dist = agt_ctrs[i].view(-1, 1, 2) - ctx_ctrs[i].view(1, -1, 2)
            dist = torch.sqrt((dist**2).sum(2))
            mask = (dist <= dist_th)

            idcs = torch.nonzero(mask, as_tuple=False)
            if len(idcs) == 0:
                continue

            hi.append(idcs[:, 0] + hi_count)
            wi.append(idcs[:, 1] + wi_count)
            hi_count += len(agt_idcs[i])
            wi_count += len(ctx_idcs[i])

        hi = torch.cat(hi, 0)
        wi = torch.cat(wi, 0)

        agt_ctrs = torch.cat(agt_ctrs, 0)
        ctx_ctrs = torch.cat(ctx_ctrs, 0)
        dist = agt_ctrs[hi] - ctx_ctrs[wi]
        dist = self.dist(dist)

        query = self.query(agts[hi])
        ctx = ctx[wi]
        ctx = torch.cat((dist, ctx), 1)
        ctx = self.GRN(query, ctx)

        agts = self.agt(agts)
        agts.index_add_(0, hi, ctx)

        agt_att = self.norm(agts)
        out = res + agt_att
        dense = self.linear(out)
        agts = res + self.norm2(dense)
        return agts


class GLU(nn.Module):
    def __init__(self, input_size):
        super(GLU, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, input_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        sig = self.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return torch.mul(sig, x)


class GatedResdualNetwork(nn.Module):
    def __init__(self, n_agt):
        super(GatedResdualNetwork, self).__init__()
        self.input_fc = nn.Linear(n_agt, n_agt)
        self.elu1 = nn.ELU()
        self.context = nn.Linear(n_agt * 2, n_agt)
        self.bn1 = nn.BatchNorm1d(n_agt)
        self.bn2 = nn.BatchNorm1d(n_agt)
        self.gate = GLU(n_agt)
        self.fc2 = nn.Linear(n_agt, n_agt)
        self.leakyrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x, context):
        res = self.input_fc(x)
        context = self.context(context)
        res = res + context

        res = self.bn1(res)
        res = self.elu1(res)
        res = self.fc2(res)
        res = self.gate(res)
        x = x + res
        x = self.bn2(x)
        x = self.leakyrelu(x)
        return x
