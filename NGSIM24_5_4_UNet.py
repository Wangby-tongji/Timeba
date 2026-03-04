# Copyright (c) 2020 Uber Technologies, Inc.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import os
import sys
from math import gcd
from numbers import Number

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from layer.Transformer_EncDec import Encoder, EncoderLayer
from layer.SelfAttention_Family import ProbAttention, AttentionLayer
from layer.Embed import DataEmbedding_inverted, DataEmbedding
from data import ArgoDataset, collate_fn
from utils import gpu, to_long,  Optimizer, StepLR
from layers import Linear, LinearRes, LinearRes2
from numpy import float64, ndarray
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from copy import deepcopy
if torch.cuda.is_available():
    # 选择第一个可用的 GPU，您也可以根据需要更改设备编号
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU instead.")

file_path = os.path.abspath(__file__)
root_path = os.path.dirname(file_path)
model_name = os.path.basename(file_path).split(".")[0]

### config ###
config = dict()
"""Train"""
config["display_iters"] = 7491
config["val_iters"] = 7491 * 2
config["save_freq"] = 4.0
config["epoch"] = 0
config["horovod"] = False
config["opt"] = "adam"
config["num_epochs"] = 52
config["lr"] = [1e-3, 1e-4, 4e-5]
config["lr_epochs"] = [32, 42]

config["lr_func"] = StepLR(config["lr"], config["lr_epochs"])


if "save_dir" not in config:
    config["save_dir"] = os.path.join(
        root_path, "results", model_name
    )

if not os.path.isabs(config["save_dir"]):
    config["save_dir"] = os.path.join(root_path, "results", config["save_dir"])

config["batch_size"] = 16
config["val_batch_size"] = 16
config["workers"] = 0
config["val_workers"] = config["workers"]


"""Dataset"""
# Raw Dataset
root_path = "/home/ps/WorkSpaces/wby/Timeba/data"
config["train_split"] = os.path.join(root_path, "NGSIM/NGSIM6/train")
config["val_split"] = os.path.join(root_path, "NGSIM/NGSIM6/val")
config["test_split"] = os.path.join(root_path, "NGSIM/NGSIM6/test")

"""Model"""
config["rot_aug"] = False
config["pred_range"] = [-100.0, 100.0, -100.0, 100.0]
config["num_scales"] = 6
config["n_actor"] = 512
config["actor2actor_dist"] = 100.0
config["pred_size"] = 50
config["pred_step"] = 1
config["num_preds"] = config["pred_size"] // config["pred_step"]
config["num_mods"] = 6
config["cls_coef"] = 1.0
config["reg_coef"] = 1.0
config["mgn"] = 0.2
config["cls_th"] = 2.0
config["cls_ignore"] = 0.2
### end of config ###



class Net(nn.Module):
    """
    Lane Graph Network contains following components:
        1. ActorNet: a 1D CNN to process the trajectory input
        2. MapNet: LaneGraphCNN to learn structured map representations 
           from vectorized map data
        3. Actor-Map Fusion Cycle: fuse the information between actor nodes 
           and lane nodes:
            a. A2M: introduces real-time traffic information to 
                lane nodes, such as blockage or usage of the lanes
            b. M2M:  updates lane node features by propagating the 
                traffic information over lane graphs
            c. M2A: fuses updated map features with real-time traffic 
                information back to actors
            d. A2A: handles the interaction between actors and produces
                the output actor features
        4. PredNet: prediction header for motion forecasting using 
           feature from A2A
    """
    def __init__(self, config):
        super(Net, self).__init__()
        self.config = config

        self.actor_net = ActorNet(config)
        self.a2a = A2A(config)
        self.pred_net = PredNet(config)

    def forward(self, data: Dict) -> Dict[str, List[Tensor]]:
        # construct actor feature
        actors, actor_idcs = actor_gather(gpu(data["feats"]))
        actor_ctrs = gpu(data["ctrs"])
        actors = self.actor_net(actors)
        actors = self.a2a(actors, actor_idcs, actor_ctrs)

        # prediction
        out = self.pred_net(actors, actor_idcs, actor_ctrs)
        rot, orig = gpu(data["rot"]), gpu(data["orig"])
        # transform prediction to world coordinates
        for i in range(len(out["reg"])):
            out["reg"][i] = torch.matmul(out["reg"][i], rot[i]) + orig[i].view(
                1, 1, 1, -1
            )
        return out



def actor_gather(actors: List[Tensor]) -> Tuple[Tensor, List[Tensor]]:
    batch_size = len(actors)
    num_actors = [len(x) for x in actors]
    actors = [x.transpose(1, 2) for x in actors]

    actors = torch.cat(actors, 0)

    actor_idcs = []
    count = 0
    for i in range(batch_size):
        idcs = torch.arange(count, count + num_actors[i])#.to(actors.device)
        actor_idcs.append(idcs)
        count += num_actors[i]
    return actors, actor_idcs




class _ConvBlock1D(nn.Module):
    """(Conv1d -> GN -> GELU) * 2"""
    def __init__(self, c_in: int, c_out: int, groups: int = 1):
        super().__init__()
        g = max(1, min(groups, c_out))
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(g, c_out),
            nn.GELU(),
            nn.Conv1d(c_out, c_out, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(g, c_out),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class _Down1D(nn.Module):
    """Downsample by 2 with a strided conv + conv block."""
    def __init__(self, c_in: int, c_out: int, groups: int = 1):
        super().__init__()
        self.down = nn.Conv1d(c_in, c_out, kernel_size=4, stride=2, padding=1, bias=False)
        g = max(1, min(groups, c_out))
        self.norm = nn.GroupNorm(g, c_out)
        self.act = nn.GELU()
        self.block = _ConvBlock1D(c_out, c_out, groups=groups)

    def forward(self, x: Tensor) -> Tensor:
        x = self.down(x)
        x = self.norm(x)
        x = self.act(x)
        return self.block(x)


class _Up1D(nn.Module):
    """Upsample by 2 with a transposed conv, concat skip, then conv block."""
    def __init__(self, c_in: int, c_skip: int, c_out: int, groups: int = 1):
        super().__init__()
        self.up = nn.ConvTranspose1d(c_in, c_out, kernel_size=4, stride=2, padding=1, bias=False)
        self.block = _ConvBlock1D(c_out + c_skip, c_out, groups=groups)

    @staticmethod
    def _match_length(x: Tensor, ref: Tensor) -> Tensor:
        # Make x and ref have same temporal length (T) by cropping or padding.
        tx, tr = x.size(-1), ref.size(-1)
        if tx == tr:
            return x
        if tx > tr:
            return x[..., :tr]
        # pad right
        pad = tr - tx
        return F.pad(x, (0, pad))

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.up(x)
        x = self._match_length(x, skip)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class _UNet1D(nn.Module):
    """
    Pure 1D U-Net encoder/decoder over actor trajectories.
    Input:  (M, C_in, T)
    Output: (M, C_out, T)
    """
    def __init__(self, c_in: int, c_out: int = 512, base: int = 64, depth: int = 4, groups: int = 1):
        super().__init__()
        assert depth >= 2, "UNet depth should be >= 2"
        chs = [base * (2 ** i) for i in range(depth)]  # e.g., 64,128,256,512

        self.stem = _ConvBlock1D(c_in, chs[0], groups=groups)

        self.downs = nn.ModuleList()
        for i in range(1, depth):
            self.downs.append(_Down1D(chs[i - 1], chs[i], groups=groups))

        # bottleneck
        self.mid = _ConvBlock1D(chs[-1], chs[-1], groups=groups)

        self.ups = nn.ModuleList()
        for i in range(depth - 1, 0, -1):
            self.ups.append(_Up1D(chs[i], chs[i - 1], chs[i - 1], groups=groups))

        self.head = nn.Conv1d(chs[0], c_out, kernel_size=1, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        skips = []
        x = self.stem(x)
        skips.append(x)

        for down in self.downs:
            x = down(x)
            skips.append(x)

        x = self.mid(x)

        # decode: use skips in reverse order (excluding the last skip that matches x)
        for up, skip in zip(self.ups, reversed(skips[:-1])):
            x = up(x, skip)

        return self.head(x)


class ActorNet(nn.Module):
    """
    Actor feature extractor (Ablation): **pure 1D UNet**.
    Replaces the original Mamba + custom U-shape design.
    """
    def __init__(self, config):
        super(ActorNet, self).__init__()
        self.config = config

        # Keep interface identical to the original ActorNet
        c_in = 5
        c_out = config.get("n_actor", 512)

        # You can tune these 2 knobs for ablation
        base = config.get("unet_base", 64)
        depth = config.get("unet_depth", 4)

        # Use GroupNorm with 1 group (matches the original ng=1 default)
        self.unet = _UNet1D(c_in=c_in, c_out=c_out, base=base, depth=depth, groups=1)

    def forward(self, actors: Tensor) -> Tensor:
        # actors: (M, C, T)
        feats = self.unet(actors)          # (M, n_actor, T)
        return feats[:, :, -1]             # (M, n_actor)

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
    def forward(self, actors: Tensor, actor_idcs: List[Tensor], actor_ctrs: List[Tensor]) -> Tensor:
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

    def forward(self, actors: Tensor, actor_idcs: List[Tensor], actor_ctrs: List[Tensor]) -> Dict[str, List[Tensor]]:
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
        row_idcs = row_idcs.view(-1, 1).repeat(1, sort_idcs.size(1)).view(-1)
        sort_idcs = sort_idcs.view(-1)
        reg = reg[row_idcs, sort_idcs].view(cls.size(0), cls.size(1), -1, 2)

        out = dict()
        out["cls"], out["reg"] = [], []
        for i in range(len(actor_idcs)):
            idcs = actor_idcs[i]
            ctrs = actor_ctrs[i].view(-1, 1, 1, 2)
            out["cls"].append(cls[idcs])
            out["reg"].append(reg[idcs])
        return out


class Linear_dev(nn.Module):
    def __init__(self, n_in, n_out, norm='GN', ng=32, act=True):
        super(Linear_dev, self).__init__()
        assert (norm in ['GN', 'BN', 'SyncBN'])

        self.linear = nn.Linear(n_in, n_out, bias=False)

        if norm == 'GN':
            self.norm = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == 'BN':
            self.norm = nn.BatchNorm1d(n_out)
        else:
            exit('SyncBN has not been added!')

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
            nn.Linear(n_agt * 4, n_agt)
        )
        self.relu = nn.GELU()

    def forward(self, agts, agt_idcs, agt_ctrs, ctx, ctx_idcs, ctx_ctrs,
                dist_th):
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
    # Gated Linear Unit
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
        self.context = nn.Linear(n_agt*2, n_agt)
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

    def forward(self, agts: Tensor, agt_ctrs: Tensor, dest_ctrs: Tensor) -> Tensor:
        n_agt = agts.size(1)
        num_mods = dest_ctrs.size(1)

        dist = (agt_ctrs.unsqueeze(1) - dest_ctrs).view(-1, 2)
        dist = self.dist(dist)
        agts = agts.unsqueeze(1).repeat(1, num_mods, 1).view(-1, n_agt)

        agts = torch.cat((dist, agts), 1)
        agts = self.agt(agts)
        return agts


class PredLoss(nn.Module):
    def __init__(self, config):
        super(PredLoss, self).__init__()
        self.config = config
        self.reg_loss = nn.SmoothL1Loss(reduction="sum")

    def forward(self, out: Dict[str, List[Tensor]], gt_preds: List[Tensor], has_preds: List[Tensor]) -> Dict[str, Union[Tensor, int]]:
        cls, reg = out["cls"], out["reg"]
        cls = torch.cat([x for x in cls], 0)
        reg = torch.cat([x for x in reg], 0)
        gt_preds = torch.cat([x for x in gt_preds], 0)
        has_preds = torch.cat([x for x in has_preds], 0)

        loss_out = dict()
        zero = 0.0 * (cls.sum() + reg.sum())
        loss_out["cls_loss"] = zero.clone()
        loss_out["num_cls"] = 0
        loss_out["reg_loss"] = zero.clone()
        loss_out["num_reg"] = 0

        num_mods, num_preds = self.config["num_mods"], self.config["num_preds"]
        # assert(has_preds.all())

        last = has_preds.float() + 0.1 * torch.arange(num_preds).float().to(
            has_preds.device
        ) / float(num_preds)
        max_last, last_idcs = last.max(1)
        mask = max_last > 1.0

        cls = cls[mask]
        reg = reg[mask]
        gt_preds = gt_preds[mask]
        has_preds = has_preds[mask]
        last_idcs = last_idcs[mask]

        row_idcs = torch.arange(len(last_idcs)).long().to(last_idcs.device)
        dist = []
        for j in range(num_mods):
            dist.append(
                torch.sqrt(
                    (
                        (reg[row_idcs, j, last_idcs] - gt_preds[row_idcs, last_idcs])
                        ** 2
                    ).sum(1)
                )
            )
        dist = torch.cat([x.unsqueeze(1) for x in dist], 1)
        min_dist, min_idcs = dist.min(1)
        row_idcs = torch.arange(len(min_idcs)).long().to(min_idcs.device)

        mgn = cls[row_idcs, min_idcs].unsqueeze(1) - cls
        mask0 = (min_dist < self.config["cls_th"]).view(-1, 1)
        mask1 = dist - min_dist.view(-1, 1) > self.config["cls_ignore"]
        mgn = mgn[mask0 * mask1]
        mask = mgn < self.config["mgn"]
        coef = self.config["cls_coef"]
        loss_out["cls_loss"] += coef * (
            self.config["mgn"] * mask.sum() - mgn[mask].sum()
        )
        loss_out["num_cls"] += mask.sum().item()

        reg = reg[row_idcs, min_idcs]
        coef = self.config["reg_coef"]
        loss_out["reg_loss"] += coef * self.reg_loss(
            reg[has_preds], gt_preds[has_preds]
        )
        loss_out["num_reg"] += has_preds.sum().item()
        return loss_out


class Loss(nn.Module):
    def __init__(self, config):
        super(Loss, self).__init__()
        self.config = config
        self.pred_loss = PredLoss(config)

    def forward(self, out: Dict, data: Dict) -> Dict:
        loss_out = self.pred_loss(out, gpu(data["gt_preds"]), gpu(data["has_preds"]))
        loss_out["loss"] = loss_out["cls_loss"] / (
            loss_out["num_cls"] + 1e-10
        ) + loss_out["reg_loss"] / (loss_out["num_reg"] + 1e-10)
        return loss_out


class PostProcess(nn.Module):
    def __init__(self, config):
        super(PostProcess, self).__init__()
        self.config = config

    def forward(self, out,data):
        post_out = dict()
        post_out["preds"] = [x[0:1].detach().cpu().numpy() for x in out["reg"]]
        post_out["gt_preds"] = [x[0:1].numpy() for x in data["gt_preds"]]
        post_out["has_preds"] = [x[0:1].numpy() for x in data["has_preds"]]
        return post_out

    def append(self, metrics: Dict, loss_out: Dict, post_out: Optional[Dict[str, List[ndarray]]]=None) -> Dict:
        if len(metrics.keys()) == 0:
            for key in loss_out:
                if key != "loss":
                    metrics[key] = 0.0

            for key in post_out:
                metrics[key] = []

        for key in loss_out:
            if key == "loss":
                continue
            if isinstance(loss_out[key], torch.Tensor):
                metrics[key] += loss_out[key].item()
            else:
                metrics[key] += loss_out[key]

        for key in post_out:
            metrics[key] += post_out[key]
        return metrics

    def display(self, metrics, dt, epoch, lr=None):
        """Every display-iters print training/val information"""
        if lr is not None:
            print("Epoch %3.3f, lr %.5f, time %3.2f" % (epoch, lr, dt))
        else:
            print(
                "************************* Validation, time %3.2f *************************"
                % dt
            )

        cls = metrics["cls_loss"] / (metrics["num_cls"] + 1e-10)
        reg = metrics["reg_loss"] / (metrics["num_reg"] + 1e-10)
        loss = cls + reg

        preds = np.concatenate(metrics["preds"], 0)
        gt_preds = np.concatenate(metrics["gt_preds"], 0)
        has_preds = np.concatenate(metrics["has_preds"], 0)
        ade1, fde1, ade, fde, min_idcs = pred_metrics(preds, gt_preds, has_preds)

        print(
            "loss %2.4f %2.4f %2.4f, ade1 %2.4f, fde1 %2.4f, ade %2.4f, fde %2.4f"
            % (loss, cls, reg, ade1, fde1, ade, fde)
        )
        print()


def pred_metrics(preds, gt_preds, has_preds):
    assert has_preds.all()
    preds = np.asarray(preds, np.float32)
    gt_preds = np.asarray(gt_preds, np.float32)

    """batch_size x num_mods x num_preds"""
    err = np.sqrt(((preds - np.expand_dims(gt_preds, 1)) ** 2).sum(3))

    ade1 = err[:, 0].mean()
    fde1 = err[:, 0, -1].mean()

    min_idcs = err[:, :, -1].argmin(1)
    row_idcs = np.arange(len(min_idcs)).astype(np.int64)
    err = err[row_idcs, min_idcs]
    ade = err.mean()
    fde = err[:, -1].mean()
    return ade1, fde1, ade, fde, min_idcs


def get_model():
    net = Net(config).cuda()
    # net = net.cuda()

    loss = Loss(config).cuda()
    post_process = PostProcess(config).cuda()

    params = net.parameters()
    opt = Optimizer(params, config)


    return config, ArgoDataset, collate_fn, net, loss, post_process, opt
