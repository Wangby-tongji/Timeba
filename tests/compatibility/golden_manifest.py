"""Golden state-dict manifest for the historical full Timeba model.

The manifest describes ``NGSIM24_5_4.Net`` at the untouched ``orin`` commit:

* input channels: 5
* prediction steps: 50
* modes: 6
* actor width: 512

It contains no model weights.  The Mamba entries follow the parameter layout
exposed by ``mamba_ssm.modules.mamba_simple.Mamba`` used by the historical
source.  Runtime compatibility tests compare this source-derived manifest with
the real legacy model before accepting a refactor.
"""

from math import ceil, prod


SOURCE_COMMIT = "1114c0dd976b07914010827ddd26f4f43f9d70e6"
SOURCE_MODULE = "NGSIM24_5_4.Net"
INPUT_DIM = 5
HISTORY_LEN = 24
PRED_LEN = 50
NUM_MODES = 6
ACTOR_DIM = 512


def _build_state_dict_manifest():
    entries = []

    def add(key, *shape, parameter=True):
        entries.append(
            {
                "key": key,
                "shape": list(shape),
                "parameter": parameter,
            }
        )

    def conv1d(prefix, n_in, n_out, kernel_size, bias=False):
        add(prefix + "weight", n_out, n_in, kernel_size)
        if bias:
            add(prefix + "bias", n_out)

    def conv_transpose1d(prefix, n_in, n_out, kernel_size, bias=True):
        add(prefix + "weight", n_in, n_out, kernel_size)
        if bias:
            add(prefix + "bias", n_out)

    def linear(prefix, n_in, n_out, bias=True):
        add(prefix + "weight", n_out, n_in)
        if bias:
            add(prefix + "bias", n_out)

    def group_norm(prefix, channels):
        add(prefix + "weight", channels)
        add(prefix + "bias", channels)

    def batch_norm(prefix, channels):
        add(prefix + "weight", channels)
        add(prefix + "bias", channels)
        add(prefix + "running_mean", channels, parameter=False)
        add(prefix + "running_var", channels, parameter=False)
        add(prefix + "num_batches_tracked", parameter=False)

    def mamba(prefix, d_model):
        d_inner = 2 * d_model
        dt_rank = ceil(d_model / 16)

        # Mamba owns A_log and D directly, so they precede child-module keys.
        add(prefix + "A_log", d_inner, 16)
        add(prefix + "D", d_inner)
        linear(prefix + "in_proj.", d_model, 2 * d_inner, bias=False)
        add(prefix + "conv1d.weight", d_inner, 1, 4)
        add(prefix + "conv1d.bias", d_inner)
        linear(prefix + "x_proj.", d_inner, dt_rank + 32, bias=False)
        linear(prefix + "dt_proj.", dt_rank, d_inner)
        linear(prefix + "out_proj.", d_inner, d_model, bias=False)

    def mamba_block(prefix, n_in, n_out, stride):
        conv1d(prefix + "conv1.", n_in, n_out, 3)
        conv1d(prefix + "conv2.", n_out, n_out, 3)
        mamba(prefix + "mamba.", n_out)
        group_norm(prefix + "bn1.", n_out)
        group_norm(prefix + "bn2.", n_out)
        if stride != 1 or n_in != n_out:
            conv1d(prefix + "downsample.0.", n_in, n_out, 1)
            group_norm(prefix + "downsample.1.", n_out)

    def conv_block(prefix, n_in, n_out):
        conv1d(prefix + "conv.", n_in, n_out, 3)
        group_norm(prefix + "norm.", n_out)

    def unet_block(prefix, skip_channels):
        conv_transpose1d(prefix + "up.", 512, 512, 2)
        group_norm(prefix + "bn1.", 1024)
        conv1d(prefix + "adptconv.", skip_channels, 512, 3, bias=True)
        group_norm(prefix + "bn2.", 512)
        conv1d(prefix + "conv1.", 1024, 512, 3, bias=True)
        mamba(prefix + "mamba.", 512)
        group_norm(prefix + "bn3.", 512)

    def linear_dev(prefix, n_in, n_out):
        linear(prefix + "linear.", n_in, n_out, bias=False)
        group_norm(prefix + "norm.", n_out)

    def glu(prefix, channels):
        linear(prefix + "fc1.", channels, channels)
        linear(prefix + "fc2.", channels, channels)

    def gated_residual_network(prefix, channels):
        linear(prefix + "input_fc.", channels, channels)
        linear(prefix + "context.", 2 * channels, channels)
        batch_norm(prefix + "bn1.", channels)
        batch_norm(prefix + "bn2.", channels)
        glu(prefix + "gate.", channels)
        linear(prefix + "fc2.", channels, channels)

    def attention(prefix, channels):
        linear(prefix + "dist.0.", 2, channels)
        linear_dev(prefix + "dist.2.", channels, channels)
        linear_dev(prefix + "query.", channels, channels)
        gated_residual_network(prefix + "GRN.", channels)
        linear(prefix + "agt.", channels, channels, bias=False)
        group_norm(prefix + "norm.", channels)
        group_norm(prefix + "norm2.", channels)
        linear(prefix + "linear.0.", channels, 4 * channels)
        linear(prefix + "linear.2.", 4 * channels, channels)

    def linear_res(prefix, channels):
        linear(prefix + "linear1.", channels, channels, bias=False)
        linear(prefix + "linear2.", channels, channels, bias=False)
        group_norm(prefix + "norm1.", channels)
        group_norm(prefix + "norm2.", channels)

    def normalized_linear(prefix, n_in, n_out):
        linear(prefix + "linear.", n_in, n_out, bias=False)
        group_norm(prefix + "norm.", n_out)

    def destination_attention(prefix, channels):
        linear(prefix + "dist.0.", 2, channels)
        normalized_linear(prefix + "dist.2.", channels, channels)
        normalized_linear(prefix + "agt.", 2 * channels, channels)

    def linear_res2(prefix, n_in, n_out):
        linear(prefix + "linear1.", n_in, n_out, bias=False)
        linear(prefix + "linear2.", n_out, n_out, bias=False)
        if n_in != n_out:
            linear(prefix + "transform.", n_in, n_out, bias=False)

    stages = (64, 128, 256, 512)
    n_in = INPUT_DIM
    for index, n_out in enumerate(stages):
        stride = 1 if index == 0 else 2
        mamba_block(f"actor_net.groups.{index}.0.", n_in, n_out, stride)
        n_in = n_out

    for index, channels in enumerate(stages):
        conv_block(f"actor_net.lateral.{index}.", channels, ACTOR_DIM)

    # All four entries are registered historically, including the unused last.
    for index, channels in enumerate(stages):
        unet_block(f"actor_net.Unet.{index}.", channels)

    mamba_block("actor_net.output.", ACTOR_DIM, ACTOR_DIM, stride=1)

    attention("a2a.att.0.", ACTOR_DIM)
    attention("a2a.att.1.", ACTOR_DIM)

    for index in range(NUM_MODES):
        linear_res(f"pred_net.pred.{index}.0.", ACTOR_DIM)
        linear(
            f"pred_net.pred.{index}.1.",
            ACTOR_DIM,
            2 * PRED_LEN,
        )

    destination_attention("pred_net.att_dest.", ACTOR_DIM)
    linear_res2("pred_net.cls.", ACTOR_DIM, 1)
    return tuple(entries)


ORDERED_STATE_DICT = _build_state_dict_manifest()
ORDERED_STATE_DICT_KEYS = tuple(entry["key"] for entry in ORDERED_STATE_DICT)
STATE_DICT_SHAPES = {
    entry["key"]: tuple(entry["shape"]) for entry in ORDERED_STATE_DICT
}
TOTAL_PARAMETER_COUNT = sum(
    prod(entry["shape"]) for entry in ORDERED_STATE_DICT if entry["parameter"]
)
TRAINABLE_PARAMETER_COUNT = TOTAL_PARAMETER_COUNT
