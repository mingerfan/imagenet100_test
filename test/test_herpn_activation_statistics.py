import torch
import torch.nn as nn

from fhe_statistics.activation_configs import ACTIVATION_CONFIGS
from fhe_statistics.statistics_fn import FheInfo
from models.gate_net_cmp.block_def import HermitePoly4, SwishHerPN


class HerPNActivationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.act1 = HermitePoly4(warmup_epochs=0)
        self.act1.set_poly_schedule(start_epoch=0, transition_epochs=0)
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1)
        self.act2 = SwishHerPN()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8, 4)

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def test_herpn_activations_are_leaf_and_counted():
    assert "poly4_herpn" in ACTIVATION_CONFIGS
    assert "hermitepoly4" in ACTIVATION_CONFIGS
    assert "swish_herpn" in ACTIVATION_CONFIGS

    fhe_info = FheInfo(
        HerPNActivationModel(),
        input_shape=(1, 3, 16, 16),
        model_name="HerPNActivationModel",
    )

    module_types = []
    for node in fhe_info.traced.graph.nodes:
        if node.op == "call_module":
            module_types.append(type(fhe_info.traced.get_submodule(str(node.target))).__name__)

    assert module_types.count("HermitePoly4") == 1
    assert module_types.count("SwishHerPN") == 1

    fhe_info.run_statistics()
    assert fhe_info.op_stats["poly4_herpn"]["count"] == 1
    assert fhe_info.op_stats["swish_herpn"]["count"] == 1

    op_types = {meta.op_type for meta in fhe_info.node_meta_list.values()}
    assert "unknown_module_HermitePoly4" not in op_types
    assert "unknown_module_SwishHerPN" not in op_types
