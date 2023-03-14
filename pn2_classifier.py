import torch
from torch_geometric.nn import MLP, knn_interpolate, PointConv, global_max_pool, fps, radius
from torch.functional import F

from pn2_regressor import SAModule, GlobalSAModule, FPModule


class PN2_Classification(torch.nn.Module):
    def __init__(self, num_features, num_target_classes):
        super().__init__()

        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(0.2, 2, MLP([3 + num_features, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 8, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        # self.fp3_module = FPModule(1, MLP([1024 + 256, 256, 256]))
        # self.fp2_module = FPModule(3, MLP([256 + 128, 256, 128]))
        # self.fp1_module = FPModule(3, MLP([128 + num_features, 128, 128, 128]))

        self.mlp = MLP([1024, 512, 256, 128, num_target_classes], dropout=0.5,
                       batch_norm=True)
    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, _, _ = sa3_out

        # fp3_out = self.fp3_module(*sa3_out, *sa2_out)
        # fp2_out = self.fp2_module(*fp3_out, *sa1_out)
        # x, _, _ = self.fp1_module(*fp2_out, *sa0_out)

        return F.log_softmax(self.mlp(x), dim=-1)
