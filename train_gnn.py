import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
class MolGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = GATv2Conv(9, 256, heads=4, dropout=0.3)
        self.c2 = GATv2Conv(1024, 256, heads=4, dropout=0.3)
        self.out = nn.Linear(256, 1)
    def forward(self, data):
        x, ei, b = data.x, data.edge_index, data.batch
        x = F.elu(self.c1(x, ei)); x = F.elu(self.c2(x, ei))
        return torch.sigmoid(self.out(global_mean_pool(x, b)))
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = MolGNN().to(device)
    print(f"GNN on {torch.cuda.get_device_name(0)}, {sum(p.numel() for p in m.parameters())/1e6:.1f}M params")
