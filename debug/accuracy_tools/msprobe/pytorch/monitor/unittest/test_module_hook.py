import sys

sys.path.append('./')
import argparse
import torch

try:
    import torch_npu

    device = torch.device('npu:0')
except ModuleNotFoundError:
    device = torch.device('cpu')
import torch.nn.functional as F
from msprobe.pytorch.monitor.module_hook import TrainerMon  # Modify PYTHONPATH to import TrainerMon

parser = argparse.ArgumentParser(prog="monitor debug", description="monitor sample code", epilog="")
parser.add_argument("-o", "--out_dir", type=str, default=".")
args = parser.parse_args()
DTYPE = torch.float32


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(784, 10, dtype=DTYPE)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x).type(DTYPE))


net = Model().to(device=device)

config = {
    "targets": {
        "fc": {"input": "tuple[2]:0", "output": "tensor::"},
        "relu": {"input": "..", "output": ".."}
    }
}

optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

hooker = TrainerMon('./monitor/unittest/config_1.json', opt_ty='Megatron_Float16OptimizerWithFloat16Params')
hooker.hook_modules(model=net, global_batch_size=2, dp=1, micro_batch_size=2, fwd_or_bkd=0, params_have_main_grad=False)


class ToyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data = torch.randn(16, 784, dtype=DTYPE, requires_grad=True)
        self.labels = torch.randint(low=0, high=9, size=(16,))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx].to(device), self.labels[idx].to(device)


train_ds = ToyDataset()
train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, batch_size=2)


for (inputs, labels) in train_loader:
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = F.cross_entropy(outputs, labels)
    
    loss.backward()
    optimizer.step()
