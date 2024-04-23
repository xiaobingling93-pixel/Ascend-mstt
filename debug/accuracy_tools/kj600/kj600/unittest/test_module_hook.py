import argparse
import torch_npu
import torch
import torch.nn.functional as F
from kj600.module_hook import TrainerMon # Modify PYTHONPATH to import TrainerMon
#from hook_api import reg_grad_hook, reg_grad_one_hook, reg_module_backward_hook, reg_module_forward_hook
#from torch.cuda.amp import GradScaler

from torch.npu.amp import GradScaler


# from ptdbg_ascend import PrecisionDebugger as PD
# from monitor import GradientMonitor

print(torch_npu.__version__)

#debugger = PD(dump_path="./dump/", hook_name="dump", step=[1, 2, 3], enable_dataloader=False)
#debugger.configure_hook(mode="list", scope=["optim_Adam_step"], )

parser = argparse.ArgumentParser(prog="kj600 debug", description="kj600 sample code", epilog="")
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

npu = torch.device('npu:0') 
net = Model().to(device=npu)

config = {
    "targets": {
        "fc": {"input": "tuple[2]:0", "output": "tensor::"}, 
        "relu": {"input": "..", "output": ".."}
    }
}
# reg_grad_hook(net, hook_factory=hook_factory, config=config)
# reg_grad_one_hook(net, hook=monitor_hook, config=config)
# net.fc.register_forward_hook(get_actv_hook("fc"))
# reg_module_forward_hook(net, module_fwd_hook, config)
# reg_module_backward_hook(net, module_bwd_hook, config)
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

hooker = TrainerMon('./kj600/unittest/config_1.json')
hooker.hook_modules(model=net, global_batch_size=2, dp=1, micro_batch_size=2, fwd_or_bkd=0, params_have_main_grad=False)
# hooker.hook_optimizer(optimizer)


class ToyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data = torch.randn(16, 784, dtype=DTYPE, requires_grad=True)
        self.labels = torch.randint(low=0, high=9, size=(16,))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx].to(npu), self.labels[idx].to(npu)

train_ds = ToyDataset()
train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, batch_size=2)


# scaler = GradScaler()
for (inputs, labels) in train_loader:
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = F.cross_entropy(outputs, labels)
    
    loss.backward()
    optimizer.step()
