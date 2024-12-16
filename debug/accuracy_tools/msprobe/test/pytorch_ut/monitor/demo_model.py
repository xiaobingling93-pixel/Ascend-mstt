import torch
import torch.nn.functional as F
from msprobe.pytorch import TrainerMon
from msprobe.pytorch.common import seed_all

device = torch.device('cpu')
dtype_float32 = torch.float32
seed_all(mode=True)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(784, 10, dtype=dtype_float32)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x).type(dtype_float32))


class ToyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data = torch.randn(16, 784, dtype=dtype_float32, requires_grad=True)
        self.labels = torch.randint(low=0, high=9, size=(16,))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx].to(device), self.labels[idx].to(device)


def monitor_demo(config: str = "./config/monitor_config.json"):
    net = Model().to(device=device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

    hooker = TrainerMon(
        config,
        params_have_main_grad=False,
        opt_ty='Megatron_FP32Optimizer'
    )
    hooker.monitor_gnorm_with_ad(
        model=net,
        grad_acc_steps=1,
        optimizer=optimizer
    )
    hooker.set_wrapped_optimizer(optimizer)

    train_ds = ToyDataset()
    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, batch_size=10)

    for (inputs, labels) in train_loader:
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = F.cross_entropy(outputs, labels)

        loss.backward()
        optimizer.step()

    hooker.summary_writer.close()
