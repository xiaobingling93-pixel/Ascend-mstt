import unittest
import shutil
import torch
import json
import os
try:
    import torch_npu
    device = torch.device('npu:0')
except ModuleNotFoundError:
    device = torch.device('cpu')
from kj600.module_hook import TrainerMon

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from msprobe.core.common.file_check import FileOpen

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(784, 2)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))

class ToyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data = torch.randn(16, 784, requires_grad=True)
        self.labels = torch.randint(low=0, high=8, size=(16,))
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.data[idx].to(device), self.labels[idx].to(device)
def get_file_path():
    output_dir = os.environ.get("KJ600_OUTPUT_DIR")
    for root1, dirs, files in os.walk(output_dir):
        for root2, dir, file in os.walk(os.path.join(root1, dirs[-1])):
            return os.path.join(root2, file[0])

def get_config():
    os.environ["KJ600_OUTPUT_DIR"] = "./test_kj600_output"
    with FileOpen("config_basic_functions.json", 'r') as file:
        config_test = json.load(file)
    return config_test
def get_tensorbaord(event_file_path):
    tensorboard = EventAccumulator(event_file_path)
    tensorboard.Reload()
    tags = tensorboard.Tags()
    scalers_tag = []
    for tag in tags['scalars']:
        tag = tag.split('/')
        scalers_tag.append(tag[1])
    images_tag = []
    for tag in tags['images']:
        tag = tag.split('/')
        images_tag.append(tag[1])
    return scalers_tag, images_tag

def clean_output():
    folder_path = os.environ.get("KJ600_OUTPUT_DIR")
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)

def train():
    model = Model().to(device=device)
    hooker = TrainerMon('config_basic_functions.json', False,
                        opt_ty="Megatron_Float16OptimizerWithFloat16Params")  # or opt_ty=Megatron_DistributedOptimizer
    hooker.hook_modules(model=model, grad_acc_steps=1)

    train_ds = ToyDataset()
    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, batch_size=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

    for (inputs, targets) in train_loader:
        optimizer.zero_grad()
        # inputs and param torch.float32 -> torch.float16
        inputs = inputs.half()
        for param in model.parameters():
            param.data = param.data.half()
        # outputs torch.float32
        outputs = model(inputs)
        output = outputs[0]
        targets = targets.float()
        # loss torch.float16 -> torch.float32
        loss = torch.nn.functional.cross_entropy(output, targets)

        loss.backward()
        optimizer.step()

class TestKj600(unittest.TestCase):
    def __init__(self, method_name: str) -> None:
        super(TestKj600, self).__init__(method_name)
        self.config_test = get_config()
        self.event_file_path = None
        self.scalers_tag = None
        self.images_tag = None

    @classmethod
    def setUpClass(cls):
        train()

    def setUp(self):
        self.config_test = get_config()
        self.event_file_path = get_file_path()
        self.scalers_tag, self.images_tag = get_tensorbaord(self.event_file_path)

    def test_ops(self):
        if self.config_test["ops"]:
            for op in self.config_test.get("ops"):
                if op == "id":
                    assert any(op in item for item in self.scalers_tag) == self.config_test.get('mg_direction'), f"{op} in ops did not take effect"
                else:
                    assert any(op in item for item in self.scalers_tag), f"{op} in ops did not take effect"
            print("ops has taken effect")

    def test_ur_distribution(self):
        if self.config_test.get("ur_distribution"):
            assert any('adam_update' in item for item in self.images_tag) and any(
                'adam_ratio' in item for item in self.images_tag), "ur_distribution did not take effect"
            print("ur_distribution has taken effect")

    def test_xy_distribution(self):
        if self.config_test.get("xy_distribution"):
            assert any('input' in item for item in self.scalers_tag) and any(
                'output' in item for item in self.scalers_tag), "xy_distribution did not take effect"
            print("xy_distribution has taken effect")

    def test_mv_distribution(self):
        if self.config_test.get("mv_distribution"):
            assert any('exp_avg' in item for item in self.scalers_tag) and any(
                'exp_avg_sq' in item for item in self.scalers_tag), "mv_distribution did not take effect"
            print("mv_distribution has taken effect")

    def test_mg_direction(self):
        if self.config_test.get("mg_direction"):
            assert any('mg_direction' in item for item in self.scalers_tag), "mg_direction did not take effect"
            print("mg_direction has taken effect")

    def test_wg_distribution(self):
        if self.config_test.get("wg_distribution"):
            assert any('weight' in item for item in self.scalers_tag), "wg_distribution did not take effect"
            print("wg_distribution has taken effect")

    @classmethod
    def tearDownClass(cls) -> None:
        clean_output()


if __name__ == "__main__":
    unittest.main()
