from grad_tool.common.constant import GradConst
from grad_tool.common.utils import print_warn_log


class GradientMonitor:

    def __init__(self, config_path, framework="PyTorch") -> None:
        self.framework = framework
        if self.framework not in GradConst.FRAMEWORKS:
            raise RuntimeError(f"{self.framework} is not supported! Choose from {GradConst.FRAMEWORKS}.")
        if self.framework == GradConst.PYTORCH:
            from grad_tool.grad_pt.grad_monitor import PtGradientMonitor as grad_monitor
        else:
            from grad_tool.grad_ms.grad_monitor import MsGradientMonitor as grad_monitor
        self.grad_monitor = grad_monitor(config_path)

    def monitor(self, module):
        self.grad_monitor.monitor(module)
