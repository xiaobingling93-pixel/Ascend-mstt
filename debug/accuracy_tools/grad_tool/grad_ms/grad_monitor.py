from grad_tool.common.base_monitor import BaseMonitor
from grad_tool.common.utils import print_info_log
from grad_tool.grad_ms.global_context import grad_context
from grad_tool.grad_ms.grad_analyzer import csv_generator
from grad_tool.grad_ms.hook import hook_optimizer


class MsGradientMonitor(BaseMonitor):

    def __init__(self, config_file: str):
        super(MsGradientMonitor, self).__init__(config_file)
        grad_context.init_context(self.config)
        csv_generator.init(grad_context)

    def monitor(self, module):
        hook_optimizer(module)

    def stop(self):
        csv_generator.stop()
