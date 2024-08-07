from msprobe.mindspore.grad_probe.global_context import grad_context
from msprobe.mindspore.grad_probe.grad_analyzer import csv_generator
from msprobe.mindspore.grad_probe.hook import hook_optimizer
from msprobe.core.grad_probe.constant import GradConst


class GradientMonitor:

    def __init__(self, common_dict, task_config):
        config = {}
        config[GradConst.OUTPUT_PATH] = common_dict.dump_path
        config[GradConst.STEP] = common_dict.step
        config[GradConst.RANK] = common_dict.rank
        config[GradConst.PARAM_LIST] = task_config.param_list
        config[GradConst.LEVEL] = task_config.grad_level
        config[GradConst.BOUNDS] = task_config.bounds
        self.config = config
        grad_context.init_context(self.config)

    @staticmethod
    def monitor(opt):
        csv_generator.init(grad_context)
        hook_optimizer(opt)

    @staticmethod
    def stop():
        csv_generator.stop()
