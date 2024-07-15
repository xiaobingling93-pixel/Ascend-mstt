import math
import statistics

from kj600.features import square_sum, get_max, get_min, get_zeros, get_nans, get_norm


def get_summary_writer_tag_name(module_or_param_name:str, tag:str, rank):
    if rank is None:
        return f"{module_or_param_name}/{tag}"
    else:
        return f"{module_or_param_name}/{rank}/{tag}"


# 用于存储所有metric实现类的注册表
config_metric_registry = {}


def register_config_metric(key, cls=None):
    """装饰器 用于注册Metric的实现类"""
    if cls is None:
        # 无参数时，返回装饰器函数
        return lambda cls: register_config_metric(key, cls)
    config_metric_registry[key] = cls
    return cls

class TensorMetrics:
    def __init__(self) -> None:
        self.metrics = {} #tensor_tag --> []
        self.cur_idx = {}

    fun_map = {"norm": get_norm, "max": get_max, "min": get_min}
    #get stats and insert into metrics dictionary
    def stat_insert(self, tensor, stat_ops, module_name, tensor_name, rank, eps=1e-8):
        prefix = get_summary_writer_tag_name(module_name, tensor_name, rank)
        for stat_op in stat_ops:
            y = TensorMetrics.fun_map[stat_op](tensor)
            key = f"{prefix}_{stat_op}"
            if key not in self.metrics:
                self.metrics[key] = []
                self.cur_idx[key] = 0
            self.metrics[key].append(y)

    def flush(self, tb_writer):
        for key, metric_list in self.metrics.items():
            start = self.cur_idx[key]
            for v in metric_list[start:]:
                tb_writer.add_scalar(key, v.item(), global_step=self.cur_idx[key])
                self.cur_idx[key] += 1

class Metric(object):
    @staticmethod
    def get_metric_value(tensor, eps):
        pass

    @staticmethod
    def metric_tensorboard(metric_name, summary_writer, metric_value, step):
        pass

    def get_metrics(self, tag2tensor: dict, eps):
        metrics_dict = {}
        for tag, tensor in tag2tensor.items():
            metrics_dict[tag] = self.get_metric_value(tensor, eps)
        return metrics_dict

@register_config_metric("min")
class MinMetric(Metric):
    @staticmethod
    def get_metric_value(tensor, eps):
        return get_min(tensor)

    @staticmethod
    def metric_tensorboard(metric_name, summary_writer, metric_value, step):
        for key in metric_value[0][metric_name].keys():
            min_value = min([item[metric_name][key].item() for item in metric_value])
            summary_writer.add_scalar(f'{key}_min', min_value, step)


@register_config_metric("max")
class MaxMetric(Metric):
    @staticmethod
    def get_metric_value(tensor, eps):
        return get_max(tensor)

    @staticmethod
    def metric_tensorboard(metric_name, summary_writer, metric_value, step):
        for key in metric_value[0][metric_name].keys():
            max_value = max([item[metric_name][key].item() for item in metric_value])
            summary_writer.add_scalar(f'{key}_max', max_value, step)


@register_config_metric("norm")
class NormMetric(Metric):
    @staticmethod
    def get_metric_value(tensor, eps):
        return square_sum(tensor)

    @staticmethod
    def metric_tensorboard(metric_name, summary_writer, metric_value, step):
        for key in metric_value[0][metric_name].keys():
            norm_value = math.sqrt(sum([item[metric_name][key].item() for item in metric_value]))
            summary_writer.add_scalar(f'{key}_norm', norm_value, step)


@register_config_metric("zeros")
class ZerosMetric(Metric):
    @staticmethod
    def get_metric_value(tensor, eps):
        return get_zeros(tensor, eps)

    @staticmethod
    def metric_tensorboard(metric_name, summary_writer, metric_value, step):
        for key in metric_value[0][metric_name].keys():
            zeros_value = statistics.mean([item[metric_name][key].item() for item in metric_value])
            summary_writer.add_scalar(f'{key}_zeros', zeros_value, step)

@register_config_metric("nans")
class NaNsMetric(Metric):
    @staticmethod
    def get_metric_value(t, eps):
        return get_nans(t)
    
    @staticmethod
    def metric_tensorboard(metric_name, summary_writer, metric_value, step):
        for key in metric_value[0][metric_name].keys():
            nans_value = sum([v[metric_name][key].item() for v in metric_value])
            summary_writer.add_scalar(f'{key}_nans', nans_value, step)

@register_config_metric("id")
class IdentMetric(Metric):
    @staticmethod
    def get_metric_value(tensor, eps):
        if tensor.dim() != 0:
            return None
        return tensor

    @staticmethod
    def metric_tensorboard(metric_name, summary_writer, metric_value, step): #metric_value is a dict, key is parameter name and value is a list of scalar tensor
        if len(metric_value) == 1:
            for key, value in metric_value[0][metric_name].items():
                if not value:
                    continue
                summary_writer.add_scalar(f'{key}_identical', value.item(), step)


def get_metrics(metric_name, tag2tensor, eps):
    try:
        fun_metric = config_metric_registry[metric_name]
        return fun_metric().get_metrics(tag2tensor, eps)
    except KeyError as e:
        raise ValueError(f"Not supported this metric, expected metric: {config_metric_registry.keys()}, actual metric: {metric_name}") from e


def write_metrics_tensorboard(metric_name, summary_writer, metric_value, step):
    try:
        fun_metric = config_metric_registry[metric_name]
        return fun_metric.metric_tensorboard(metric_name, summary_writer, metric_value, step)
    except KeyError as e:
        raise ValueError(f"Not supported this metric, expected metric: {config_metric_registry.keys()}, actual metric: {metric_name}") from e
