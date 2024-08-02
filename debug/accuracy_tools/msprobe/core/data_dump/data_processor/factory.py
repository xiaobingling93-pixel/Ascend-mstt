from msprobe.core.common.const import Const


class DataProcessorFactory:
    _data_processor = {}
    _module_processor = {}

    @classmethod
    def register_processor(cls, framework, task, processor_class):
        key = (framework, task)
        cls._data_processor[key] = processor_class

    @classmethod
    def register_module_processor(cls, framework, processor_class):
        cls._module_processor[framework] = processor_class

    @classmethod
    def get_module_processor(cls, framework):
        processor_class = cls._module_processor.get(framework)
        if not processor_class:
            raise ValueError(f"ModuleProcesser not found for framework: {framework}")
        return processor_class

    @classmethod
    def create_processor(cls, config, data_writer):
        cls.register_processors(config.framework)
        task = Const.KERNEL_DUMP if config.level == "L2" else config.task
        key = (config.framework, task)
        processor_class = cls._data_processor.get(key)
        if not processor_class:
            raise ValueError(f"Processor not found for framework: {config.framework}, task: {config.task}")
        return processor_class(config, data_writer)

    @classmethod
    def register_processors(cls, framework):
        if framework == Const.PT_FRAMEWORK:
            from .pytorch_processor import (
                StatisticsDataProcessor as PytorchStatisticsDataProcessor,
                TensorDataProcessor as PytorchTensorDataProcessor,
                OverflowCheckDataProcessor as PytorchOverflowCheckDataProcessor,
                FreeBenchmarkDataProcessor as PytorchFreeBenchmarkDataProcessor,
                KernelDumpDataProcessor as PytorchKernelDumpDataProcessor
            )
            from ....pytorch.module_processer import ModuleProcesser
            cls.register_processor(Const.PT_FRAMEWORK, Const.STATISTICS, PytorchStatisticsDataProcessor)
            cls.register_processor(Const.PT_FRAMEWORK, Const.TENSOR, PytorchTensorDataProcessor)
            cls.register_processor(Const.PT_FRAMEWORK, Const.OVERFLOW_CHECK, PytorchOverflowCheckDataProcessor)
            cls.register_processor(Const.PT_FRAMEWORK, Const.FREE_BENCHMARK, PytorchFreeBenchmarkDataProcessor)
            cls.register_processor(Const.PT_FRAMEWORK, Const.KERNEL_DUMP, PytorchKernelDumpDataProcessor)
            cls.register_module_processor(Const.PT_FRAMEWORK, ModuleProcesser)
        elif framework == Const.MS_FRAMEWORK:
            from .mindspore_processor import (
                StatisticsDataProcessor as MindsporeStatisticsDataProcessor,
                TensorDataProcessor as MindsporeTensorDataProcessor,
                OverflowCheckDataProcessor as MindsporeOverflowCheckDataProcessor
            )
            cls.register_processor(Const.MS_FRAMEWORK, Const.STATISTICS, MindsporeStatisticsDataProcessor)
            cls.register_processor(Const.MS_FRAMEWORK, Const.TENSOR, MindsporeTensorDataProcessor)
            cls.register_processor(Const.MS_FRAMEWORK, Const.OVERFLOW_CHECK, MindsporeOverflowCheckDataProcessor)
