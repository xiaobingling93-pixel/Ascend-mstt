from msprobe.core.common.const import Const


class DataProcessor:
    def __init__(self, data_frame):
        self.data_frame = data_frame
        if self.data_frame == Const.PT_FRAMEWORK:
            from msprobe.pytorch.compare.pt_diff_analyze import pt_diff_analyze
            self.process_func = pt_diff_analyze
        elif self.data_frame == Const.MS_FRAMEWORK:
            from msprobe.mindspore.compare.ms_diff_analyze import ms_diff_analyze
            self.process_func = ms_diff_analyze
        else:
            raise ValueError(f"Unsupported data_frame: {self.data_frame}")

    def process(self, npu_path, bench_path, output_path):
        return self.process_func(npu_path, bench_path, output_path)
