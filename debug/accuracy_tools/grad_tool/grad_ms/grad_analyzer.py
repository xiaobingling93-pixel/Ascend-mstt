import os
import shutil
import time
from typing import List, Tuple
from multiprocessing import Process 

import numpy as np
import mindspore as ms
from mindspore.communication import get_rank
from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter

from grad_tool.common.constant import GradConst
from grad_tool.common.utils import ListCache, print_warn_log
from grad_tool.common.utils import create_directory, check_file_or_directory_path, write_csv
from grad_tool.grad_ms.global_context import grad_context
from grad_tool.grad_ms.global_context import GlobalContext


def get_rank_id():
    try:
        rank_id = get_rank()
    except Exception as err:
        rank_id = 0
    return rank_id


class GradAnalyzer:

    @staticmethod
    def dump(dump_dir: str, g_name: str, dump_step: Parameter, grad: ms.Tensor):
        '''
        Dump gradient statistic data.
            level0: [step, max, min, norm, shape_dim, shape]
            level1: [step, max, min, norm, shape_dim, shape, dist_dim, dist]
            level2: [step, max, min, norm, shape_dim, shape] + grad_bool_data
            level3: [step, max, min, norm, shape_dim, shape, dist_dim, dist] + grad_bool_data
        '''
        dump_path = os.path.join(dump_dir, g_name)
        dump_dir_path = dump_path + "_dir"
        save_op = ms.ops.TensorDump()
        level = grad_context.get_context(GradConst.LEVEL)

        if level == GradConst.LEVEL0 or level == GradConst.LEVEL2:
            level_stat = GradAnalyzer.calculate_level0(dump_step, grad)
        else:
            level_stat = GradAnalyzer.calculate_level1(dump_step, grad)

        save_op(dump_path, level_stat)
        if level == GradConst.LEVEL2 or level == GradConst.LEVEL3:
            grad_direction = GradAnalyzer.calculate_direction(grad)
            save_op(dump_dir_path, grad_direction)

    @staticmethod
    def calculate_level0(dump_step: Parameter, grad: ms.Tensor):
        is_bf16 = grad.dtype
        max_val = grad.max().float() if is_bf16 else grad.max()
        min_val = grad.min().float() if is_bf16 else grad.min()
        norm_val = grad.norm().float() if is_bf16 else grad.norm()
        shape = grad.shape
        extrem_stat = ms.ops.stack([dump_step[0].astype(max_val.dtype), max_val, min_val, norm_val])
        shape_stat = ms.Tensor([len(shape)] + list(shape)).astype(max_val.dtype)
        level0_stat = ms.ops.concat((extrem_stat, shape_stat), axis=0)
        return level0_stat

    @staticmethod
    def calculate_level1(dump_step: Parameter, grad: ms.Tensor):
        level0_stat = GradAnalyzer.calculate_level0(dump_step, grad)
        bounds = grad_context.get_context(GradConst.BOUNDS)
        zero_grad = (grad == 0).sum()
        dist_dim = ms.Tensor([len(bounds) + 2]).astype(level0_stat.dtype)
        bucket_result = ms.ops.bucketize(grad, bounds).astype(ms.int8)
        dist_stat = [(bucket_result == i).sum() for i in range(len(bounds) + 1)]
        dist_stat.append(zero_grad)
        dist_stat = ms.ops.stack(dist_stat, axis=0).astype(level0_stat.dtype)
        element_num = dist_stat.sum() - dist_stat[-1]
        if element_num != 0:
            dist_stat = dist_stat / element_num
        level1_stat = ms.ops.concat((level0_stat, dist_dim, dist_stat), axis=0)
        return level1_stat

    @staticmethod
    def calculate_direction(grad: ms.Tensor):
        return grad > 0


class CSVGenerator(Process):

    def __init__(self) -> None:
        super().__init__()
        self.dump_dir = None
        self.save_dir = None
        self.level = GradConst.LEVEL0
        self.cache_list = ListCache()
        self.current_step = None
        self.bounds = [-10, -1, -0.1, -0.01, -0.001, 0, 0.001, 0.01, 0.1, 1, 10],

    def init(self, context: GlobalContext):
        rank_id = get_rank_id()
        output_path = context.get_context(GradConst.OUTPUT_PATH)
        self.level = context.get_context(GradConst.LEVEL)
        self.bounds = context.get_context(GradConst.BOUNDS)
        step_range = context.get_context(GradConst.STEP)
        self.step_end = 0 if step_range is None else step_range[1]
        self.dump_dir = f"{output_path}/rank_{rank_id}/Dump/"
        self.save_dir = f"{output_path}/rank_{rank_id}/"
        self.current_step = None
        self.finish_flag = False

    def run(self):
        while not self.finish_flag:
            if not os.path.exists(self.dump_dir):
                time.sleep(0.1)
                continue
            npy_files = os.listdir(self.dump_dir)
            npy_files.sort(key=lambda x: int(x.split("_")[0]))
            if not npy_files:
                continue
            self.traverse_files(npy_files)
        shutil.rmtree(self.dump_dir)

    def traverse_files(self, npy_files: List):
        for npy_file in npy_files:
            file_path = os.path.join(self.dump_dir, npy_file)
            while not os.path.exists(file_path):
                time.sleep(0.01)
            check_file_or_directory_path(file_path)
            if GradConst.STEP_FINISH in npy_file:
                self.cache_list.flush()
                os.remove(file_path)
                if self.current_step == self.step_end:
                    self.finish_flag = True
            elif file_path.split("_")[-1] == GradConst.DIR_SUFFIX:
                prefix_idx = len(npy_file.split("_")[0])
                new_name = npy_file[prefix_idx + 1:].replace("_" + GradConst.DIR_SUFFIX, "." + GradConst.NPY_SUFFIX)
                if not new_name:
                    raise RuntimeError("Invalid dump data name.")
                if self.current_step is None:
                    raise RuntimeError("Current record step is None.")
                step_dir = os.path.join(self.save_dir, f"step_{self.current_step}")
                if not os.path.exists(step_dir):
                    create_directory(step_dir)
                dst_file = os.path.join(step_dir, new_name)
                shutil.move(file_path, dst_file)
            elif file_path.split(".")[-1] == GradConst.NPY_SUFFIX:
                stat_data = self.load_npy_data(file_path)
                if stat_data is None:
                    continue
                step = int(stat_data[GradConst.STEP_IDX])
                if self.current_step is None or step != self.current_step:
                    self.current_step = step
                    self.create_csv_file()
                self.gen_csv_line(file_path, stat_data)
                os.remove(file_path)

    def load_npy_data(self, file_path: str):
        stat_data = None
        max_try = 10
        while max_try:
            try:
                stat_data = np.load(file_path)
                return stat_data
            except Exception as err:
                print_warn_log(f"load numpy file failed, retry...")
                max_try -= 1
                time.sleep(0.1)
        return stat_data

    def gen_csv_line(self, file_path: str, stat_data) -> None:
        shape_dim = int(stat_data[GradConst.SHAPE_DIM_IDX])
        file_name = os.path.basename(file_path)
        prefix_idx = len(file_name.split("_")[0])
        param_name = file_name[(prefix_idx + 1) : -(len(GradConst.NPY_SUFFIX) + 1)]
        if not param_name:
            raise RuntimeError("Invalid gradient statistic file name.")
        csv_line = [param_name]
        if self.level == GradConst.LEVEL1 or self.level == GradConst.LEVEL3:
            csv_line.extend(self.get_dist_data(shape_dim, stat_data))
        csv_line.extend(self.get_extrem_data(shape_dim, stat_data))
        self.cache_list.append(csv_line)

    def get_dist_data(self, shape_dim: int, stat_data: np.ndarray):
        return list(stat_data[(shape_dim + GradConst.SHAPE_DIM_IDX + 2):])

    def get_extrem_data(self, shape_dim: int, stat_data: np.ndarray):
        extrem_data = list(stat_data[(GradConst.STEP_IDX + 1):(GradConst.STEP_IDX + 4)])
        shape_data = stat_data[(GradConst.SHAPE_DIM_IDX + 1):(GradConst.SHAPE_DIM_IDX + shape_dim + 1)]
        shape_data = list(shape_data.astype(int))
        extrem_data.append(shape_data)
        return extrem_data

    def create_csv_file(self):
        headers = ["Param_name"]
        if self.level == GradConst.LEVEL1 or self.level == GradConst.LEVEL3:
            headers.extend(self.get_dist_header())
        headers.extend(self.get_extrem_headers())
        output_path = f"{self.save_dir}/grad_summary_{self.current_step}.csv"
        write_csv(output_path, [], headers)
        self.cache_list.set_output_file(output_path)
        self.cache_list.clear()

    def get_extrem_headers(self) -> List[str]:
        return ["Max", "Min", "Norm", "Shape"]

    def get_dist_header(self) -> List[str]:
        intervals = []
        for i, _ in enumerate(self.bounds):
            if i == 0:
                intervals.append(f"(-inf, {self.bounds[i]}]")
            else:
                intervals.append(f"({self.bounds[i-1]}, {self.bounds[i]}]")
        intervals.extend([f"({self.bounds[-1]}, inf)", "=0"])
        return intervals

csv_generator = CSVGenerator()
