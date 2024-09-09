import os
import time
from typing import List, Tuple
import multiprocessing
from multiprocessing import Process

import numpy as np
import mindspore as ms
from mindspore.communication import get_rank
from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter

from msprobe.core.grad_probe.utils import ListCache
from msprobe.core.grad_probe.constant import GradConst
from msprobe.mindspore.common.log import logger
from msprobe.core.common.file_utils import (create_directory, check_file_or_directory_path,
                                            write_csv, remove_path, move_file, load_npy)
from msprobe.mindspore.grad_probe.global_context import grad_context, GlobalContext


def get_rank_id():
    try:
        rank_id = get_rank()
    except Exception as err:
        rank_id = 0
    return rank_id


@ms.jit
def grad_dump(dump_dir: str, g_name: str, dump_step: Parameter, grad: ms.Tensor, level: str, bounds: List):
    '''
    Dump gradient statistic data.
        level0: [step, max, min, norm, shape_dim, shape]
        level1: [step, max, min, norm, shape_dim, shape] + grad_bool_data
        level2: [step, max, min, norm, shape_dim, shape, dist_dim, dist] + grad_bool_data
    '''
    dump_path = os.path.join(dump_dir, g_name)
    dump_dir_path = dump_path + "_dir"
    save_op = ms.ops.TensorDump()

    grad_flat = grad.reshape(-1)
    max_val = grad_flat.max(axis=0).float()
    min_val = grad_flat.min(axis=0).float()
    norm_val = grad_flat.norm(ord=2).float()
    shape = grad.shape
    extrem_list = [dump_step[0].float(), max_val, min_val, norm_val]
    extrem_stat = ms.ops.stack(extrem_list)
    shape_list = [len(shape)] + list(shape)
    shape_stat = ms.Tensor(shape_list).float()
    level0_stat = ms.ops.concat((extrem_stat, shape_stat), axis=0)
    level_stat = level0_stat

    if level == GradConst.LEVEL2:
        zero_grad = (grad == 0).sum()
        dist_dim = ms.Tensor([len(bounds) + 2]).float()
        bucket_result = ms.ops.bucketize(grad.float(), bounds)
        bucket_result = bucket_result.astype(ms.int8)
        dist_stat = [(bucket_result == i).sum() for i in range(len(bounds) + 1)]
        dist_stat.append(zero_grad)
        dist_stat.append(ms.Tensor(1, dtype=ms.int64))  # make sure dist_stat is not empty
        dist_stat = ms.ops.stack(dist_stat, axis=0).float()
        level2_stat = ms.ops.concat((level0_stat, dist_dim, dist_stat), axis=0)
        level_stat = level2_stat

    save_op(dump_path, level_stat)
    if level == GradConst.LEVEL1 or level == GradConst.LEVEL2:
        grad_direction = grad > 0
        save_op(dump_dir_path, grad_direction)


class CSVGenerator(Process):

    def __init__(self) -> None:
        super().__init__()
        self.dump_dir = None
        self.save_dir = None
        self.level = GradConst.LEVEL0
        self.cache_list = ListCache()
        self.current_step = None
        self.stop_event = None
        self.last_finish = False
        self.bounds = [-0.1, 0.0, 0.1],

    def init(self, context: GlobalContext):
        rank_id = get_rank_id()
        output_path = context.get_context(GradConst.OUTPUT_PATH)
        self.level = context.get_context(GradConst.LEVEL)
        self.bounds = context.get_context(GradConst.BOUNDS)
        self.dump_dir = f"{output_path}/rank{rank_id}/Dump/"
        self.save_dir = f"{output_path}/rank{rank_id}/"
        self.current_step = None
        self.stop_event = multiprocessing.Event()
        self.last_finish = False

    def run(self):
        while True:
            if not os.path.exists(self.dump_dir):
                time.sleep(0.1)
                if self.stop_event.is_set():
                    break
                continue
            npy_files = os.listdir(self.dump_dir)
            npy_files.sort(key=lambda x: int(x.split("_")[0]))
            self.traverse_files(npy_files)
            empty = len(os.listdir(self.dump_dir)) == 0
            if self.stop_event.is_set() and empty and self.last_finish:
                break
        if os.path.exists(self.dump_dir):
            remove_path(self.dump_dir)

    def stop(self):
        self.stop_event.set()

    def traverse_files(self, npy_files: List):
        for npy_file in npy_files:
            file_path = os.path.join(self.dump_dir, npy_file)
            while not os.path.exists(file_path):
                time.sleep(0.01)
            check_file_or_directory_path(file_path)
            if GradConst.STEP_FINISH in npy_file:
                self.cache_list.flush()
                remove_path(file_path)
                self.last_finish = True
            elif file_path.split("_")[-1] == GradConst.DIR_SUFFIX:
                prefix_idx = len(npy_file.split("_")[0])
                new_name = npy_file[prefix_idx + 1:].replace("_" + GradConst.DIR_SUFFIX, "." + GradConst.NPY_SUFFIX)
                if not new_name:
                    raise RuntimeError("Invalid dump data name.")
                if self.current_step is None:
                    raise RuntimeError("Current record step is None.")
                step_dir = os.path.join(self.save_dir, f"step{self.current_step}")
                if not os.path.exists(step_dir):
                    create_directory(step_dir)
                dst_file = os.path.join(step_dir, new_name)
                move_file(file_path, dst_file)
                self.last_finish = False
            elif file_path.split(".")[-1] == GradConst.NPY_SUFFIX:
                stat_data = self.load_npy_data(file_path)
                if stat_data is None:
                    continue
                if not self.check_valid(stat_data):
                    remove_path(file_path)
                    continue
                step = int(stat_data[GradConst.STEP_IDX])
                update_step = self.current_step is None or step != self.current_step
                self.current_step = step
                if update_step:
                    self.create_csv_file()
                self.gen_csv_line(file_path, stat_data)
                remove_path(file_path)
                self.last_finish = False

    def check_valid(self, stat_data):
        level = grad_context.get_context(GradConst.LEVEL)
        try:
            shape_dim = int(stat_data[GradConst.SHAPE_DIM_IDX])
            if level == GradConst.LEVEL2:
                dist_dim = int(stat_data[shape_dim + GradConst.SHAPE_DIM_IDX + 1])
                length = shape_dim + dist_dim + 7
            else:
                length = shape_dim + 5
        except IndexError as err:
            return False
        if length != len(stat_data):
            return False
        return True

    def load_npy_data(self, file_path: str):
        stat_data = None
        max_try = 10
        while max_try:
            try:
                stat_data = load_npy(file_path)
                return stat_data
            except Exception as err:
                logger.warning(f"load numpy file failed, retry...")
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
        if self.level == GradConst.LEVEL2:
            csv_line.extend(self.get_dist_data(shape_dim, stat_data))
        csv_line.extend(self.get_extrem_data(shape_dim, stat_data))
        self.cache_list.append(csv_line)

    def get_dist_data(self, shape_dim: int, stat_data: np.ndarray):
        dist_data = stat_data[(shape_dim + GradConst.SHAPE_DIM_IDX + 2):-1]
        element_num = dist_data.sum() - dist_data[-1]
        if element_num != 0:
            dist_data = dist_data / element_num
        return list(dist_data)

    def get_extrem_data(self, shape_dim: int, stat_data: np.ndarray):
        extrem_data = list(stat_data[(GradConst.STEP_IDX + 1):(GradConst.STEP_IDX + 4)])
        shape_data = stat_data[(GradConst.SHAPE_DIM_IDX + 1):(GradConst.SHAPE_DIM_IDX + shape_dim + 1)]
        shape_data = list(shape_data.astype(int))
        extrem_data.append(shape_data)
        return extrem_data

    def create_csv_file(self):
        headers = ["Param_name"]
        if self.level == GradConst.LEVEL2:
            headers.extend(self.get_dist_header())
        headers.extend(self.get_extrem_headers())
        output_path = f"{self.save_dir}/grad_summary_{self.current_step}.csv"
        write_csv([headers], output_path)
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
