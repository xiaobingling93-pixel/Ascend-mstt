import os
from typing import List
from abc import ABC, abstractmethod

from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

from grad_tool.common.constant import GradConst
from grad_tool.common.utils import write_csv, check_file_or_directory_path, print_info_log, create_directory


class BaseComparator(ABC):

    @classmethod
    def compare_distributed(cls, path1: str, path2: str, output_dir: str):
        ranks = cls._get_matched_dirs(path1, path2, "rank")
        print_info_log(f"the following ranks will be compared: {ranks}")
        if not ranks:
            raise RuntimeError("no matched ranks for comparison, please dump data in same configuration")
        if not os.path.isdir(output_dir):
            create_directory(output_dir)
        for rank in tqdm(ranks, desc="rank"):
            print_info_log(f"now comparing rank {rank}:")
            cls.compare(os.path.join(path1, f"rank_{rank}"),
                         os.path.join(path2, f"rank_{rank}"),
                         os.path.join(output_dir, f"rank_{rank}"))

    @classmethod
    def compare(cls, path1: str, path2: str, output_dir: str):
        steps = cls._get_matched_dirs(path1, path2, "step")
        if not steps:
            raise RuntimeError("no matched steps for comparison, please dump data in same configuration")
        similarities = cls._calculate_separated_similarities(path1, path2, steps)
        if not os.path.isdir(output_dir):
            create_directory(output_dir)
        cls._save_similarities(similarities, steps, output_dir)

    @classmethod
    def _get_matched_dirs(cls, path1: str, path2: str, dir_prefix):
        check_file_or_directory_path(path1, file_type=GradConst.DIR)
        check_file_or_directory_path(path2, file_type=GradConst.DIR)
        dirs = []
        for dirname in os.listdir(path1):
            splits = dirname.split('_')
            if not splits or splits[0] != dir_prefix or not splits[1].isdigit():
                continue

            folder2 = os.path.join(path2, dirname)
            if not os.path.isdir(folder2):
                continue
            dirs.append(int(splits[1]))
        dirs = sorted(dirs)
        return dirs

    @classmethod
    def _save_similarities(cls, similarities: List[float], steps: List[int], output_dir: str):
        if not similarities:
            raise ValueError(f"length of similarities is 0")
        for key, value in tqdm(similarities.items(), desc="save similarities (by param)"):
            if len(value) != len(steps):
                raise RuntimeError(f"similarities length of {key}:{len(value)} not equal steps:{len(steps)}")
            plt.plot(steps, value)
            plt.xlabel('steps')
            plt.ylabel('similarities')
            plt.title(f'{key}_similarities')
            picture_dir = os.path.join(output_dir, "similarities_picture")
            if not os.path.isdir(picture_dir):
                create_directory(picture_dir)
            plt.savefig(os.path.join(picture_dir, f"{key}_similarities.png"))
            plt.close()
            head_tuple = tuple(['step'] + [str(step) for step in steps])
            write_csv(os.path.join(output_dir, "similarities.csv"), [[key] + value], head_tuple)

    @staticmethod
    def _get_grad_weight_order(path1, path2):
        for summary_file in os.listdir(path1):
            if not summary_file.endswith(".csv"):
                continue
            if not os.path.exists(os.path.join(path2, summary_file)):
                continue
            summary_csv = pd.read_csv(os.path.join(path1, summary_file))
            return summary_csv["param_name"]
        raise RuntimeError("no matched grad_summary.csv for comparison, please dump data in same configuration")
    
    @staticmethod
    def _get_name_matched_grad_file(param_name, grad_files):
        for grad_file in grad_files:
            if param_name == grad_file[:grad_file.rfind('.')]:
                return grad_file
        raise RuntimeError("no matched grad_file for comparison, please dump data in same configuration")

    @classmethod
    def _calculate_separated_similarities(cls, path1, path2, steps):
        similarities = {}
        print_info_log(f"{len(steps)} steps will be compared")
        grad_weight_order = cls._get_grad_weight_order(path1, path2)
        for step in tqdm(steps, desc="culculate similarities (by step)"):
            grad_files = cls._get_matched_grad_files(path1, path2, step)
            same_count_summary = 0
            total_count_summary = 0
            for grad_name in grad_weight_order:
                grad_file = cls._get_name_matched_grad_file(grad_name, grad_files)
                grad1 = os.path.join(path1, f"step_{step}", grad_file)
                grad2 = os.path.join(path2, f"step_{step}", grad_file)
                same_count, total_count = cls._calculate_similarity(grad1, grad2)
                same_count_summary += same_count
                total_count_summary += total_count
                idx = grad_file.rfind(".")
                param_name = grad_file[:idx]
                if param_name not in similarities:
                    similarities[param_name] = []
                if total_count == 0:
                    similarities[param_name].append(0)
                else:
                    similarities[param_name].append(same_count / total_count)
            if GradConst.SUMMARY not in similarities:
                similarities[GradConst.SUMMARY] = []
            if total_count_summary == 0:
                similarities[GradConst.SUMMARY].append(0)
            else:
                similarities[GradConst.SUMMARY].append(same_count_summary / total_count_summary)
        return similarities

    @classmethod
    def _get_matched_grad_files(cls, path1: str, path2: str, step: int):
        path1 = os.path.join(path1, f"step_{step}")
        path2 = os.path.join(path2, f"step_{step}")
        check_file_or_directory_path(path1, file_type=GradConst.DIR)
        check_file_or_directory_path(path2, file_type=GradConst.DIR)
        grad_files = []
        for grad_file in os.listdir(path1):
            splits = grad_file.split('.')
            if len(splits) < 1 or splits[-1] not in GradConst.GRAD_FILE_SUFFIX:
                continue
            folder2 = os.path.join(path2, grad_file)
            if not os.path.exists(folder2):
                continue
            grad_files.append(grad_file)
        return sorted(grad_files)

    @classmethod
    def _calculate_similarity(cls, grad_file1: str, grad_file2: str):
        npy1, npy2 = cls._load_grad_files(grad_file1, grad_file2)
        same_count = (npy1 == npy2).sum()
        total_count = npy1.size
        return same_count, total_count

    @classmethod
    @abstractmethod
    def _load_grad_files(cls, grad_file1: str, grad_file2: str):
        raise NotImplementedError("_load_grad_files is not implemented.")
