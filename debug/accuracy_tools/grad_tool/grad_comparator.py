import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from grad_tool.utils import write_csv, path_check, print_info_log


class GradComparator:
    @staticmethod
    def compare(path1: str, path2: str, output_dir):
        steps = GradComparator._get_matched_steps(path1, path2)
        if not steps:
            raise Exception("no matched steps for comparison, please dump data in same configuration")
        similarities = GradComparator._calculate_separated_similarities(path1, path2, steps)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        GradComparator._save_similarities(similarities, steps, output_dir)

    @staticmethod
    def _calculate_separated_similarities(path1, path2, steps):
        similarities = {}
        print_info_log(f"{len(steps)} steps will be compared")
        for step in tqdm(steps, desc="culculate similarities (by step)"):
            pt_files = GradComparator._get_matched_pt_files(path1, path2, step)
            same_count_summary = 0
            total_count_summary = 0
            for pt_file in pt_files:
                pt1 = os.path.join(path1, f"step_{step}", pt_file)
                pt2 = os.path.join(path2, f"step_{step}", pt_file)
                same_count, total_count = GradComparator._calculate_similarity(pt1, pt2)
                same_count_summary += same_count
                total_count_summary += total_count
                if pt_file not in similarities:
                    similarities[pt_file] = []
                if total_count == 0:
                    similarities[pt_file].append(0)
                else:
                    similarities[pt_file].append(same_count / total_count)
            if "summary" not in similarities:
                similarities["summary"] = []
            if total_count_summary == 0:
                similarities["summary"].append(0)
            else:
                similarities["summary"].append(same_count_summary / total_count_summary)
        return similarities

    @staticmethod
    def _get_matched_steps(path1: str, path2: str):
        path_check(path1, isdir=True)
        path_check(path2, isdir=True)
        steps = []
        for dirname in os.listdir(path1):
            splits = dirname.split('_')
            if not splits or splits[0] != 'step' or not splits[1].isdigit():
                continue

            folder2 = os.path.join(path2, dirname)
            if not os.path.exists(folder2):
                continue
            steps.append(int(splits[1]))
        steps = sorted(steps)
        return steps

    @staticmethod
    def _get_matched_pt_files(path1: str, path2: str, step: int):
        path1 = os.path.join(path1, f"step_{step}")
        path2 = os.path.join(path2, f"step_{step}")
        path_check(path1, isdir=True)
        path_check(path2, isdir=True)
        pt_files = []
        for dirname in os.listdir(path1):
            splits = dirname.split('.')
            if len(splits) < 1 or splits[-1] != 'pt':
                continue
            folder2 = os.path.join(path2, dirname)
            if not os.path.exists(folder2):
                continue
            pt_files.append(dirname)
        return sorted(pt_files)

    @staticmethod
    def _save_similarities(similarities: [float], steps: [int], output_dir: str):
        if not similarities:
            raise Exception(f"length of similarities is 0")
        for key, value in tqdm(similarities.items(), desc="save similarities (by param)"):
            if len(value) != len(steps):
                raise Exception(f"similarities length of {key}:{len(value)} not equal steps:{len(steps)}")
            plt.plot(steps, value)
            plt.xlabel('steps')
            plt.ylabel('similarities')
            plt.title(f'{key}_similarities')
            plt.savefig(f'{output_dir}/{key}_similarities.png')
            plt.savefig(os.path.join(output_dir, f"{key}_similarities.png"))
            plt.close()
            head_tuple = tuple(['step'] + [str(step) for step in steps])
            write_csv(os.path.join(output_dir, f"{key}_similarities.csv"), [['similarity'] + value], head_tuple)

    @staticmethod
    def _calculate_similarity(pt_file1: str, pt_file2: str):
        tensor1 = torch.load(pt_file1)
        tensor2 = torch.load(pt_file2)
        if tensor1.shape != tensor2.shape:
            raise Exception(f"tensor shape is not equal: {pt_file1}, {pt_file2}")
        if tensor1.dtype != torch.bool:
            raise Exception(f"tensor type is not bool: {pt_file1}")
        if tensor2.dtype != torch.bool:
            raise Exception(f"tensor type is not bool: {pt_file2}")
        same_count = (tensor1 == tensor2).sum().item()
        total_count = tensor1.numel()
        return same_count, total_count
