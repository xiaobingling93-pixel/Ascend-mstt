from abc import ABC, abstractmethod

import mindspore
import torch
import numpy as np

from msprobe.core.common.exceptions import ApiAccuracyCheckerException
from msprobe.core.common.log import logger
from msprobe.mindspore.api_accuracy_checker.const import (COSINE_SIMILARITY, MAX_ABSOLUTE_DIFF, MAX_RELATIVE_DIFF,
                                                          PASS, ERROR, SKIP)


class CompareResult:
    def __init__(self, compare_value, pass_status, err_msg):
        self.compare_value = compare_value
        self.pass_status = pass_status
        self.err_msg = err_msg


class BaseCompareAlgorithm(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.compare_algorithm_name = None

    def __call__(self, bench_compute_element, tested_compute_element):
        '''
        Args:
            bench_compute_element: ComputeElement
            tested_compute_element: ComputeElement

        Return:
            compare_result: CompareResult
        '''
        if self.check_validity(bench_compute_element, tested_compute_element):
            compare_value = self.run_compare(bench_compute_element, tested_compute_element)
            pass_status = self.check_pass(compare_value)
        else:
            logger.warning(f"not suitable for computing {self.compare_algorithm_name}, skip this.")
            compare_value = None
            pass_status = SKIP

        err_msg = self.generate_err_msg(pass_status)

        compare_result = CompareResult(compare_value, pass_status, err_msg)
        return compare_result

    @abstractmethod
    def check_validity(self, bench_compute_element, tested_compute_element):
        '''
        Args:
            bench_compute_element: ComputeElement
            tested_compute_element: ComputeElement

        Return:
            check_res: boolean
        '''
        raise NotImplementedError

    @abstractmethod
    def run_compare(self, bench_compute_element, tested_compute_element):
        '''
        Args:
            bench_compute_element: ComputeElement
            tested_compute_element: ComputeElement

        Return:
            compare_value: float/int
        '''
        raise NotImplementedError

    @abstractmethod
    def check_pass(self, compare_value):
        '''
        Args:
            compare_value: float/int

        Return:
            pass_status: str
        '''
        raise NotImplementedError

    @abstractmethod
    def generate_err_msg(self, pass_status):
        '''
        Args:
            pass_status: str

        Return:
            err_msg: str
        '''
        raise NotImplementedError

    @classmethod
    def convert_to_np_float64_ndarray(tensor):
        if isinstance(tensor, mindspore.Tensor):
            ndarray = tensor.astype(mindspore.float64).numpy()
        elif isinstance(tensor, torch.Tensor):
            ndarray = tensor.to(torch.float64, copy=True).numpy()
        else:
            err_msg = "BaseCompareAlgorithm.convert_to_np_float64_ndarray failed: " \
                "input is not mindspore.Tensor or torch.Tensor"
            logger.error_log_with_exp(err_msg, ApiAccuracyCheckerException(ApiAccuracyCheckerException.UnsupportType))
        return ndarray

    @classmethod
    def check_two_tensor(bench_compute_element, tested_compute_element):
        bench_parameter = bench_compute_element.get_parameter()
        tested_parameter = tested_compute_element.get_parameter()

        bench_is_tensor = isinstance(bench_parameter, (mindspore.Tensor, torch.Tensor))
        tested_is_tensosr = isinstance(tested_parameter, (mindspore.Tensor, torch.Tensor))
        shape_same = bench_compute_element.get_shape() == tested_compute_element.get_shape()
        return bench_is_tensor and tested_is_tensosr and shape_same


class CosineSimilarityCompareAlgorithm(BaseCompareAlgorithm):
    def __init__(self) -> None:
        super().__init__()
        self.compare_algorithm_name = COSINE_SIMILARITY
        self.pass_threshold = 0.9999

    def check_validity(self, bench_compute_element, tested_compute_element):
        return self.check_two_tensor(bench_compute_element, tested_compute_element)

    def run_compare(self, bench_compute_element, tested_compute_element):
        bench_ndarray = self.convert_to_np_float64_ndarray(bench_compute_element.get_parameter())
        tested_ndarray = self.convert_to_np_float64_ndarray(tested_compute_element.get_parameter())

        bench_norm = np.linalg.norm(bench_ndarray)
        tested_norm = np.linalg.norm(tested_ndarray)
        dot_product = np.dot(bench_ndarray.flatten(), tested_ndarray.flatten())
        cosine_similarity = dot_product / (bench_norm * tested_norm)
        return cosine_similarity

    def check_pass(self, compare_value):
        if compare_value > self.pass_threshold:
            return PASS
        else:
            return ERROR

    def generate_err_msg(self, pass_status):
        if pass_status == PASS:
            err_msg = ""
        elif pass_status == SKIP:
            err_msg = "two inputs are not valid for computing cosine similarity, skip comparing"
        elif pass_status == ERROR:
            err_msg = f"cosine similarity is less than threshold: {self.pass_threshold}"
        else:
            logger.warning(f"unseen pass_status: {pass_status}")
            err_msg = ""
        return err_msg


class MaxAbsoluteDiffCompareAlgorithm(BaseCompareAlgorithm):
    def __init__(self) -> None:
        super().__init__()
        self.compare_algorithm_name = MAX_ABSOLUTE_DIFF
        self.pass_threshold = 0.001

    def check_validity(self, bench_compute_element, tested_compute_element):
        return self.check_two_tensor(bench_compute_element, tested_compute_element)

    def run_compare(self, bench_compute_element, tested_compute_element):
        bench_ndarray = self.convert_to_np_float64_ndarray(bench_compute_element.get_parameter())
        tested_ndarray = self.convert_to_np_float64_ndarray(tested_compute_element.get_parameter())

        max_absolute_diff = np.max(np.abs(bench_ndarray - tested_ndarray))
        return max_absolute_diff

    def check_pass(self, compare_value):
        if compare_value < self.pass_threshold:
            return PASS
        else:
            return ERROR

    def generate_err_msg(self, pass_status):
        if pass_status == PASS:
            err_msg = ""
        elif pass_status == SKIP:
            err_msg = "two inputs are not valid for computing max absolute difference, skip comparing"
        elif pass_status == ERROR:
            err_msg = f"max absolute difference is greater than threshold: {self.pass_threshold}"
        else:
            logger.warning(f"unseen pass_status: {pass_status}")
            err_msg = ""
        return err_msg


class MaxRelativeDiffCompareAlgorithm(BaseCompareAlgorithm):
    def __init__(self) -> None:
        super().__init__()
        self.compare_algorithm_name = MAX_RELATIVE_DIFF
        self.pass_threshold = 0.01
        self.epsilon = 1e-8

    def check_validity(self, bench_compute_element, tested_compute_element):
        return self.check_two_tensor(bench_compute_element, tested_compute_element)

    def run_compare(self, bench_compute_element, tested_compute_element):
        bench_ndarray = self.convert_to_np_float64_ndarray(bench_compute_element.get_parameter())
        tested_ndarray = self.convert_to_np_float64_ndarray(tested_compute_element.get_parameter())

        abs_diff = np.abs(bench_ndarray - tested_ndarray)
        bench_ndarray_nonzero = bench_ndarray + (bench_ndarray == 0) * self.epsilon # prevent division by 0
        max_relative_diff = np.max(abs_diff / bench_ndarray_nonzero)
        return max_relative_diff

    def check_pass(self, compare_value):
        if compare_value < self.pass_threshold:
            return PASS
        else:
            return ERROR

    def generate_err_msg(self, pass_status):
        if pass_status == PASS:
            err_msg = ""
        elif pass_status == SKIP:
            err_msg = "two inputs are not valid for computing max relative difference, skip comparing"
        elif pass_status == ERROR:
            err_msg = f"max relative difference is greater than threshold: {self.pass_threshold}"
        else:
            logger.warning(f"unseen pass_status: {pass_status}")
            err_msg = ""
        return err_msg


compare_algorithms = {
    COSINE_SIMILARITY: CosineSimilarityCompareAlgorithm(),
    MAX_ABSOLUTE_DIFF: MaxAbsoluteDiffCompareAlgorithm(),
    MAX_RELATIVE_DIFF: MaxRelativeDiffCompareAlgorithm(),
}