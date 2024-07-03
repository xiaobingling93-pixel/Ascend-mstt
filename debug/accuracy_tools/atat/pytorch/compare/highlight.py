import math
import abc
import numpy as np
from ...core.utils import CompareConst, get_header_index


class HighlightCheck(abc.ABC):
    @abc.abstractmethod
    def apply(self, info, color_columns, summary_compare):
        raise NotImplementedError


class CheckOrderMagnitude(HighlightCheck):
    """检查Max diff的数量级差异"""
    def apply(self, info, color_columns, summary_compare=True):
        api_in, api_out, num = info
        max_diff_index = get_header_index('Max diff' if summary_compare else 'MaxAbsErr', summary_compare)
        if abs(api_in[max_diff_index]) > abs(api_out[max_diff_index]):
            return
        in_order = 0 if abs(api_in[max_diff_index]) < 1 else math.log10(abs(api_in[max_diff_index]))
        out_order = 0 if abs(api_out[max_diff_index]) < 1 else math.log10(abs(api_out[max_diff_index]))
        if out_order - in_order >= CompareConst.ORDER_MAGNITUDE_DIFF_YELLOW:
            color_columns.yellow.append(num)


class CheckOneThousandErrorRatio(HighlightCheck):
    """检查千分误差比率"""
    def apply(self, info, color_columns, summary_compare=True):
        api_in, api_out, num = info
        one_thousand_index = get_header_index('One Thousandth Err Ratio', summary_compare)
        if not isinstance(api_in[one_thousand_index], (float, int)) or not isinstance(api_out[one_thousand_index], (float, int)):
            return
        if api_in[one_thousand_index] > CompareConst.ONE_THOUSAND_ERROR_IN_RED and api_out[one_thousand_index] < CompareConst.ONE_THOUSAND_ERROR_OUT_RED:
            color_columns.red.append(num)
        elif api_in[one_thousand_index] - api_out[one_thousand_index] > CompareConst.ONE_THOUSAND_ERROR_DIFF_YELLOW:
            color_columns.yellow.append(num)


class CheckCosineSimilarity(HighlightCheck):
    """检查余弦相似度"""
    def apply(self, info, color_columns, summary_compare=True):
        api_in, api_out, num = info
        cosine_index = get_header_index('Cosine', summary_compare)
        if not isinstance(api_in[cosine_index], (float, int)) or not isinstance(api_out[cosine_index], (float, int)):
            return
        if api_in[cosine_index] - api_out[cosine_index] > CompareConst.COSINE_DIFF_YELLOW:
            color_columns.yellow.append(num)


class CheckMaxRelativeDiff(HighlightCheck):
    """检查最大相对差异"""
    def apply(self, info, color_columns, summary_compare=True):
        api_in, api_out, num = info
        max_diff_index = get_header_index('Max diff', summary_compare)
        bench_max_index = get_header_index('Bench max', summary_compare)
        input_max_relative_diff = np.abs(np.divide(api_in[max_diff_index], max(0.01, api_in[bench_max_index])))
        output_max_relative_diff = np.abs(np.divide(api_out[max_diff_index], max(0.01, api_out[bench_max_index])))
        if not isinstance(input_max_relative_diff, (float, int)) or not isinstance(output_max_relative_diff,
                                                                                   (float, int)):
            return
        if output_max_relative_diff > CompareConst.MAX_RELATIVE_OUT_RED:
            color_columns.red.append(num)
        elif output_max_relative_diff > CompareConst.MAX_RELATIVE_OUT_YELLOW and input_max_relative_diff < CompareConst.MAX_RELATIVE_IN_YELLOW:
            color_columns.yellow.append(num)


class CheckOverflow(HighlightCheck):
    """检查是否存在溢出"""
    def apply(self, info, color_columns, summary_compare=True):
        line, num = info
        npu_max_index = get_header_index('NPU max', summary_compare)
        npu_min_index = get_header_index('NPU min', summary_compare)
        max_diff_index = get_header_index('Max diff' if summary_compare else 'MaxAbsErr', summary_compare)
        if str(line[npu_max_index]) in CompareConst.OVERFLOW_LIST or str(
                line[npu_min_index]) in CompareConst.OVERFLOW_LIST:
            color_columns.red.append(num)
            return
        # check if Max_Diff > 1e+10
        if isinstance(line[max_diff_index], (float, int)) and line[max_diff_index] > CompareConst.MAX_DIFF_RED:
            color_columns.red.append(num)


class HighlightRules:
    """高亮规则集合，用于检查API的误差"""
    # 适用于每行的规则
    basic_rules = {
        "check_overflow": CheckOverflow()
    }

    # 用于比较输入和输出的规则
    compare_rules = {
        "check_order_magnitude": CheckOrderMagnitude(),
        "check_one_thousand_error": CheckOneThousandErrorRatio(),
        "check_cosine_similarity": CheckCosineSimilarity()
    }
    summary_compare_rules = {
        "check_order_magnitude": CheckOrderMagnitude(),
        "check_max_relative_diff": CheckMaxRelativeDiff(),
    }
