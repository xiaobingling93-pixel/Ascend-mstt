from api_accuracy_checker.compare.compare_utils import CompareConst


class CompareColumn:
    def __init__(self):
        self.bench_type = CompareConst.NA
        self.npu_type = CompareConst.NA
        self.shape = CompareConst.NA
        self.cosine_sim = CompareConst.NA
        self.max_abs_err = CompareConst.NA
        self.rel_err_hundredth = CompareConst.NA
        self.rel_err_thousandth = CompareConst.NA
        self.rel_err_ten_thousandth = CompareConst.NA
        self.error_rate = CompareConst.NA
        self.EB = CompareConst.NA
        self.RMSE = CompareConst.NA
        self.small_value_err_ratio = CompareConst.NA
        self.Max_rel_error = CompareConst.NA
        self.Mean_rel_error = CompareConst.NA

    def to_column_value(self, is_pass, message):
        return [self.bench_type, self.npu_type, self.shape, self.cosine_sim, self.max_abs_err, self.rel_err_hundredth,
                self.rel_err_thousandth, self.rel_err_ten_thousandth, self.error_rate, self.EB, self.RMSE, 
                self.small_value_err_ratio, self.Max_rel_error, self.Mean_rel_error, is_pass, message]