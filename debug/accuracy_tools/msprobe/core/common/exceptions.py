class CodedException(Exception):
    def __init__(self, code, error_info=''):
        super().__init__()
        self.code = code
        self.error_info = self.err_strs.get(code) + error_info

    def __str__(self):
        return self.error_info


class MsprobeException(CodedException):
    INVALID_PARAM_ERROR = 0
    OVERFLOW_NUMS_ERROR = 1

    err_strs = {
        INVALID_PARAM_ERROR: "[msprobe] 无效参数：",
        OVERFLOW_NUMS_ERROR: "[msprobe] 超过预设溢出次数 当前溢出次数："
    }


class FileCheckException(CodedException):
    INVALID_FILE_ERROR = 0
    FILE_PERMISSION_ERROR = 1
    SOFT_LINK_ERROR = 2
    ILLEGAL_PATH_ERROR = 3
    ILLEGAL_PARAM_ERROR = 4
    FILE_TOO_LARGE_ERROR = 5

    err_strs = {
        SOFT_LINK_ERROR: "[msprobe] 检测到软链接： ",
        FILE_PERMISSION_ERROR: "[msprobe] 文件权限错误： ",
        INVALID_FILE_ERROR: "[msprobe] 无效文件： ",
        ILLEGAL_PATH_ERROR: "[msprobe] 非法文件路径： ",
        ILLEGAL_PARAM_ERROR: "[msprobe] 非法打开方式： ",
        FILE_TOO_LARGE_ERROR: "[msprobe] 文件过大： "
    }


class ParseJsonException(CodedException):
    UnexpectedNameStruct = 0
    InvalidDumpJson = 1
    err_strs = {
        UnexpectedNameStruct: "[msprobe] Unexpected name in json: ",
        InvalidDumpJson: "[msprobe] json格式不正确: ",
    }


class ScopeException(CodedException):
    InvalidApiStr = 0
    InvalidScope = 1
    ArgConflict = 2
    err_strs = {
        InvalidApiStr: "[msprobe] Invalid api_list: ",
        InvalidScope: "[msprobe] Invalid scope: ",
        ArgConflict: "[msprobe] Scope and api_list conflict: ",
    }


class RepairException(CodedException):
    InvalidRepairType = 0
    err_strs = {
        InvalidRepairType: "[msprobe] Invalid repair_type: "
    }


class StepException(CodedException):
    InvalidPostProcess = 0
    err_strs = {
        InvalidPostProcess: "[msprobe] 错误的step后处理配置: ",
    }


class FreeBenchmarkException(CodedException):
    UnsupportedType = 0
    InvalidGrad = 1
    err_strs = {
        UnsupportedType: "[msprobe] Free benchmark get unsupported type: ",
        InvalidGrad: "[msprobe] Free benchmark gradient invalid: ",
    }


class DistributedNotInitializedError(Exception):
    def __init__(self, msg):
        super().__init__()
        self.msg = msg

    def __str__(self):
        return self.msg

class ApiAccuracyCheckerException(CodedException):
    ParseJsonFailed = 0
    UnsupportType = 1
    WrongValue = 2
    ApiWrong = 3
    err_strs = {
        ParseJsonFailed: "[msprobe] Api Accuracy Checker parse json failed: ",
        UnsupportType: "[msprobe] Api Accuracy Checker get unsupported type: ",
        WrongValue: "[msprobe] Api Accuracy Checker get wrong value: ",
        ApiWrong: "[msprobe] Api Accuracy Checker something wrong with api: ",
    }