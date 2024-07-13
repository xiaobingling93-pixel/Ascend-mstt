class CodedException(Exception):
    def __init__(self, code, error_info=''):
        super().__init__()
        self.error_info = self.err_strs.get(code) + error_info

    def __str__(self):
        return self.error_info


class MsaccException(CodedException):
    INVALID_PARAM_ERROR = 0
    OVERFLOW_NUMS_ERROR = 1

    err_strs = {
        INVALID_PARAM_ERROR: "[msacc] 无效参数： ",
        OVERFLOW_NUMS_ERROR: "[msacc] 超过预设溢出次数 当前溢出次数:"
    }


class FileCheckException(CodedException):
    INVALID_FILE_ERROR = 0
    FILE_PERMISSION_ERROR = 1
    SOFT_LINK_ERROR = 2
    ILLEGAL_PATH_ERROR = 3
    ILLEGAL_PARAM_ERROR = 4
    FILE_TOO_LARGE_ERROR = 5

    err_strs = {
        SOFT_LINK_ERROR: "[msacc] 检测到软链接： ",
        FILE_PERMISSION_ERROR: "[msacc] 文件权限错误： ",
        INVALID_FILE_ERROR: "[msacc] 无效文件： ",
        ILLEGAL_PATH_ERROR: "[msacc] 非法文件路径： ",
        ILLEGAL_PARAM_ERROR: "[msacc] 非法打开方式： ",
        FILE_TOO_LARGE_ERROR: "[msacc] 文件过大： "
    }


class ParseJsonException(CodedException):
    UnexpectedNameStruct = 0
    InvalidDumpJson = 1
    err_strs = {
        UnexpectedNameStruct: "[msacc] Unexpected name in json: ",
        InvalidDumpJson: "[msacc] json格式不正确: ",
    }


class ScopeException(CodedException):
    InvalidApiStr = 0
    InvalidScope = 1
    ArgConflict = 2
    err_strs = {
        InvalidApiStr: "[msacc] Invalid api_list: ",
        InvalidScope: "[msacc] Invalid scope: ",
        ArgConflict: "[msacc] Scope and api_list conflict: ",
    }


class RepairException(CodedException):
    InvalidRepairType = 0
    err_strs = {
        InvalidRepairType: "[msacc] Invalid repair_type: "
    }


class StepException(CodedException):
    InvalidPostProcess = 0
    err_strs = {
        InvalidPostProcess: "[msacc] 错误的step后处理配置: ",
    }


class FreeBenchmarkException(CodedException):
    UnsupportedType = 0
    InvalidGrad = 1
    err_strs = {
        UnsupportedType: "[msacc] Free benchmark get unsupported type: ",
        InvalidGrad: "[msacc] Free benchmark gradient invalid: ",
    }


class DistributedNotInitializedError(Exception):
    def __init__(self, msg):
        super().__init__()
        self.msg = msg

    def __str__(self):
        return self.msg
