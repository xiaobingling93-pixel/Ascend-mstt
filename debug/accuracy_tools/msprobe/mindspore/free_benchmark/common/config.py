from msprobe.mindspore.common.const import FreeBenchmarkConst


class Config:
    is_enable: bool = False
    handler_type = FreeBenchmarkConst.DEFAULT_HANDLER_TYPE
    pert_type = FreeBenchmarkConst.DEFAULT_PERT_TYPE
    stage = FreeBenchmarkConst.DEFAULT_STAGE
    dump_level = FreeBenchmarkConst.DEFAULT_DUMP_LEVEL
    steps: list = []
    ranks: list = []
    dump_path: str = ""
