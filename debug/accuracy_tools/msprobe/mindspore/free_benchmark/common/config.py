from msprobe.core.common.const import MsFreeBenchmarkConst


class Config:
    is_enable: bool = False
    handler_type = MsFreeBenchmarkConst.DEFAULT_HANDLER_TYPE
    pert_type = MsFreeBenchmarkConst.DEFAULT_PERT_TYPE
    stage = MsFreeBenchmarkConst.DEFAULT_STAGE
    dump_level = MsFreeBenchmarkConst.DEFAULT_DUMP_LEVEL
    steps: list = []
    ranks: list = []
    dump_path: str = ""
