from msprobe.core.common.const import Const


class Runtime:
    step_count: int = 0
    rank_id: int = -1
    is_running: bool = False
    run_mode: str = Const.PYNATIVE_MODE
    current_iter: int = 0
    current_rank: None
