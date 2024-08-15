from typing import Optional, Any, Tuple, Dict, Callable


class HandlerParams:
    """
    参数结合体

    """
    args: Optional[Tuple] = None
    kwargs: Optional[Dict] = None
    index: Optional[int] = None
    original_result: Optional[Any] = None
    fuzzed_result: Optional[Any] = None
    is_consistent: Optional[bool] = True
    save_flag: Optional[bool] = True
    fuzzed_value: Optional[Any] = None
    original_func: Optional[Callable] = None
