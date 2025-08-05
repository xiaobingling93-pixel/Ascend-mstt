from typing import TypedDict, Optional, Any


class ResultType(TypedDict):
    """
    Return type of the function.
    """
    success: bool
    data: Optional[Any]  
    error: Optional[str]
