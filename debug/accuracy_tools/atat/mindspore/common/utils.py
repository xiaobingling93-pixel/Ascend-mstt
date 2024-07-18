import mindspore
from atat.core.common.exceptions import DistributedNotInitializedError


def get_rank_if_initialized():
    if mindspore.communication.GlobalComm.INITED:
        return mindspore.communication.get_rank()
    else:
        raise DistributedNotInitializedError("mindspore distributed environment is not initialized")
