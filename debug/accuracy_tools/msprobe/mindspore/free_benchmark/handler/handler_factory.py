from msprobe.mindspore.common.log import logger
from msprobe.mindspore.free_benchmark.common.config import Config
from msprobe.mindspore.common.const import FreeBenchmarkConst
from .check_handler import CheckHandler
from .fix_handler import FixHandler


class HandlerFactory:
    result_handlers = {
        FreeBenchmarkConst.CHECK: CheckHandler,
        FreeBenchmarkConst.FIX: FixHandler,
    }

    @staticmethod
    def create(api_name: str):
        handler = HandlerFactory.result_handlers.get(Config.handler_type)
        if handler:
            return handler(api_name)
        else:
            logger.error(f"{Config.handler_type} is not supported.")
            raise Exception
