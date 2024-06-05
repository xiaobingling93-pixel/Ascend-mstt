from atat.pytorch.free_benchmark import FreeBenchmarkException
from atat.pytorch.free_benchmark.common.constant import PreheatConfig
from atat.pytorch.free_benchmark.common.utils import Tools
from atat.pytorch.free_benchmark.common.enums import HandlerType
from atat.pytorch.free_benchmark.common.params import HandlerParams
from atat.pytorch.free_benchmark.result_handlers.check_handler import CheckerHandler
from atat.pytorch.free_benchmark.result_handlers.preheat_handler import PreheatHandler
from atat.pytorch.free_benchmark.result_handlers.fix_handler import FixHandler


class FuzzHandlerFactory:

    result_handlers = {
        HandlerType.CHECK: CheckerHandler,
        HandlerType.FIX: FixHandler,
        HandlerType.PREHEAT: PreheatHandler,
    }

    @staticmethod
    def create(params: HandlerParams):
        if_preheat = params.preheat_config.get(PreheatConfig.IF_PREHEAT)
        if not if_preheat:
            handler = FuzzHandlerFactory.result_handlers.get(params.handler_type)
        else:
            handler = FuzzHandlerFactory.result_handlers.get(HandlerType.PREHEAT)
            # TODO
        if not handler:
            raise FreeBenchmarkException(
                FreeBenchmarkException.UnsupportedType,
                f"无标杆工具支持 [ {HandlerType.CHECK}、{HandlerType.FIX}] 形式",
            )
        return handler(params)
