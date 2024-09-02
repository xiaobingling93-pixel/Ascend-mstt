from msprobe.mindspore.common.const import FreeBenchmarkConst
from msprobe.mindspore.free_benchmark.common.config import Config
from .add_noise import AddNoisePerturbation
from .bit_noise import BitNoisePerturbation
from .no_change import NoChangePerturbation
from .improve_precision import ImprovePrecisionPerturbation
from .exchange_value import ExchangeValuePerturbation


class PerturbationFactory:
    """
    扰动工厂类

    """
    perturbations = {
        FreeBenchmarkConst.IMPROVE_PRECISION: ImprovePrecisionPerturbation,
        FreeBenchmarkConst.ADD_NOISE: AddNoisePerturbation,
        FreeBenchmarkConst.BIT_NOISE: BitNoisePerturbation,
        FreeBenchmarkConst.NO_CHANGE: NoChangePerturbation,
        FreeBenchmarkConst.EXCHANGE_VALUE: ExchangeValuePerturbation
    }

    @staticmethod
    def create(api_name: str):
        perturbation = PerturbationFactory.perturbations.get(Config.pert_type)
        if perturbation:
            return perturbation(api_name)
        else:
            raise Exception(f'{Config.pert_type} is a invalid perturbation type')
