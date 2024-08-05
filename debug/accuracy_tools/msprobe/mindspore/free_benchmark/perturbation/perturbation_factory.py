from msprobe.core.common.const import MsFreeBenchmarkConst
from msprobe.mindspore.free_benchmark.common.config import Config
from .add_noise import AddNoisePerturbation
from .bit_noise import BitNoisePerturbation
from .no_change import NoChangePerturbation
from .improve_precision import ImprovePrecisionPerturbation


class PerturbationFactory:
    """
    扰动工厂类

    """
    perturbations = {
        MsFreeBenchmarkConst.IMPROVE_PRECISION: ImprovePrecisionPerturbation,
        MsFreeBenchmarkConst.ADD_NOISE: AddNoisePerturbation,
        MsFreeBenchmarkConst.BIT_NOISE: BitNoisePerturbation,
        MsFreeBenchmarkConst.NO_CHANGE: NoChangePerturbation,
    }

    @staticmethod
    def create(api_name: str):
        perturbation = PerturbationFactory.perturbations.get(Config.pert_type)
        if perturbation:
            return perturbation(api_name)
        else:
            raise Exception(f'{Config.pert_type} is a invalid perturbation type')
