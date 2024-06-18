class PerturbationMode:
    ADD_NOISE = "add_noise"
    CHANGE_VALUE = "change_value"
    IMPROVE_PRECISION = "improve_precision"
    NO_CHANGE = "no_change"
    BIT_NOISE = "bit_noise"
    TO_CPU = "to_cpu"


class DeviceType:
    NPU = "npu"
    CPU = "cpu"


class FuzzThreshold:
    BF16_THD = 1e-4
    F16_THD = 1e-6
    F32_THD = 1e-8
    F64_THD = 1e-16


class NormType:
    ONE_NORM = (1, "one_norm")
    TWO_NORM = (2, "two_norm")
    ENDLESS_NORM = (3, "endless_norm")


class HandlerType:
    CHECK = "check"
    PREHEAT = "preheat"
    FIX = "fix"


class FuzzLevel:
    BASE_LEVEL = "L1"
    ADV_LEVEL = "L2"
    REAL_LEVEL = "L3"
