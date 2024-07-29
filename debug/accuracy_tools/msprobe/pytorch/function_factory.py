from msprobe.pytorch.common.utils import logger


class Register(dict):
    def __init__(self, *args, **kwargs):
        super(Register, self).__init__(*args, **kwargs)
        self._dict = {}

    def __call__(self, target):
        return self.register(target)

    def __setitem__(self, key, value):
        self._dict[key] = value

    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def __str__(self):
        return str(self._dict)

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def items(self):
        return self._dict.items()

    def register(self, target):

        def add_register_item(key, value):
            if key in self._dict:
                logger.warning(f"{value.__name__} has been registered before, so we will overriden it.")
            self[key] = value
            return value

        if callable(target):
            return add_register_item(target.__name__, target)
        else:
            raise Exception(f"The func {target} is not callable.")


npu_custom_functions = Register()
npu_custom_grad_functions = Register()

from msprobe.pytorch.bench_functions.apply_adam_w import npu_apply_adam_w
from msprobe.pytorch.bench_functions.confusion_transpose import npu_confusion_transpose, \
    npu_confusion_transpose_backward
from msprobe.pytorch.bench_functions.fast_gelu import fast_gelu, npu_fast_gelu_backward
from msprobe.pytorch.bench_functions.layer_norm_eval import npu_layer_norm_eval
from msprobe.pytorch.bench_functions.linear import npu_linear, npu_linear_backward
from msprobe.pytorch.bench_functions.matmul_backward import matmul_backward
from msprobe.pytorch.bench_functions.npu_fusion_attention import softmax_forward, softmax_grad, broadcast_kv, \
    calculate_qk, fusion_attention_forward, fusion_attention_backward, parse_bsnd_args, convert_from_bnsd, \
    convert_to_bnsd, generate_atten_mask, generate_kv, rebuid_softmax_by_qkv, rebuild_softmax_by_max_sum, \
    npu_fusion_attention_forward_patch, npu_fusion_attention_backward_patch, npu_fusion_attention, \
    npu_fusion_attention_grad
from msprobe.pytorch.bench_functions.rms_norm import npu_rms_norm, npu_rms_norm_backward
from msprobe.pytorch.bench_functions.rotary_mul import npu_rotary_mul, npu_rotary_mul_backward
from msprobe.pytorch.bench_functions.scaled_mask_softmax import npu_scaled_masked_softmax, \
    npu_scaled_masked_softmax_backward
from msprobe.pytorch.bench_functions.swiglu import npu_swiglu, npu_swiglu_backward, swish_grad, swish
