# Copyright (c) 2024-2024, Huawei Technologies Co., Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn

from collections import namedtuple

from msprobe.pytorch.bench_fuctions.npu_fusion_attention import softmax_forward\
    , softmax_grad, broadcast_kv, calculate_qk, fusion_attention_forward, fusion_attention_backward, parse_bsnd_args\
    , convert_from_bnsd, convert_to_bnsd, convert_from_bsnd, convert_to_bsnd, generate_attn_mask,\
    generate_kv, rebuid_softmax_by_qkv, rebuild_softmax_by_max_sum,\
    get_head_num, get_input_layout, npu_fusion_attention_forward_patch, npu_fusion_attention_backward_patch

GTYPE = torch.float64  # arm host必须选择float64，x86环境选择float32即可，64也行。arm计算很慢，s=8k的场景建议使用x86
SOFTMAX_BUILD_MODE = "QKV"  # "MAX_SUM"

FaForwardParams = namedtuple("FaForwardParams",
                             ["q", "k", "v", "drop_mask", "attn_mask", "pse", "scalar_value", "keep_prob"])
FaBackwardParams = namedtuple("FaBackwardParams",
                              ["dx", "q", "k", "v", "softmax_res", "drop_mask", "pse", "scalar_value", "keep_prob"])
RebuildSoftmaxParams = namedtuple("RebuildSoftmaxParams",
                                  ["q", "k", "attn_mask", "pse", "scalar_value", "softmax_max", "softmax_sum"])


class FlashAttentionScore(nn.Module):
    def __init__(self):
        super(FlashAttentionScore, self).__init__()
        # You can initialize any parameters here if necessary

    def forward(self, *inputs, **kwargs):
        # Extract the inputs for the attention calculation
        new_args, dims_kwargs, new_kwargs = npu_fusion_attention_forward_patch(*inputs, **kwargs)
        query, key, value = new_args[0], new_args[1], new_args[2]

        input_layout = get_input_layout(*inputs, **kwargs)

        n1 = dims_kwargs.get("n1")
        n2 = dims_kwargs.get("n2")
        s1 = dims_kwargs.get("s1")
        s2 = dims_kwargs.get("s2")
        b = dims_kwargs.get("b")
        dtype = dims_kwargs.get("dtype")
        attn_mask = new_kwargs.get("attn_mask")
        keep_prob = new_kwargs.get("keep_prob")
        sparse_mode = new_kwargs.get("sparse_mode")
        pre_tockens = new_kwargs.get("pre_tockens")
        next_tockens = new_kwargs.get("next_tokens")
        pse = new_kwargs.get("real_shift")
        scalar_value = new_kwargs.get("scalar_value")

        args_temp = [sparse_mode, attn_mask, b, n1, s1, s2, pre_tockens, next_tockens, dtype]

        attn_mask = generate_attn_mask(*args_temp)
        query = convert_to_bnsd(query, n1, input_layout)
        key = convert_to_bnsd(key, n2, input_layout)
        value = convert_to_bnsd(value, n2, input_layout)

        forward_params = FaForwardParams(
            q=query,
            k=key,
            v=value,
            drop_mask=None,
            attn_mask=attn_mask,
            pse=pse,
            scalar_value=scalar_value,
            keep_prob=keep_prob
        )

        out_golden, softmax_max, softmax_sum = fusion_attention_forward(forward_params)

        # If output dimension is 5, reshape accordingly
        if out_golden.dim() == 5:
            out_golden = out_golden.reshape(out_golden.size(0),
                                            out_golden.size(1) * out_golden.size(2),
                                            out_golden.size(3), out_golden.size(4))

        out_golden = convert_from_bnsd(out_golden, input_layout)

        # Ensure the output matches the desired layout
        out_golden = out_golden.cpu(), softmax_max.repeat(1, 1, 1, 8).cpu(), softmax_sum.repeat(1, 1, 1, 8).cpu()

        return out_golden

    def backward(self, *inputs, **kwargs):
        # The backward pass will be similar to what was described for the gradient computation
        new_args, dims_kwargs, new_kwargs = npu_fusion_attention_backward_patch(*inputs, **kwargs)
        query, key, value, dx, input_layout = new_args[0], new_args[1], new_args[2], new_args[3], new_args[5]
        n1 = dims_kwargs.get("n1")
        n2 = dims_kwargs.get("n2")
        s1 = dims_kwargs.get("s1")
        s2 = dims_kwargs.get("s2")
        b = dims_kwargs.get("b")
        dtype = dims_kwargs.get("dtype")
        attn_mask = new_kwargs.get("attn_mask")
        keep_prob = new_kwargs.get("keep_prob")
        sparse_mode = new_kwargs.get("sparse_mode")
        pre_tockens = new_kwargs.get("pre_tockens")
        next_tockens = new_kwargs.get("next_tockens")
        pse = new_kwargs.get("pse")
        softmax_max = new_kwargs.get("softmax_max")
        softmax_sum = new_kwargs.get("softmax_sum")
        scalar_value = new_kwargs.get("scalar_value")

        args_temp = [sparse_mode, attn_mask, b, n1, s1, s2, pre_tockens, next_tockens, dtype]
        attn_mask = generate_attn_mask(*args_temp)

        query = convert_to_bnsd(query, n1, input_layout)
        dx = convert_to_bnsd(dx, n1, input_layout)
        key = convert_to_bnsd(key, n2, input_layout)
        value = convert_to_bnsd(value, n2, input_layout)

        k_new, v_new = generate_kv(key, value, n1, n2)

        if SOFTMAX_BUILD_MODE == "QKV":
            softmax_res = rebuid_softmax_by_qkv(query, k_new, attn_mask, pse, scalar_value)
        else:
            softmax_params = RebuildSoftmaxParams(query, k_new, attn_mask, pse, scalar_value, softmax_max, softmax_sum)
            softmax_res = rebuild_softmax_by_max_sum(softmax_params)

        backward_params = FaBackwardParams(dx, query, k_new, v_new, softmax_res, None, pse, scalar_value, keep_prob)
        dq, dk, dv = fusion_attention_backward(backward_params)

        # Reshape as needed
        if dq.dim() == 5:
            dq = dq.reshape(dq.size(0), dq.size(1) * dq.size(2), dq.size(3), dq.size(4))
        if dk.dim() == 5:
            dk = dk.reshape(dk.size(0), dk.size(1) * dk.size(2), dk.size(3), dk.size(4))
        if dv.dim() == 5:
            dv = dv.reshape(dv.size(0), dv.size(1) * dv.size(2), dv.size(3), dv.size(4))

        dq = convert_from_bnsd(dq, input_layout)
        dk = convert_from_bnsd(dk, input_layout)
        dv = convert_from_bnsd(dv, input_layout)

        return dq.cpu(), dk.cpu(), dv.cpu()
