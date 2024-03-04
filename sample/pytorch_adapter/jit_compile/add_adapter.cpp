#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

using torch::autograd::AutogradContext;
using torch::autograd::Function;
using tensor_list = std::vector<at::Tensor>;
using namespace at;

extern "C" void add_custom_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y, uint8_t *z);

// 为NPU设备注册前向实现
at::Tensor my_add_impl_npu(const at::Tensor &self, const at::Tensor &other)
{
    // 创建输出内存
    at::Tensor result = at::Tensor(self);
    // 将pytorch中的结构翻译成为CANN认识的数据类型和结构
    // 1. (重要)通过对tensor的shape分析，选择合适的tiling（该算子为了简化，固定了tiling，只有特定shape下计算才正确）
    // 2. 对数据类型和格式转换  -- 此处无需数据格式处理，直接使用
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto x = self.storage().data();
    auto y = other.storage().data();
    auto z = result.storage().data();

    uint32_t blockDim = 8;
    auto callback = [stream, blockDim, x, y, z]() -> int {
        add_custom_do(blockDim, stream, (uint8_t *)x, (uint8_t *)y, (uint8_t *)z);
        return 0;  // 此处可以通过某种方式获取算子执行结果，还未实现
    };
    // 下发算子
    at_npu::native::OpCommand cmd;
    cmd.Name("my_add").SetCustomHandler(callback).Run();
    return result;
}

// 为NPU设备注册反向实现
std::tuple<at::Tensor, at::Tensor> my_add_backward_impl_npu(const at::Tensor &self)
{
    at::Tensor result = at::Tensor(self);  // 创建输出内存

    return {result, result};
}

// 为Meta设备注册前向实现
at::Tensor my_add_impl_meta(const at::Tensor &self, const at::Tensor &other)
{
    return empty_like(self);
}

// 为Meta设备注册反向实现
std::tuple<at::Tensor, at::Tensor> my_add_backward_impl_meta(const at::Tensor &self)
{
    auto result = empty_like(self);
    return std::make_tuple(result, result);
}

// 寻找注册在该op上的不同设备的实现
at::Tensor my_add_impl(const at::Tensor &self, const at::Tensor &other)
{
    static auto op =
        torch::Dispatcher::singleton().findSchemaOrThrow("myaten::my_add", "").typed<decltype(my_add_impl)>();
    return op.call(self, other);
}
// 寻找注册在该op上的不同设备的实现
std::tuple<at::Tensor, at::Tensor> my_add_backward_impl(const at::Tensor &self)
{
    static auto op = torch::Dispatcher::singleton()
                         .findSchemaOrThrow("myaten::my_add_backward", "")
                         .typed<decltype(my_add_backward_impl)>();
    return op.call(self);
}

// 在myaten命名空间里注册my_add和my_add_backward两个schema
TORCH_LIBRARY(myaten, m)
{
    m.def("my_add(Tensor self, Tensor other) -> Tensor");
    m.def("my_add_backward(Tensor self) -> (Tensor, Tensor)");
}

// 通过继承torch::autograd::Function类实现前反向绑定
class MyAddFunction : public torch::autograd::Function<MyAddFunction> {
public:
    static at::Tensor forward(AutogradContext *ctx, at::Tensor self, at::Tensor other)
    {
        at::AutoDispatchBelowADInplaceOrView guard;
        return my_add_impl(self, other);
    }

    static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs)
    {
        auto grad_output = grad_outputs[0];
        auto result = my_add_backward_impl(grad_output);
        return {std::get<0>(result), std::get<1>(result)};
    }
};

at::Tensor my_add_impl_autograd(const at::Tensor &self, const at::Tensor &other)
{
    return MyAddFunction::apply(self, other);
}

// 给op绑定NPU的自动求导实现
// 如果是pytorch 2.1以下的版本，AutogradPrivateUse1需要改成AutogradXLA
TORCH_LIBRARY_IMPL(myaten, AutogradPrivateUse1, m)
{
    m.impl("my_add", &my_add_impl_autograd);
}

// 为NPU设备注册前反向实现
// NPU设备在pytorch 2.1及以上版本使用的设备名称是PrivateUse1，在2.1以下版本用的是XLA，如果是2.1以下版本PrivateUse1需要改成XLA
TORCH_LIBRARY_IMPL(myaten, PrivateUse1, m)
{
    m.impl("my_add", &my_add_impl_npu);
    m.impl("my_add_backward", &my_add_backward_impl_npu);
}

// 为Meta设备注册前反向实现
TORCH_LIBRARY_IMPL(myaten, Meta, m)
{
    m.impl("my_add", &my_add_impl_meta);
    m.impl("my_add_backward", &my_add_backward_impl_meta);
}

// 通过pybind将c++接口和python接口绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("my_add", &my_add_impl_autograd, "x + y");
}
