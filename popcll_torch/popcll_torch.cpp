#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

torch::Tensor popcll_torch_cuda_forward(
    torch::Tensor input);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_LONG(x) AT_ASSERTM(x.type().scalarType() == at::ScalarType::Long, #x " must be a long tensor")
#define CHECK_INPUT(x)   \
    CHECK_CUDA(x);       \
    CHECK_CONTIGUOUS(x); \
    CHECK_LONG(x)

torch::Tensor popcll_torch_forward(
    torch::Tensor input)
{
    CHECK_INPUT(input);

    return popcll_torch_cuda_forward(input);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("popcll", &popcll_torch_forward, "popcll_torch (CUDA)");
}